#!/usr/bin/env python3
from bcc import BPF, USDT
import atexit, signal, sys, re


binary_path = "./output/text_generator_main"

# ======== Config ========
SHOW_SEQUENCE = False  # 시퀀스 테이블이 필요할 때만 True

# ======== helpers  ========
_nat_tok = re.compile(r"\d+|\D+")


def naturalsort_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in _nat_tok.findall(s)]


# ======== eBPF text ========
BPF_TEXT = r"""
#include <uapi/linux/ptrace.h>

// ── VM_FAULT_* 플래그 (커널 버전에 따라 상수값 다를 수 있음) ──
#define VM_FAULT_MAJOR   4
#define VM_FAULT_NOPAGE  256
#define VM_FAULT_LOCKED  512
#define VM_FAULT_RETRY   1024
// NOTE: These VM_FAULT_* constants may vary across kernel versions.
//       Verify these against your target kernel headers if you see
//       unexpected classification of page-fault types.


/* ------------------------------------------------------------------
 * BPF type & map declarations
 * - This section groups all C struct/type definitions first, then the
 *   perf outputs and BPF maps. The comments explain key types and map
 *   key/value semantics so new readers can follow what is collected.
 * ------------------------------------------------------------------ */

/* --- Event type sent to userspace via perf buffer ---
 * event_t: simple notification for phase START/END
 *  - ts   : timestamp in ns (monotonic)
 *  - pid  : tgid (process id) packed as u32 in userspace
 *  - kind : 0=start, 1=end
 *  - phase: null-terminated phase name (user-provided)
 *
 * NOTE: `phase` is limited to 64 bytes; long strings will be truncated
 *       when read into this struct (userspace should handle truncation).
 */
struct event_t {
    u64 ts;          // ns, monotonic
    u32 pid;         // tgid
    int kind;        // 0 = start, 1 = end
    char phase[64];  // update at end or check
};


/* --- Page-fault accounting (per TID entry; summed to TGID in Python) ---
 * pf_stat_t accumulates counts and elapsed ns for different PF types.
 * Note: keys in the pfstat map are u64 pid_tgid (upper 32 bits = tgid,
 * lower 32 bits = tid). The reader filters by tgid when summing.
 */
struct pf_stat_t {
    u64 start_ns;         // PF entry ts (0 means inactive)
    u64 major_cnt;        // major faults count
    u64 minor_cnt;        // minor faults (no page) count
    u64 minor_retry_cnt;  // minor-retry faults count
    u64 major_ns;         // accumulated major handling time (ns)
    u64 minor_ns;         // accumulated minor (NOPAGE) handling time (ns)
    u64 minor_retry_ns;   // accumulated minor (RETRY) handling time (ns)
    u64 readahead_cnt;    // readahead invocation count
};


/* --- Read/Write syscall timing (per TID entry) ---
 * - start_ns is used to record sys_enter timestamp; exit adds delta.
 * - map key is u64 pid_tgid (tid-scoped) to avoid races between threads.
 */
struct rw_stat_t {
    u64 read_cnt;
    u64 read_ns;
    u64 read_bytes;

    u64 write_cnt;
    u64 write_ns;
    u64 write_bytes;

    u64 start_ns;   // sys_enter timestamp storage
};


/* --- Block I/O helper structs ---
 * block_start_t : stored at issue time keyed by bio (dev,sector)
 *  - pid_tgid records the originating pid/tid so completion can be
 *    attributed back to the issuing thread.
 * block_stat_t  : accumulator per pid_tgid for completed requests
 * block_io_interval_t : optional interval payload (perf) used for
 *    more advanced post-processing (interval-level events).
 */

struct block_start_t {
    u64 start_ns;
    u64 pid_tgid;   // pid/tid captured at start time (u64: [63:32]=tgid)
    char op;        // 'R' or 'W' (from rwbs[0])
    u64 bytes;      // request size captured at start
};

struct block_stat_t {
    u64 time_ns;       // total elapsed between issue and complete
    u64 count;         // number of completed requests
    u64 read_bytes;    // bytes completed for reads
    u64 write_bytes;   // bytes completed for writes
};

struct bio_key_t { u64 dev; u64 sector; };

struct block_io_interval_t {
    u64 pid_tgid;      // pid/tid for the interval
    u64 start_ns;      // interval start (ns)
    u64 end_ns;        // interval end (ns)
    u64 bytes;         // bytes processed in interval
    char op;           // 'R' or 'W'
};


struct sched_stat_t {
    u64 runtime_ns;
    u64 wait_ns;
};

/* --- Perf output endpoints (userspace consumers) --- */
BPF_PERF_OUTPUT(events);     // phase start/end events -> Python handler
/* PERF OUTPUT declared for detailed intervals. Currently optional: the
 * BPF code in this file does not submit `intervals` in all code paths.
 * Keep this endpoint if you later want per-IO interval payloads to be
 * submitted to userspace; otherwise it can be removed.
 */
BPF_PERF_OUTPUT(intervals);  // optional detailed I/O intervals -> Python


/* --- BPF Maps ---
 * Naming & key conventions:
 *  - phase_ts_pid : key=u32 (tgid) -> value=u64 start timestamp
 *  - pfstat, rwstat, blockstat, schedstat : key=u64 (pid_tgid)
 *      where upper 32 bits = tgid, lower 32 bits = tid. Python code
 *      filters by tgid and sums per-thread entries into a per-process
 *      aggregate for reporting.
 *
 * Python usage hint: when iterating map keys from userspace you can
 * extract tgid with: `tgid = (key_u64 >> 32) & 0xffffffff` before
 * deciding whether to include the entry in a per-process sum.
 */
/* --- Maps: purpose + key conventions (detailed) ---
 *
 * phase_ts_pid (u32 -> u64)
 *  - Tracks whether a phase gate is active for a given process (tgid).
 *  - Key: u32 tgid (process). Value: start timestamp (ns) when logic_start
 *    was observed. Cleared at logic_end. Used as a fast membership test
 *    inside many probes to avoid unnecessary work when no phase is active.
 */
BPF_HASH(phase_ts_pid, u32, u64);

/*
 * pfstat (u64 pid_tgid -> pf_stat_t)
 *  - Per-thread (tid-scoped) page-fault accounting entries are stored here.
 *  - Key: u64 pid_tgid (upper 32 bits = tgid, lower 32 bits = tid).
 *  - Rationale: kprobe/kretprobe handlers execute on the faulting thread,
 *    so keeping per-tid entries avoids races across threads. Python later
 *    reads and reduces per-tid entries into a per-tgid summary for output.
 */
BPF_HASH(pfstat, u64, struct pf_stat_t);

/*
 * rwstat (u64 pid_tgid -> rw_stat_t)
 *  - Tracks syscall enter/exit timing for read/write per-thread.
 *  - start_ns is set on sys_enter and used on exit to compute elapsed ns.
 */
BPF_HASH(rwstat, u64, struct rw_stat_t);

/*
 * readahead_count (u32 tgid -> u64)
 *  - Simple per-process counter incremented from the ondemand_readahead kprobe.
 *  - Keyed by tgid because readahead is attributed to the process scope.
 *
 * DECLARED BUT UNUSED: this map exists for historical reasons. The current
 * implementation increments readahead counters inside `pfstat` entries
 * (pf_stat_t.readahead_cnt) and userspace reads that field. Consider
 * removing this separate map or using it consistently.
 */
BPF_HASH(readahead_count, u32, u64);

/*
 * block_start : keyed by bio (dev,sector) -> block_start_t
 *  - At BIO issue time we capture a small blob (start_ns, pid_tgid, bytes, op)
 *    indexed by the bio key. On completion we lookup using the same key.
 *  - Note: bio keys can collide or be reused by the kernel in some cases;
 *    code defensively checks lookups and deletes entries after completion.
 *  - NOTE: `bio_key_t` layout and alignment must match the kernel tracepoint
 *    usage; key reuse/collisions are mitigated by defensive lookup logic.
 */
BPF_HASH(block_start, struct bio_key_t, struct block_start_t);

/*
 * blockstat (u64 pid_tgid -> block_stat_t)
 *  - Accumulates completed block I/O statistics per issuing pid/tid.
 *  - Key: u64 pid_tgid so that completions are attributed back to the
 *    issuing thread. Python will combine per-tid entries into per-tgid
 *    aggregates when generating the report.
 */
BPF_HASH(blockstat, u64, struct block_stat_t);

/*
 * schedstat (u64 pid_tgid -> sched_stat_t)
 *  - Runtime/wait counters are accumulated via sched tracepoints and stored
 *    per pid_tgid (thread-scoped) for the same reasons as other per-thread
 *    counters. Summation into a per-process view is done in userspace.
 */
BPF_HASH(schedstat, u64, struct sched_stat_t);

/* 02 BPF Helpers */
static __always_inline u32 get_tgid(void) {
    // Return the current process id (tgid). Use this in probes where we
    // only need process-level membership (phase gate checks etc.).
    return (u32)(bpf_get_current_pid_tgid() >> 32);
}
static __always_inline u64 get_pid_tgid(void) {
    // Return the raw 64-bit pid_tgid as produced by bpf_get_current_pid_tgid().
    // Layout: upper 32 bits = tgid, lower 32 bits = tid. Many map keys use
    // this packed value so consumers can recover both process and thread ids.
    return bpf_get_current_pid_tgid(); // [63:32]=tgid, [31:0]=tid
}

/* BPF Functions */
/* ---------- USDT: logic_start / logic_end ---------- */
int trace_logic_start(struct pt_regs *ctx) {
    u64 now = bpf_ktime_get_ns();
    u64 pid_tgid = get_pid_tgid();
    u32 tgid = (u32)(pid_tgid >> 32);

    // Phase 게이트 ON
    phase_ts_pid.update(&tgid, &now);

    // pfstat/rwstat은 TID 단위 lazy-init (여기서 초기화 불필요)

    // 이벤트 알림
    struct event_t e = {};
    e.ts = now;
    e.pid = tgid;
    e.kind = 0;
    events.perf_submit(ctx, &e, sizeof(e));
    return 0;
}

int trace_logic_end(struct pt_regs *ctx) {
    u64 now = bpf_ktime_get_ns();
    u32 tgid = get_tgid();

    // Phase 게이트 OFF
    phase_ts_pid.delete(&tgid);

    // 이벤트 알림(+phase 문자열)
    struct event_t e = {};
    e.ts = now;
    e.pid = tgid;
    e.kind = 1;
    u64 addr = 0;
    bpf_usdt_readarg(1, ctx, &addr);
    bpf_probe_read_user_str(e.phase, sizeof(e.phase), (void *)addr);
    events.perf_submit(ctx, &e, sizeof(e));
    return 0;
}

/* ---------- tracepoint: syscalls (read/write) ---------- */
/* Replaced function-style tracepoint handlers with TRACEPOINT_PROBE to
 * ensure the tracepoint `args` struct is available in BPF context.
 */
TRACEPOINT_PROBE(syscalls, sys_enter_read) {
    u64 pid_tgid = get_pid_tgid();
    u32 tgid = (u32)(pid_tgid >> 32);
    if (!phase_ts_pid.lookup(&tgid)) return 0;

    struct rw_stat_t zero = {};
    struct rw_stat_t *s = rwstat.lookup_or_try_init(&pid_tgid, &zero);
    if (!s) return 0;

    s->start_ns = bpf_ktime_get_ns();
    return 0;
}

TRACEPOINT_PROBE(syscalls, sys_exit_read) {
    u64 pid_tgid = get_pid_tgid();
    struct rw_stat_t *s = rwstat.lookup(&pid_tgid);
    if (!s || s->start_ns == 0) return 0;

    u64 delta = bpf_ktime_get_ns() - s->start_ns;
    s->read_cnt++;
    s->read_ns += delta;

    long ret = args->ret; // bytes read or error(<0)
    if (ret > 0) s->read_bytes += (u64)ret;

    s->start_ns = 0;
    return 0;
}

TRACEPOINT_PROBE(syscalls, sys_enter_write) {
    u64 pid_tgid = get_pid_tgid();
    u32 tgid = (u32)(pid_tgid >> 32);
    if (!phase_ts_pid.lookup(&tgid)) return 0;

    struct rw_stat_t zero = {};
    struct rw_stat_t *s = rwstat.lookup_or_try_init(&pid_tgid, &zero);
    if (!s) return 0;

    s->start_ns = bpf_ktime_get_ns();
    return 0;
}

TRACEPOINT_PROBE(syscalls, sys_exit_write) {
    u64 pid_tgid = get_pid_tgid();
    struct rw_stat_t *s = rwstat.lookup(&pid_tgid);
    if (!s || s->start_ns == 0) return 0;

    u64 delta = bpf_ktime_get_ns() - s->start_ns;
    s->write_cnt++;
    s->write_ns += delta;

    long ret = args->ret; // bytes written or error(<0)
    if (ret > 0) s->write_bytes += (u64)ret;

    s->start_ns = 0;
    return 0;
}

/* ---------- kprobe: handle_mm_fault ---------- */
int kprobe__handle_mm_fault(struct pt_regs *ctx) {
    u64 pid_tgid = get_pid_tgid();
    u32 tgid = (u32)(pid_tgid >> 32);
    if (!phase_ts_pid.lookup(&tgid)) return 0;

    struct pf_stat_t zero = {};
    struct pf_stat_t *s = pfstat.lookup_or_try_init(&pid_tgid, &zero);
    if (!s) return 0;

    s->start_ns = bpf_ktime_get_ns();
    return 0;
}

/* ---------- kretprobe: handle_mm_fault ---------- */
int kretprobe__handle_mm_fault(struct pt_regs *ctx) {
    u64 pid_tgid = get_pid_tgid();
    struct pf_stat_t *s = pfstat.lookup(&pid_tgid);
    if (!s) return 0;

    u64 st = s->start_ns;
    if (st == 0) return 0;

    u64 delta = bpf_ktime_get_ns() - st;
    s->start_ns = 0;

    long retval = PT_REGS_RC(ctx);

    if ((retval & VM_FAULT_MAJOR) != 0) {
        s->major_cnt++;
        s->major_ns += delta;
    } else if ((retval & VM_FAULT_RETRY) != 0) { // Retry 전용
        s->minor_retry_cnt++;
        s->minor_retry_ns += delta;
    } else  { 
        // 커널에 'minor' 비트는 없음 → 'major 아님 && 에러 아님'을 minor로 취급
        s->minor_cnt++;
        s->minor_ns += delta;
    }
    
    return 0;
}

/* ---------- kprobe: ondemand_readahead ---------- */
int kprobe__ondemand_readahead(struct pt_regs *ctx) {
    u64 pid_tgid = get_pid_tgid();
    u32 tgid = (u32)(pid_tgid >> 32);
    if (!phase_ts_pid.lookup(&tgid)) return 0;

    struct pf_stat_t zero = {};
    struct pf_stat_t *s = pfstat.lookup_or_try_init(&pid_tgid, &zero);
    if (!s) return 0;

    s->readahead_cnt++;
    return 0;
}



/* ---------- tracepoint: block_io_start (start) / block_io_done (done) ---------- */
TRACEPOINT_PROBE(block, block_io_start) {
    u32 tgid = get_tgid();
    if (!phase_ts_pid.lookup(&tgid)) return 0;

    struct block_start_t st = {};
    st.start_ns = bpf_ktime_get_ns();
    u64 pid_tgid = get_pid_tgid();
    st.pid_tgid = pid_tgid;
    st.bytes = args->bytes;

    char c = 0;
    bpf_probe_read_kernel(&c, sizeof(c), &args->rwbs[0]);
    if (c == 'W') st.op = 'W';
    else st.op = 'R';

    struct bio_key_t key = { .dev = args->dev, .sector = args->sector };
    block_start.update(&key, &st);

   
    return 0;
}

TRACEPOINT_PROBE(block, block_io_done) {
    struct bio_key_t key = { .dev = args->dev, .sector = args->sector };
    struct block_start_t *st = block_start.lookup(&key);
    if (!st) return 0;

    u64 now = bpf_ktime_get_ns();
    u64 delta = now - st->start_ns;

    struct block_stat_t zero = {};
    struct block_stat_t *bs = blockstat.lookup_or_try_init(&st->pid_tgid, &zero);
    if (bs) {
        bs->time_ns += delta;
        bs->count += 1;
        if (st->op == 'W') bs->write_bytes += st->bytes;
        else bs->read_bytes += st->bytes;
    }

    block_start.delete(&key);

    
    return 0;
}

/* ---------- sched tracepoints: cpu runtime / cpu wait ---------- */
TRACEPOINT_PROBE(sched, sched_stat_runtime) {
    u32 tgid = get_tgid();
    if (!phase_ts_pid.lookup(&tgid)) return 0;
    u64 pid_tgid = get_pid_tgid();
    struct sched_stat_t zero = {};
    struct sched_stat_t *s = schedstat.lookup_or_try_init(&pid_tgid, &zero);
    if (!s) return 0;
    s->runtime_ns += args->runtime;
    return 0;
}

TRACEPOINT_PROBE(sched, sched_stat_wait) {
    u32 tgid = get_tgid();
    if (!phase_ts_pid.lookup(&tgid)) return 0;
    u64 pid_tgid = get_pid_tgid();
    struct sched_stat_t zero = {};
    struct sched_stat_t *s = schedstat.lookup_or_try_init(&pid_tgid, &zero);
    if (!s) return 0;
    s->wait_ns += args->delay;
    return 0;
}



"""

# ======== Dtypes ======
import ctypes
from collections import defaultdict
from dataclasses import dataclass, field, fields as _dc_fields
from typing import Any, Dict, Union, List

u32 = ctypes.c_uint
u64 = ctypes.c_ulonglong


@dataclass
class PhaseRaw:
    pid: int
    phase_name: str
    start_ns: int
    end_ns: int
    pf_vals: Dict[str, Any] = field(default_factory=dict)
    rw_vals: Dict[str, Any] = field(default_factory=dict)
    blk_vals: Dict[str, Any] = field(default_factory=dict)
    sched_vals: Dict[str, Any] = field(default_factory=dict)
    io_wall_time_us: int = 0

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PhaseRaw":
        kw = {}
        for f in _dc_fields(PhaseRaw):
            key = f.name
            if key == "phase_name" and "phase" in d:
                kw[key] = d.get("phase")
            else:
                kw[key] = d.get(key, getattr(f, "default", None))
        return PhaseRaw(**kw)


@dataclass
class PhaseRecord:
    phase: str = "<unknown>"
    pid: int = 0
    tid: int = 0
    wall_ms: int = 0
    wall_pid_non_io_ms: int = 0
    wall_pid_io_ms: int = 0
    wall_read_ms: int = 0
    wall_write_ms: int = 0
    wall_pid_pf_ms: int = 0
    wall_pid_major_pf_ms: int = 0
    wall_pid_minor_pf_ms: int = 0
    wall_pid_minor_retry_pf_ms: int = 0
    total_read_ms: int = 0
    total_write_ms: int = 0
    pf_ms: int = 0
    total_major_pf_ms: int = 0
    total_minor_pf_ms: int = 0
    total_minor_retry_pf_ms: int = 0
    avg_major_us: int = 0
    avg_minor_us: int = 0
    avg_minor_retry_us: int = 0
    major_fault_count: int = 0
    minor_fault_count: int = 0
    minor_fault_retry_count: int = 0
    readahead_count: int = 0
    sys_read_count: int = 0
    sys_write_count: int = 0
    avg_read_us: int = 0
    avg_write_us: int = 0
    cpu_runtime_us: int = 0
    cpu_wait_us: int = 0
    block_io_time_us: int = 0
    block_io_count: int = 0
    block_read_bytes: int = 0
    block_write_bytes: int = 0
    avg_block_io_us: int = 0
    avg_block_read_bytes: int = 0
    avg_block_write_bytes: int = 0
    io_wall_time_us: int = 0
    io_parallel_ratio: float = 0.0
    cpu_util_ratio: float = 0.0
    io_stall_ratio: float = 0.0

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PhaseRecord":
        kw = {}
        for f in _dc_fields(PhaseRecord):
            kw[f.name] = d.get(f.name, getattr(f, "default", None))
        return PhaseRecord(**kw)


# ======== Helpers and Event Handler  ========


def _to_int(x):
    """ctypes/int 혼재 안전 변환"""
    return int(x.value) if hasattr(x, "value") else int(x)


def _safe_div(a, b):
    return a // b if b else 0


def _is_percpu_leaf(v):
    return isinstance(v, list)


def _read_stat_generic(stat_map, pid, fields):
    """
    BPF 맵에서 pid에 해당하는 통계를 합산하고 삭제하는 범용 함수.

    stat_map: b["pfstat"], b["rwstat"] 등의 BPF 맵 객체
    pid     : 필터링할 tgid(u32)
    fields  : 합산할 필드 이름의 리스트 (e.g., ["major_cnt", "minor_cnt"])
    """
    # 1. 'fields' 인자를 기반으로 total 딕셔너리를 동적으로 생성합니다.
    total = {field: 0 for field in fields}
    to_delete = []

    # 2. 기존과 동일한 순회 및 tgid 필터링 로직입니다.
    for k, v in stat_map.items():
        key_u64 = _to_int(k)
        tgid = (key_u64 >> 32) & 0xFFFFFFFF
        if tgid != pid:
            continue

        # 3. 'fields' 리스트를 순회하며 값을 동적으로 합산합니다.
        def _sum_fields(target_obj):
            for field in fields:
                # getattr(obj, "field_name")은 obj.field_name과 동일하게 동작합니다.
                total[field] += int(getattr(target_obj, field, 0))

        if _is_percpu_leaf(v):
            for leaf in v:
                _sum_fields(leaf)
        else:
            _sum_fields(v)

        to_delete.append(k)

    # 4. 기존과 동일한 삭제 로직입니다.
    for k in to_delete:
        try:
            del stat_map[k]
        except Exception:
            pass

    return total


# def _read_pfstat_sum_and_clear(pfstat_map, pid):
#     # Sum totals per-TGID (as before) and also accumulate main-thread (tid==tgid).
#     fields = [
#         "major_cnt",
#         "minor_cnt",
#         "minor_retry_cnt",
#         "major_ns",
#         "minor_ns",
#         "minor_retry_ns",
#         "readahead_cnt",
#     ]
#     total = {f: 0 for f in fields}
#     main = {f: 0 for f in fields}
#     to_delete = []

#     for k, v in pfstat_map.items():
#         key_u64 = _to_int(k)
#         tgid = (key_u64 >> 32) & 0xFFFFFFFF
#         if tgid != pid:
#             continue
#         tid = key_u64 & 0xFFFFFFFF

#         print(tgid, tid, pid)  # Debug: print tgid and tid
#         def _acc(dst, obj):
#             for f in fields:
#                 dst[f] += int(getattr(obj, f, 0))

#         if _is_percpu_leaf(v):
#             for leaf in v:
#                 _acc(total, leaf)
#                 if tid == tgid:
#                     _acc(main, leaf)
#         else:
#             _acc(total, v)
#             if tid == tgid: ## TODO -> 이걸 이제 연산에서의 첫 번째 쓰레드로 받아야하는데, 찍어보니까, 진짜 메인 프로세스의 pid를 가리키고 있어, 이쪽으로 절 대 안옴.
#                 _acc(main, v)

#         to_delete.append(k)

#     for k in to_delete:
#         try:
#             del pfstat_map[k]
#         except Exception:
#             pass

#     # Return a single dict compatible with existing callers, with main_* keys added
#     res = total.copy()
#     for f in fields:
#         res["main_" + f] = main[f]

#     return res


def _read_pfstat_sum_and_clear(pfstat_map, pid):
    # Sum totals per-TGID (as before) and also accumulate main-thread (tid==tgid).
    fields = [
        "major_cnt",
        "minor_cnt",
        "minor_retry_cnt",
        "major_ns",
        "minor_ns",
        "minor_retry_ns",
        "readahead_cnt",
    ]
    total = {f: 0 for f in fields}
    to_delete = []

    # Collect per-tid accumulators so we can pick the smallest tid as "main"
    per_thread = {}  # tid -> {field: value}

    for k, v in pfstat_map.items():
        key_u64 = _to_int(k)
        tgid = (key_u64 >> 32) & 0xFFFFFFFF
        if tgid != pid:
            continue
        tid = key_u64 & 0xFFFFFFFF

        # init per-thread accumulator if missing
        if tid not in per_thread:
            per_thread[tid] = {f: 0 for f in fields}

        def _acc_into(dst, obj):
            for f in fields:
                dst[f] += int(getattr(obj, f, 0))

        if _is_percpu_leaf(v):
            for leaf in v:
                _acc_into(per_thread[tid], leaf)
        else:
            _acc_into(per_thread[tid], v)

        to_delete.append(k)

    # sum per-thread accumulators into total
    for tid_acc in per_thread.values():
        for f in fields:
            total[f] += tid_acc[f]

    # determine "main" as the smallest tid seen for this tgid (if any)
    main = {f: 0 for f in fields}
    if per_thread:
        min_tid = min(per_thread.keys())
        main = per_thread[min_tid].copy()

    for k in to_delete:
        try:
            del pfstat_map[k]
        except Exception:
            pass

    # Return a single dict compatible with existing callers, with main_* keys added
    res = total.copy()
    for f in fields:
        res["main_" + f] = main[f]

    return res

def _read_rwstat_sum_and_clear(rwstat_map, pid):
    return _read_stat_generic(
        rwstat_map,
        pid,
        [
            "read_cnt",
            "read_ns",
            "read_bytes",
            "write_cnt",
            "write_ns",
            "write_bytes",
        ],
    )


def _read_blockstat_sum_and_clear(blockstat_map, pid):
    return _read_stat_generic(
        blockstat_map,
        pid,
        ["time_ns", "count", "read_bytes", "write_bytes"],
    )


def _read_sched_sum_and_clear(sched_map, pid):
    return _read_stat_generic(sched_map, pid, ["runtime_ns", "wait_ns"])


def _capture_phase_raw_record(
    pid: int, phase_name: str, start_ns: int, end_ns: int
) -> PhaseRaw:
    """Capture raw kernel stats for a phase without heavy calculation.

    Returns a PhaseRaw containing raw counters and timestamps. This function is
    intentionally lightweight so it can be called from the perf event handler.
    """
    pf_vals = _read_pfstat_sum_and_clear(b["pfstat"], pid)
    rw_vals = _read_rwstat_sum_and_clear(b["rwstat"], pid)
    blk_vals = _read_blockstat_sum_and_clear(b["blockstat"], pid)
    sched_vals = _read_sched_sum_and_clear(b["schedstat"], pid)

    return PhaseRaw(
        pid=pid,
        phase_name=phase_name,
        start_ns=start_ns,
        end_ns=end_ns,
        pf_vals=pf_vals,
        rw_vals=rw_vals,
        blk_vals=blk_vals,
        sched_vals=sched_vals,
        io_wall_time_us=0,
    )


def on_event(cpu, data, size):
    global missing_start
    e = b["events"].event(data)
    if e.kind == 0:
        last_start_ns[e.pid] = e.ts  # e.pid = tgid
        return

    # kind == 1 (end)
    name = e.phase.decode("utf-8", "replace").rstrip("\x00") or "<unknown>"
    start = last_start_ns.get(e.pid)
    if start is None:
        missing_start += 1  # 비정상 순서 방어
        return

    # finalize duration and clear in-flight start
    dur_ns = e.ts - start
    del last_start_ns[e.pid]

    # 집계(요약 표용)
    sum_ns[name] += dur_ns
    cnt[name] += 1
    if dur_ns < min_ns[name]:
        min_ns[name] = dur_ns
    if dur_ns > max_ns[name]:
        max_ns[name] = dur_ns
    if SHOW_SEQUENCE:
        seq.append((name, dur_ns / 1e6))  # ms

    # Build structured phase record via a single function
    raw = _capture_phase_raw_record(e.pid, name, start, e.ts)
    phase_raw_records.append(raw)


# ======== Reporting ========


def _generate_record(raw_rec: PhaseRaw) -> PhaseRecord:
    """
    Build final report record from raw capture data.
    """
    pid = raw_rec.pid
    phase_name = raw_rec.phase_name
    start_ns = raw_rec.start_ns
    end_ns = raw_rec.end_ns
    pf_vals = raw_rec.pf_vals
    rw_vals = raw_rec.rw_vals
    blk_vals = raw_rec.blk_vals
    sched_vals = raw_rec.sched_vals
    io_wall_time_us = raw_rec.io_wall_time_us

    # Base metrics
    wall_ns = end_ns - start_ns
    wall_ms = int((end_ns - start_ns) / 1e6)

    # Pagefault metrics
    major_pf_cnt = int(pf_vals["major_cnt"])
    minor_pf_cnt = int(pf_vals["minor_cnt"])
    minor_pf_retry_cnt = int(pf_vals["minor_retry_cnt"])

    readahead_cnt = int(pf_vals["readahead_cnt"])

    major_pf_ns = int(pf_vals["major_ns"])
    minor_pf_ns = int(pf_vals["minor_ns"])
    minor_pf_retry_ns = int(pf_vals["minor_retry_ns"])

    total_major_pf_ms = int(major_pf_ns // 1e6)
    total_minor_pf_ms = int(minor_pf_ns // 1e6)
    total_minor_retry_pf_ms = int(minor_pf_retry_ns // 1e6)

    total_pf_ms = int(total_major_pf_ms + total_minor_pf_ms + total_minor_retry_pf_ms)

    # Use main-thread (tid==tgid) attribution provided by _read_pfstat_sum_and_clear
    main_major_ns = int(pf_vals["main_major_ns"])
    main_minor_ns = int(pf_vals["main_minor_ns"])
    main_minor_retry_ns = int(pf_vals["main_minor_retry_ns"])

    print(
        f"{main_major_ns} {main_minor_ns} {main_minor_retry_ns}"
    ) 
    
    # Wall time for page faults
    wall_pid_major_pf_ms = int(main_major_ns // 1e6)
    wall_pid_minor_pf_ms = int(main_minor_ns // 1e6)
    wall_pid_minor_retry_pf_ms = int(main_minor_retry_ns // 1e6)

    wall_pid_pf_ms = (
        wall_pid_major_pf_ms + wall_pid_minor_pf_ms + wall_pid_minor_retry_pf_ms
    )

    avg_major_us = int((major_pf_ns // (major_pf_cnt if major_pf_cnt else 1)) // 1000)
    avg_minor_us = int((minor_pf_ns // (minor_pf_cnt if minor_pf_cnt else 1)) // 1000)
    avg_minor_retry_us = int(
        (minor_pf_retry_ns // (minor_pf_retry_cnt if minor_pf_retry_cnt else 1)) // 1000
    )

    # Syscall metrics
    sys_read_ns = rw_vals["read_ns"]
    sys_write_ns = rw_vals["write_ns"]
    sys_read_count = rw_vals["read_cnt"]
    sys_write_count = rw_vals["write_cnt"]

    total_read_ms = int(sys_read_ns // 1e6)
    total_write_ms = int(sys_write_ns // 1e6)

    wall_pid_read_ms = 0
    wall_pid_write_ms = 0

    avg_read_us = int(
        (rw_vals["read_ns"] // (sys_read_count if sys_read_count else 1)) // 1e3
    )
    avg_write_us = int(
        (rw_vals["write_ns"] // (sys_write_count if sys_write_count else 1)) // 1e3
    )

    # Block I/O
    block_io_time_us = int(blk_vals["time_ns"] / 1e3)
    block_io_count = int(blk_vals["count"])
    block_read_bytes = int(blk_vals["read_bytes"])
    block_write_bytes = int(blk_vals["write_bytes"])

    avg_block_io_us = int(_safe_div(block_io_time_us, block_io_count))
    avg_block_read_bytes = int(_safe_div(block_read_bytes, block_io_count))
    avg_block_write_bytes = int(_safe_div(block_write_bytes, block_io_count))

    # CPU
    cpu_runtime_us = sched_vals["runtime_ns"] // 1_000
    cpu_wait_us = sched_vals["wait_ns"] // 1_000

    # Compute derived
    wall_pid_io_ms = wall_pid_pf_ms + wall_pid_read_ms + wall_pid_write_ms
    total_io_ms = total_pf_ms + total_read_ms + total_write_ms

    cpu_util_ratio = (cpu_runtime_us / 1_000) / wall_ms if wall_ms else 0.0

    io_stall_ratio = (io_wall_time_us / wall_ms) if wall_ms else 0.0

    io_parallel_ratio = (
        (total_io_ms / (io_wall_time_us / 1_000)) if io_wall_time_us else 0.0
    )

    rec = PhaseRecord(
        phase=phase_name,
        pid=int(pid),
        tid=0,
        wall_ms=wall_ms,
        pf_ms=total_pf_ms,
        total_major_pf_ms=total_major_pf_ms,
        total_minor_pf_ms=total_minor_pf_ms,
        total_minor_retry_pf_ms=total_minor_retry_pf_ms,
        avg_major_us=avg_major_us,
        avg_minor_us=avg_minor_us,
        avg_minor_retry_us=avg_minor_retry_us,
        major_fault_count=major_pf_cnt,
        minor_fault_count=minor_pf_cnt,
        minor_fault_retry_count=minor_pf_retry_cnt,
        readahead_count=readahead_cnt,
        wall_pid_non_io_ms=wall_ms - wall_pid_io_ms,
        wall_pid_io_ms=wall_pid_io_ms,
        wall_pid_pf_ms=wall_pid_pf_ms,
        wall_pid_major_pf_ms=wall_pid_major_pf_ms,
        wall_pid_minor_pf_ms=wall_pid_minor_pf_ms,
        wall_pid_minor_retry_pf_ms=wall_pid_minor_retry_pf_ms,
        total_read_ms=total_read_ms,
        total_write_ms=total_write_ms,
        sys_read_count=sys_read_count,
        sys_write_count=sys_write_count,
        avg_read_us=avg_read_us,
        avg_write_us=avg_write_us,
        cpu_runtime_us=cpu_runtime_us,
        cpu_wait_us=cpu_wait_us,
        block_io_time_us=block_io_time_us,
        block_io_count=block_io_count,
        block_read_bytes=block_read_bytes,
        block_write_bytes=block_write_bytes,
        avg_block_io_us=avg_block_io_us,
        avg_block_read_bytes=avg_block_read_bytes,
        avg_block_write_bytes=avg_block_write_bytes,
        io_wall_time_us=io_wall_time_us,
        io_parallel_ratio=io_parallel_ratio,
        cpu_util_ratio=cpu_util_ratio,
        io_stall_ratio=io_stall_ratio,
    )
    return rec


def _print_phase_breakdown(rec: PhaseRecord):
    """Prints the breakdown of a specific phase using PhaseRecord attributes.

    This simplifies printing by assuming callers pass a PhaseRecord built by
    `_generate_record`.
    """

    print("\n-------------------------------------------")
    print(f"[INFO] Phase {rec.phase} Report \n")
    # print(f" PID                                    : {pid}")
    # print(f" TID                                    : {tid}")

    print("-- Elapsed Time Analysis (main pid only) --")
    print(f" Wall Clock Time                            : {rec.wall_ms} (ms)")
    print(
        f"    - Non I/O Handling Time (Estimated)     : {rec.wall_pid_non_io_ms} (ms)"
    )
    print(f"    - I/O Handling Time                     : {rec.wall_pid_io_ms} (ms)")
    print(f"        - Read Syscall                      : {rec.wall_read_ms} (ms)")
    print(f"        - Write Syscall                     : {rec.wall_write_ms} (ms)")
    print(f"        - PageFault                         : {rec.wall_pid_pf_ms} (ms)")
    print(
        f"            - Major PageFault               : {rec.wall_pid_major_pf_ms} (ms)"
    )
    print(
        f"            - Minor PageFault (NOPAGE)      : {rec.wall_pid_minor_pf_ms} (ms)"
    )
    print(
        f"            - Minor PageFault (RETRY)       : {rec.wall_pid_minor_retry_pf_ms} (ms)"
    )
    print("")

    print("-- I/O Handling Stats --")
    print(f" ## PageFault ##")
    print(f" Major Fault Count                          : {rec.major_fault_count}")
    print(f" Minor Fault Count (NOPAGE)                 : {rec.minor_fault_count}")
    print(
        f" Minor Fault Count (RETRY)                  : {rec.minor_fault_retry_count}"
    )
    print(f" ReadAhead Count                            : {rec.readahead_count}")
    print("")
    print(f" Total Major PageFault Time                 : {rec.total_major_pf_ms} (ms)")
    print(f" Total Minor PageFault Time (NOPAGE)        : {rec.total_minor_pf_ms} (ms)")
    print(
        f" Total Minor PageFault Time (RETRY)         : {rec.total_minor_retry_pf_ms} (ms)"
    )
    print(f" Avg Major Fault Handling Time              : {rec.avg_major_us} (us)")
    print(f" Avg Minor Fault Handling Time (NOPAGE)     : {rec.avg_minor_us} (us)")
    print(
        f" Avg Minor Fault Handling Time (RETRY)      : {rec.avg_minor_retry_us} (us)"
    )
    print("")
    print(f" ## Syscall ##")
    print(f" Read Syscall Count                         : {rec.sys_read_count}")
    print(f" Write Syscall Count                        : {rec.sys_write_count}")
    print("")
    print(f" Total Read Syscall Time                    : {rec.total_read_ms} (ms)")
    print(f" Total Write Syscall Time                   : {rec.total_write_ms} (ms)")
    print(f" Avg Read Syscall Handling Time             : {rec.avg_read_us} (us)")
    print(f" Avg Write Syscall Handling Time            : {rec.avg_write_us} (us)")
    print("")

    print("-- Total Time --")
    print(f" Total CPU Runtime                          : {rec.cpu_runtime_us} (us)")
    print(f" Total CPU Wait Time                        : {rec.cpu_wait_us} (us)")
    print("")

    print("-- I/O Stats --")
    print(f" Total Block I/O Time                       : {rec.block_io_time_us} (us)")
    print(f" Total Block I/O Count                      : {rec.block_io_count}")
    print(
        f" Total Block I/O Read Bytes                 : {rec.block_read_bytes} (bytes)"
    )
    print(
        f" Total Block I/O Write Bytes                : {rec.block_write_bytes} (bytes)"
    )
    print(f" Avg Block I/O Time                         : {rec.avg_block_io_us} (us)")
    print(
        f" Avg Block I/O Read Bytes                   : {rec.avg_block_read_bytes} (bytes)"
    )
    print(
        f" Avg Block I/O Write Bytes                  : {rec.avg_block_write_bytes} (bytes)"
    )
    print(f" Wall I/O Time (interval merged)            : {rec.io_wall_time_us} (us)")
    print("")

    print("-- Derived Metrics --")
    print(f" CPU Utilization Ratio                      : {rec.cpu_util_ratio:.2f}")
    print(f" I/O Stall Ratio                            : {rec.io_stall_ratio:.2f}")
    print(f" I/O Parallelism Ratio                      : {rec.io_parallel_ratio:.2f}")
    print("")


def print_report():
    print("\n===== Phase Report (start–stop) =====")

    if SHOW_SEQUENCE and seq:
        print("\n-- Sequence (ms) --")
        for name, ms in seq:
            print(f"{name:>26s} : {ms:9.3f}")

    if cnt:
        print("\n-- Summary by Phase --")

        if SHOW_SEQUENCE and seq:
            items = list(dict.fromkeys(name for name, _ in seq))  # 등장 순서
        else:
            items = sorted(cnt.keys(), key=naturalsort_key)  # 자연 정렬

        print(
            f"{'Phase':26s}  {'Count':>5s}  {'Avg(ms)':>10s}  {'Min(ms)':>10s}  {'Max(ms)':>10s}  {'Total(ms)':>12s}"
        )
        for k in items:
            avg_ms = (sum_ns[k] / cnt[k]) / 1e6
            min_ms_v = min_ns[k] / 1e6
            max_ms_v = max_ns[k] / 1e6
            tot_ms = sum_ns[k] / 1e6
            print(
                f"{k:26s}  {cnt[k]:5d}  {avg_ms:10.3f}  {min_ms_v:10.3f}  {max_ms_v:10.3f}  {tot_ms:12.3f}"
            )

        # Top-K by Avg(ms) with Ratio & Cum(%)
        TOPK = 7
        avg_map = {phase: sum_ns[phase] / cnt[phase] / 1e6 for phase in cnt}
        sorted_by_avg = sorted(avg_map.items(), key=lambda x: x[1], reverse=True)
        total_avg_sum = sum(v for _, v in sorted_by_avg)

        print(f"\n-- Top-{TOPK} Phases by Avg(ms) --")
        print(f"{'Phase':<30} {'Avg(ms)':>12} {'Ratio(%)':>12} {'Cum(%)':>12}")

        cum = 0.0
        for phase, avg in sorted_by_avg[:TOPK]:
            ratio = 100.0 * avg / total_avg_sum if total_avg_sum > 0 else 0.0
            cum += ratio
            print(f"{phase:<30} {avg:12.3f} {ratio:12.1f} {cum:12.1f}")

    # Diagnostics
    if last_start_ns or missing_start:
        print("\n-- Diagnostics --")
        if last_start_ns:
            print(f"in-flight phases without END : {len(last_start_ns)}")
        if missing_start:
            print(f"END without prior START      : {missing_start}")
    print("=====================================\n")

    #  Print per-phase breakdowns in sequence
    if phase_raw_records:
        print("\n\n===== Phase Breakdown (per occurrence) =====")
        for raw_record in phase_raw_records:
            record = _generate_record(raw_record)
            _print_phase_breakdown(record)


# ======== Setup / Main ========
if __name__ == "__main__":

    # --- Attach USDT ---
    usdt = USDT(path=binary_path)
    usdt.enable_probe_or_bail("tflite_gen:logic_start", "trace_logic_start")
    usdt.enable_probe_or_bail("tflite_gen:logic_end", "trace_logic_end")

    b = BPF(text=BPF_TEXT, usdt_contexts=[usdt])

    # --- Aggregation (phase-level: start–stop) ---
    last_start_ns = {}  # pid -> start_ns

    # 누적: phase 이름별 합/횟수/최소/최대
    sum_ns = defaultdict(int)
    cnt = defaultdict(int)
    min_ns = defaultdict(lambda: 1 << 62)
    max_ns = defaultdict(int)
    seq = []  # (phase, duration_ms) 발생 순서 기록 (옵션)
    missing_start = 0

    # (NEW) per-occurrence breakdown records
    phase_raw_records = []

    # Perf buffer 등록
    b["events"].open_perf_buffer(on_event)

    # Exit hooks
    atexit.register(print_report)

    # Start tracing
    print("Tracing USDT probes... Ctrl-C to stop.")
    try:
        while True:
            b.perf_buffer_poll()
    except KeyboardInterrupt:
        pass
