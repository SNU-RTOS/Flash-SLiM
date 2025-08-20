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


struct event_t {
    u64 ts;          // ns, monotonic
    u32 pid;         // tgid
    int kind;        // 0 = start, 1 = end
    char phase[64];  // update at end or check
};

struct pf_stat_t {
    u64 start_ns;         // PF 진입 시각 (0이면 비활성)
    u64 major_cnt;  
    u64 minor_cnt;
    u64 minor_retry_cnt;
    u64 major_ns;           // 누적 major 처리시간(ns)
    u64 minor_ns;           // 누적 minor (NO_PAGE) 처리시간(ns)
    u64 minor_retry_ns;    // 누적 minor (RETRY) 처리시간(ns)
    u64 readahead_cnt;
};

// ── Read/Write Syscall Stats ──
struct rw_stat_t {
    u64 read_cnt;
    u64 read_ns;
    u64 read_bytes;

    u64 write_cnt;
    u64 write_ns;
    u64 write_bytes;

    u64 start_ns;   // sys_enter timestamp 저장용
};


/* ---------- Minimal tracepoint ctx for sys_enter/exit ---------- */
struct trace_event_raw_sys_enter {
    unsigned long long unused;
    long id;                   // __syscall_nr
    unsigned long args[6];
};
struct trace_event_raw_sys_exit {
    unsigned long long unused;
    long id;                   // __syscall_nr
    long ret;                  // return value
};




/* 01 BPF Maps */
/* ---------- Maps ---------- */
BPF_PERF_OUTPUT(events);

// Phase 활성화 여부: PID(tgid) -> start_ts
BPF_HASH(phase_ts_pid, u32, u64);

// Page Fault 통계: KEY = u64(pid_tgid)
BPF_HASH(pfstat, u64, struct pf_stat_t);

// Read/Write 통계: KEY = u64(pid_tgid)
BPF_HASH(rwstat, u64, struct rw_stat_t);

// ── Readahead 카운트: PID 키 ──
BPF_HASH(readahead_count, u32, u64);

// ── Block I/O Start/Stats ──
struct block_start_t {
    u64 start_ns;
    u64 pid_tgid;   // attribute to PID captured at start time
    char op;        // 'R' or 'W' (from rwbs[0])
    u64 bytes;      // request size captured at start
};

struct block_stat_t {
    u64 time_ns;       // total elapsed between issue and complete
    u64 count;         // number of completed requests
    u64 read_bytes;    // bytes completed for reads
    u64 write_bytes;   // bytes completed for writes
};

// Key on (dev, sector) like the bpftrace prototype
struct bio_key_t { u64 dev; u64 sector; };
BPF_HASH(block_start, struct bio_key_t, struct block_start_t);

// Accumulate per pid_tgid (sum to tgid at phase end)
BPF_HASH(blockstat, u64, struct block_stat_t);

// ── CPU sched stats (runtime/wait) per pid_tgid ──
struct sched_stat_t {
    u64 runtime_ns;
    u64 wait_ns;
};
BPF_HASH(schedstat, u64, struct sched_stat_t);

/* 02 BPF Helpers */
static __always_inline u32 get_tgid(void) {
    return (u32)(bpf_get_current_pid_tgid() >> 32);
}
static __always_inline u64 get_pid_tgid(void) {
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

/* ---------- sys_read / sys_write (tracepoints) ---------- */
int tracepoint__syscalls__sys_enter_read(struct trace_event_raw_sys_enter *ctx) {
    u64 pid_tgid = get_pid_tgid();
    u32 tgid = (u32)(pid_tgid >> 32);
    if (!phase_ts_pid.lookup(&tgid)) return 0;

    struct rw_stat_t zero = {};
    struct rw_stat_t *s = rwstat.lookup_or_try_init(&pid_tgid, &zero);
    if (!s) return 0;

    s->start_ns = bpf_ktime_get_ns();
    return 0;
}

int tracepoint__syscalls__sys_exit_read(struct trace_event_raw_sys_exit *ctx) {
    u64 pid_tgid = get_pid_tgid();
    struct rw_stat_t *s = rwstat.lookup(&pid_tgid);
    if (!s || s->start_ns == 0) return 0;

    u64 delta = bpf_ktime_get_ns() - s->start_ns;
    s->read_cnt++;
    s->read_ns += delta;

    long ret = ctx->ret; // bytes read or error(<0)
    if (ret > 0) s->read_bytes += (u64)ret;

    s->start_ns = 0;
    return 0;
}

int tracepoint__syscalls__sys_enter_write(struct trace_event_raw_sys_enter *ctx) {
    u64 pid_tgid = get_pid_tgid();
    u32 tgid = (u32)(pid_tgid >> 32);
    if (!phase_ts_pid.lookup(&tgid)) return 0;

    struct rw_stat_t zero = {};
    struct rw_stat_t *s = rwstat.lookup_or_try_init(&pid_tgid, &zero);
    if (!s) return 0;

    s->start_ns = bpf_ktime_get_ns();
    return 0;
}

int tracepoint__syscalls__sys_exit_write(struct trace_event_raw_sys_exit *ctx) {
    u64 pid_tgid = get_pid_tgid();
    struct rw_stat_t *s = rwstat.lookup(&pid_tgid);
    if (!s || s->start_ns == 0) return 0;

    u64 delta = bpf_ktime_get_ns() - s->start_ns;
    s->write_cnt++;
    s->write_ns += delta;

    long ret = ctx->ret; // bytes written or error(<0)
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



/* ---------- tracepoint: block_io_start (start) ---------- */
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

/* ---------- tracepoint: block_io_done (done) ---------- */
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

/* ---------- sched tracepoints: runtime / wait ---------- */
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
    total_non_io_ms: int = 0
    total_io_ms: int = 0
    read_ms: int = 0
    write_ms: int = 0
    pf_ms: int = 0
    major_pf_ms: int = 0
    minor_pf_ms: int = 0
    minor_retry_pf_ms: int = 0
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


def _read_pfstat_sum_and_clear(pfstat_map, pid):
    """
    pfstat_map: b["pfstat"], key = u64(pid_tgid)
    pid       : tgid(u32)
    반환: dict(major_cnt, minor_cnt, major_ns, minor_ns, readahead_cnt)
    동작: 상위 32비트가 pid인 엔트리만 합산 후 삭제
    """
    total = {
        "major_cnt": 0,
        "minor_cnt": 0,
        "minor_retry_cnt": 0,
        "major_ns": 0,
        "minor_ns": 0,
        "minor_retry_ns": 0,
        "readahead_cnt": 0,
    }
    to_delete = []
    for k, v in pfstat_map.items():
        key_u64 = _to_int(k)
        tgid = (key_u64 >> 32) & 0xFFFFFFFF
        if tgid != pid:
            continue

        if _is_percpu_leaf(v):
            for leaf in v:
                total["major_cnt"] += int(leaf.major_cnt)
                total["minor_cnt"] += int(leaf.minor_cnt)
                total["minor_retry_cnt"] += int(leaf.minor_retry_cnt)
                total["major_ns"] += int(leaf.major_ns)
                total["minor_ns"] += int(leaf.minor_ns)
                total["minor_retry_ns"] += int(leaf.minor_retry_ns)
                total["readahead_cnt"] += int(leaf.readahead_cnt)

        else:
            total["major_cnt"] += int(v.major_cnt)
            total["minor_cnt"] += int(v.minor_cnt)
            total["minor_retry_cnt"] += int(v.minor_retry_cnt)
            total["major_ns"] += int(v.major_ns)
            total["minor_ns"] += int(v.minor_ns)
            total["minor_retry_ns"] += int(v.minor_retry_ns)
            total["readahead_cnt"] += int(v.readahead_cnt)

        to_delete.append(k)

    for k in to_delete:
        try:
            del pfstat_map[k]
        except Exception:
            pass
    return total


def _read_rwstat_sum_and_clear(rwstat_map, pid):
    """
    rwstat_map: b["rwstat"], key = u64(pid_tgid)
    pid       : tgid(u32)
    """
    total = {
        "read_cnt": 0,
        "read_ns": 0,
        "read_bytes": 0,
        "write_cnt": 0,
        "write_ns": 0,
        "write_bytes": 0,
    }
    to_delete = []
    for k, v in rwstat_map.items():
        key_u64 = _to_int(k)
        tgid = (key_u64 >> 32) & 0xFFFFFFFF
        if tgid != pid:
            continue

        if _is_percpu_leaf(v):
            for leaf in v:
                total["read_cnt"] += int(leaf.read_cnt)
                total["read_ns"] += int(leaf.read_ns)
                total["read_bytes"] += int(leaf.read_bytes)
                total["write_cnt"] += int(leaf.write_cnt)
                total["write_ns"] += int(leaf.write_ns)
                total["write_bytes"] += int(leaf.write_bytes)
        else:
            total["read_cnt"] += int(v.read_cnt)
            total["read_ns"] += int(v.read_ns)
            total["read_bytes"] += int(v.read_bytes)
            total["write_cnt"] += int(v.write_cnt)
            total["write_ns"] += int(v.write_ns)
            total["write_bytes"] += int(v.write_bytes)

        to_delete.append(k)

    for k in to_delete:
        try:
            del rwstat_map[k]
        except Exception:
            pass
    return total


def _read_blockstat_sum_and_clear(blockstat_map, pid):
    """
    blockstat_map: b["blockstat"], key = u64(pid_tgid)
    pid         : tgid(u32)
    반환: dict(time_ns, count, read_bytes, write_bytes)
    동작: 상위 32비트가 pid인 엔트리만 합산 후 삭제
    """
    total = {
        "time_ns": 0,
        "count": 0,
        "read_bytes": 0,
        "write_bytes": 0,
    }
    to_delete = []
    for k, v in blockstat_map.items():
        key_u64 = _to_int(k)
        tgid = (key_u64 >> 32) & 0xFFFFFFFF
        if tgid != pid:
            continue

        if _is_percpu_leaf(v):
            for leaf in v:
                total["time_ns"] += int(leaf.time_ns)
                total["count"] += int(leaf.count)
                total["read_bytes"] += int(leaf.read_bytes)
                total["write_bytes"] += int(leaf.write_bytes)
        else:
            total["time_ns"] += int(v.time_ns)
            total["count"] += int(v.count)
            total["read_bytes"] += int(v.read_bytes)
            total["write_bytes"] += int(v.write_bytes)

        to_delete.append(k)

    for k in to_delete:
        try:
            del blockstat_map[k]
        except Exception:
            pass
    return total


def _read_sched_sum_and_clear(sched_map, pid):
    """
    sched_map: b["schedstat"], key = u64(pid_tgid)
    pid      : tgid(u32)
    반환: dict(runtime_ns, wait_ns)
    """
    total = {"runtime_ns": 0, "wait_ns": 0}
    to_delete = []
    for k, v in sched_map.items():
        key_u64 = _to_int(k)
        tgid = (key_u64 >> 32) & 0xFFFFFFFF
        if tgid != pid:
            continue
        if _is_percpu_leaf(v):
            for leaf in v:
                total["runtime_ns"] += int(leaf.runtime_ns)
                total["wait_ns"] += int(leaf.wait_ns)
        else:
            total["runtime_ns"] += int(v.runtime_ns)
            total["wait_ns"] += int(v.wait_ns)
        to_delete.append(k)
    for k in to_delete:
        try:
            del sched_map[k]
        except Exception:
            pass
    return total


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

    major_pf_ms = int(major_pf_ns // 1e6)
    minor_pf_ms = int(minor_pf_ns // 1e6)
    minor_retry_pf_ms = int(minor_pf_retry_ns // 1e6)

    pf_ms = int(major_pf_ms + minor_pf_ms + minor_retry_pf_ms)

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

    read_ms = int(sys_read_ns // 1e6)
    write_ms = int(sys_write_ns // 1e6)
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

    total_io_ms = pf_ms + read_ms + write_ms

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
        pf_ms=pf_ms,
        major_pf_ms=major_pf_ms,
        minor_pf_ms=minor_pf_ms,
        minor_retry_pf_ms=minor_retry_pf_ms,
        avg_major_us=avg_major_us,
        avg_minor_us=avg_minor_us,
        avg_minor_retry_us=avg_minor_retry_us,
        major_fault_count=major_pf_cnt,
        minor_fault_count=minor_pf_cnt,
        minor_fault_retry_count=minor_pf_retry_cnt,
        readahead_count=readahead_cnt,
        total_non_io_ms=wall_ms - total_io_ms,
        total_io_ms=total_io_ms,
        read_ms=read_ms,
        write_ms=write_ms,
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
    print(f"    - Non I/O Handling Time (Estimated)     : {rec.total_non_io_ms} (ms)")
    print(f"    - I/O Handling Time                     : {rec.total_io_ms} (ms)")
    print(f"        - Read Syscall                      : {rec.read_ms} (ms)")
    print(f"        - Write Syscall                     : {rec.write_ms} (ms)")
    print(f"        - PageFault                         : {rec.pf_ms} (ms)")
    print(f"            - Major PageFault               : {rec.major_pf_ms} (ms)")
    print(f"            - Minor PageFault (NOPAGE)      : {rec.minor_pf_ms} (ms)")
    print(f"            - Minor PageFault (RETRY)       : {rec.minor_retry_pf_ms} (ms)")
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
    print(f" Total Major PageFault Time                 : {rec.major_pf_ms} (ms)")
    print(f" Total Minor PageFault Time (NOPAGE)        : {rec.minor_pf_ms} (ms)")
    print(f" Total Minor PageFault Time (RETRY)         : {rec.minor_retry_pf_ms} (ms)")
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
    print(f" Total Read Syscall Time                    : {rec.read_ms} (ms)")
    print(f" Total Write Syscall Time                   : {rec.write_ms} (ms)")
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
