#!/usr/bin/env python3
from bcc import BPF, USDT
import atexit, signal, sys, re
from collections import defaultdict

binary_path = "./output/text_generator_main"

# ======== Config ========
SHOW_SEQUENCE = False   # 시퀀스 테이블이 필요할 때만 True

# ======== Natural sort helper (precompiled) ========
_nat_tok = re.compile(r"\d+|\D+")
def naturalsort_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in _nat_tok.findall(s)]

# ======== eBPF text ========
BPF_TEXT = r"""
#include <uapi/linux/ptrace.h>


struct event_t {
    u64 ts;          // ns, monotonic
    u32 pid;         // tgid
    int kind;        // 0 = start, 1 = end
    char phase[96];  // update at end or check
};

struct pf_stat_t {
    u64 start_ns;         // PF 진입 시각 (0이면 비활성)
    u64 major_cnt;
    u64 minor_cnt;
    u64 major_ns;           // 누적 major 처리시간(ns)
    u64 minor_ns;           // 누적 minor 처리시간(ns)
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

// ── VM_FAULT_* 플래그 (커널 버전에 따라 상수값 다를 수 있음) ──
#define VM_FAULT_MAJOR   4
#define VM_FAULT_NOPAGE  256
#define VM_FAULT_LOCKED  512
#define VM_FAULT_RETRY   1024



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

    // RETRY: 실제 처리 미완 (major/minor로 세지 않음; 별도 계수 원하면 pf_stat_t에 필드 추가)
    //if (retval & VM_FAULT_RETRY) {
    //   return 0;
    //}

    if ((retval & VM_FAULT_MAJOR) != 0) {
        s->major_cnt++;
        s->major_ns += delta;
    } else {
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
"""

# ======== (NEW) Phase breakdown skeleton printer ========
def print_phase_breakdown(rec):
    """
    rec: {
      'phase': str,
      'pid': int,
      'tid': int or None,
      'wall_ms': int,
      # 아래 항목들은 추후 채워질 예정. 지금은 0 또는 빈값으로 출력.
      'total_non_io_ms': int, 'total_io_ms': int,
      'read_ms': int, 'write_ms': int, 'pf_ms': int,
      'major_pf_ms': int, 'minor_pf_ms': int,
      'major_fault_count': int, 'minor_fault_count': int,
      'readahead_count': int, 'sys_read_count': int, 'sys_write_count': int,
      'major_fault_elapsed_ns': int, 'minor_fault_elapsed_ns': int,
      'cpu_runtime_us': int, 'cpu_wait_us': int,
      'block_io_time_us': int, 'block_io_count': int,
      'block_read_bytes': int, 'block_write_bytes': int,
      'io_ratio_percent': int,
      'io_wall_time_ms': int, 'io_parallel_ratio': float,
    }
    """
    phase = rec.get('phase', '<unknown>')
    pid   = rec.get('pid', 0)
    tid   = rec.get('tid', 0)  # 현재는 수집 안 함(0)

    wall_ms         = rec.get('wall_ms', 0)
    total_non_io_ms = rec.get('total_non_io_ms', 0)
    total_io_ms     = rec.get('total_io_ms', 0)
    read_ms         = rec.get('read_ms', 0)
    write_ms        = rec.get('write_ms', 0)
    pf_ms           = rec.get('pf_ms', 0)
    major_pf_ms     = rec.get('major_pf_ms', 0)
    minor_pf_ms     = rec.get('minor_pf_ms', 0)

    major_fault_count = rec.get('major_fault_count', 0)
    minor_fault_count = rec.get('minor_fault_count', 0)
    readahead_count   = rec.get('readahead_count', 0)
    sys_read_count    = rec.get('sys_read_count', 0)
    sys_write_count   = rec.get('sys_write_count', 0)

    major_fault_elapsed_ns = rec.get('major_fault_elapsed_ns', 0)
    minor_fault_elapsed_ns = rec.get('minor_fault_elapsed_ns', 0)

    cpu_runtime_us = rec.get('cpu_runtime_us', 0)
    cpu_wait_us    = rec.get('cpu_wait_us', 0)

    block_io_time_us   = rec.get('block_io_time_us', 0)
    block_io_count     = rec.get('block_io_count', 0)
    block_read_bytes   = rec.get('block_read_bytes', 0)
    block_write_bytes  = rec.get('block_write_bytes', 0)

    io_wall_time_ms    = rec.get('io_wall_time_ms', 0)     # (추후) first start~last done
    io_parallel_ratio  = rec.get('io_parallel_ratio', 0.0) # (추후) total_io / wall_io
    io_ratio_percent   = rec.get('io_ratio_percent', 0)

    # 안전 나눗셈
    def safe_div(a, b):
        return a // b if b else 0

    avg_major_us = (major_fault_elapsed_ns // (major_fault_count if major_fault_count else 1)) // 1000
    avg_minor_us = (minor_fault_elapsed_ns // (minor_fault_count if minor_fault_count else 1)) // 1000

    avg_block_io_us    = safe_div(block_io_time_us, block_io_count)
    avg_block_read_b   = safe_div(block_read_bytes, block_io_count)
    avg_block_write_b  = safe_div(block_write_bytes, block_io_count)

    print("\n-------------------------------------------")
    print(f"[INFO] Phase {phase} Report \n")
    # print(f" PID                                    : {pid}")
    # print(f" TID                                    : {tid}")

    print("-- Elapsed Time Analysis --")
    print(f" Wall Clock Time                            : {wall_ms} (ms)")
    print(f"    - Non I/O Handling Time (Estimated)     : {total_non_io_ms} (ms)")
    print(f"    - I/O Handling Time                     : {total_io_ms} (ms)")
    print(f"        - Read Syscall                      : {read_ms} (ms)")
    print(f"        - Write Syscall                     : {write_ms} (ms)")
    print(f"        - PageFault                         : {pf_ms} (ms)")
    print(f"            - Major PageFault               : {major_pf_ms} (ms)")
    print(f"            - Minor PageFault               : {minor_pf_ms} (ms)")
    print("")

    print("-- I/O Handling Stats --")
    print(f" Major Fault Count                          : {major_fault_count}")
    print(f" Minor Fault Count                          : {minor_fault_count}")
    print(f" ReadAhead Count                            : {readahead_count}")
    print(f" Read Syscall Count                         : {sys_read_count}")
    print(f" Write Syscall Count                        : {sys_write_count}")
    print(f" Avg Major Fault Handling Time              : {avg_major_us} (us)")
    print(f" Avg Minor Fault Handling Time              : {avg_minor_us} (us)")
    print("")

    print("-- Total Time --")
    print(f" Total CPU Runtime                          : {cpu_runtime_us} (us)")
    print(f" Total CPU Wait Time                        : {cpu_wait_us} (us)")
    print("")

    print("-- I/O Stats --")
    print(f" Total Block I/O Time                       : {block_io_time_us} (us)")
    print(f" Total Block I/O Count                      : {block_io_count}")
    print(f" Total Block I/O Read Bytes                 : {block_read_bytes} (bytes)")
    print(f" Total Block I/O Write Bytes                : {block_write_bytes} (bytes)")
    print(f" Avg Block I/O Time                         : {avg_block_io_us} (us)")
    print(f" Avg Block I/O Read Bytes                   : {avg_block_read_b} (bytes)")
    print(f" Avg Block I/O Write Bytes                  : {avg_block_write_b} (bytes)")
    print(f" Wall I/O Time (Last done - first start)    : {io_wall_time_ms} (ms)")
    print(f" I/O Parallelism ratio (Total IO / Wall IO) : {io_parallel_ratio:.3f}")
    print(f" I/O Stall ratio (Wall / Wall IO)           : {io_ratio_percent} (%)")
    print("")

# ======== Event handler ========
import ctypes
u32 = ctypes.c_uint
u64 = ctypes.c_ulonglong

def _to_int(x):
    """ctypes/int 혼재 안전 변환"""
    return int(x.value) if hasattr(x, "value") else int(x)

def _is_percpu_leaf(v):
    # BPF_PERCPU_HASH면 leaf가 코어 수만큼 list로 들어옴
    return isinstance(v, list)

def _pf_leaf_add(total, leaf):
    total['major_cnt']     += int(leaf.major_cnt)
    total['minor_cnt']     += int(leaf.minor_cnt)
    total['major_ns']      += int(leaf.major_ns)
    total['minor_ns']      += int(leaf.minor_ns)
    total['readahead_cnt'] += int(leaf.readahead_cnt)

def _read_pfstat_sum_and_clear_per_tid(pfstat_map, pid):
    """
    pfstat_map: b["pfstat"], key = u64(pid_tgid)
    pid       : tgid(u32)
    반환: dict(major_cnt, minor_cnt, major_ns, minor_ns, readahead_cnt)
    동작: 상위 32비트가 pid인 엔트리만 합산 후 삭제
    """
    total = {'major_cnt':0,'minor_cnt':0,'major_ns':0,'minor_ns':0,'readahead_cnt':0}
    to_delete = []
    for k, v in pfstat_map.items():
        key_u64 = _to_int(k)
        tgid = (key_u64 >> 32) & 0xFFFFFFFF
        if tgid != pid:
            continue

        if _is_percpu_leaf(v):
            for leaf in v:
                _pf_leaf_add(total, leaf)
        else:
            _pf_leaf_add(total, v)
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
    total = {"read_cnt":0,"read_ns":0,"read_bytes":0,
             "write_cnt":0,"write_ns":0,"write_bytes":0}
    to_delete = []
    for k, v in rwstat_map.items():
        key_u64 = _to_int(k)
        tgid = (key_u64 >> 32) & 0xFFFFFFFF
        if tgid != pid:
            continue

        if _is_percpu_leaf(v):
            # 보통 rwstat은 percpu 아니지만, 방어적으로 합산
            for leaf in v:
                total["read_cnt"]   += int(leaf.read_cnt)
                total["read_ns"]    += int(leaf.read_ns)
                total["read_bytes"] += int(leaf.read_bytes)
                total["write_cnt"]  += int(leaf.write_cnt)
                total["write_ns"]   += int(leaf.write_ns)
                total["write_bytes"]+= int(leaf.write_bytes)
        else:
            total["read_cnt"]   += int(v.read_cnt)
            total["read_ns"]    += int(v.read_ns)
            total["read_bytes"] += int(v.read_bytes)
            total["write_cnt"]  += int(v.write_cnt)
            total["write_ns"]   += int(v.write_ns)
            total["write_bytes"]+= int(v.write_bytes)

        to_delete.append(k)

    for k in to_delete:
        try:
            del rwstat_map[k]
        except Exception:
            pass
    return total

def on_event(cpu, data, size):
    global missing_start
    e = b["events"].event(data)
    if e.kind == 0:
        last_start_ns[e.pid] = e.ts  # e.pid = tgid
        return

    # kind == 1 (end)
    name = (e.phase.decode("utf-8", "replace").rstrip("\x00") or "<unknown>")
    start = last_start_ns.get(e.pid)
    if start is None:
        missing_start += 1  # 비정상 순서 방어
        return

    dur_ns = e.ts - start
    del last_start_ns[e.pid]

    # 집계(요약 표용)
    sum_ns[name] += dur_ns
    cnt[name]    += 1
    if dur_ns < min_ns[name]: min_ns[name] = dur_ns
    if dur_ns > max_ns[name]: max_ns[name] = dur_ns
    if SHOW_SEQUENCE:
        seq.append((name, dur_ns / 1e6))  # ms

    # ===== PID(tgid) 기준 PageFault/ReadAhead 합산/클리어 (TID→PID) =====
    pf_vals   = _read_pfstat_sum_and_clear_per_tid(b["pfstat"], e.pid)
    major_cnt = pf_vals['major_cnt']
    minor_cnt = pf_vals['minor_cnt']
    major_ns  = pf_vals['major_ns']
    minor_ns  = pf_vals['minor_ns']
    ra_cnt    = pf_vals['readahead_cnt']

    # ===== TID 기반 rwstat 합산/클리어 (PID로 모음) =====
    rw_vals = _read_rwstat_sum_and_clear(b["rwstat"], e.pid)
    sys_read_count = int(rw_vals["read_cnt"])
    sys_write_count = int(rw_vals["write_cnt"])

    # 파생치 계산
    wall_ms       = int(dur_ns / 1e6)
    pf_ms         = int((major_ns + minor_ns) / 1e6)
    major_pf_ms   = int(major_ns / 1e6)
    minor_pf_ms   = int(minor_ns / 1e6)
    read_ms       = int(rw_vals["read_ns"] / 1e6)
    write_ms      = int(rw_vals["write_ns"] / 1e6)
    total_io_ms   = pf_ms + read_ms + write_ms

    
    # ===== rec 생성 (PageFault/ReadAhead 필드 채움) =====
    wall_ms = int(dur_ns / 1e6)
    rec = {
        'phase': name,
        'pid'  : int(e.pid),
        'tid'  : 0,                 # 이번 라운드는 pid 기준이므로 0 유지
        'wall_ms': wall_ms,
        
        # PageFault / ReadAhead
        'pf_ms': pf_ms,
        'major_pf_ms': major_pf_ms,
        'minor_pf_ms': minor_pf_ms,
        'major_fault_count': int(major_cnt),
        'minor_fault_count': int(minor_cnt),
        'major_fault_elapsed_ns': int(major_ns),
        'minor_fault_elapsed_ns': int(minor_ns),
        'readahead_count': int(ra_cnt),
        
        # 아직 미집계 항목은 0 placeholder
        'total_non_io_ms': wall_ms - total_io_ms,
        'total_io_ms': total_io_ms,
        'read_ms': read_ms,
        'write_ms': write_ms,
        'sys_read_count': sys_read_count,
        'sys_write_count': sys_write_count,
        'cpu_runtime_us': 0,
        'cpu_wait_us': 0,
        'block_io_time_us': 0,
        'block_io_count': 0,
        'block_read_bytes': 0,
        'block_write_bytes': 0,
        'io_wall_time_ms': 0,
        'io_parallel_ratio': 0.0,
        'io_ratio_percent': 0,
    }
    phase_records.append(rec)

# ======== Reporting ========
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
            items = sorted(cnt.keys(), key=naturalsort_key)        # 자연 정렬

        print(f"{'Phase':26s}  {'Count':>5s}  {'Avg(ms)':>10s}  {'Min(ms)':>10s}  {'Max(ms)':>10s}  {'Total(ms)':>12s}")
        for k in items:
            avg_ms   = (sum_ns[k] / cnt[k]) / 1e6
            min_ms_v =  min_ns[k] / 1e6
            max_ms_v =  max_ns[k] / 1e6
            tot_ms   =   sum_ns[k] / 1e6
            print(f"{k:26s}  {cnt[k]:5d}  {avg_ms:10.3f}  {min_ms_v:10.3f}  {max_ms_v:10.3f}  {tot_ms:12.3f}")

        # Top-K by Avg(ms) with Ratio & Cum(%)
        TOPK = 7
        avg_map = {phase: sum_ns[phase]/cnt[phase]/1e6 for phase in cnt}
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
            
    # ======== (NEW) Print per-phase breakdowns in sequence ========
    if phase_records:
        print("\n\n===== Phase Breakdown (per occurrence) =====")
        for rec in phase_records:
            print_phase_breakdown(rec)




# ======== Setup / Main ========
if __name__ == "__main__":

    # --- Attach USDT ---
    usdt = USDT(path=binary_path)
    usdt.enable_probe_or_bail("tflite_gen:logic_start", "trace_logic_start")
    usdt.enable_probe_or_bail("tflite_gen:logic_end",   "trace_logic_end")

    b = BPF(text=BPF_TEXT, usdt_contexts=[usdt])

    # --- Aggregation (phase-level: start–stop) ---
    last_start_ns = {}  # pid -> start_ns

    # 누적: phase 이름별 합/횟수/최소/최대
    sum_ns  = defaultdict(int)
    cnt     = defaultdict(int)
    min_ns  = defaultdict(lambda: 1<<62)
    max_ns  = defaultdict(int)
    seq     = []  # (phase, duration_ms) 발생 순서 기록 (옵션)
    missing_start = 0

    # (NEW) per-occurrence breakdown records
    phase_records = []

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
