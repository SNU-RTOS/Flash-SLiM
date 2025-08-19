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

BPF_PERF_OUTPUT(events);

// ── Phase 활성화 여부(=측정 중) 표식: PID 키 ──
BPF_HASH(phase_ts_pid, u32, u64);

// ── Page Fault 측정 맵: PID 키 ──
BPF_HASH(pagefault_start_ns,      u32, u64);
BPF_HASH(major_fault_count,       u32, u64);
BPF_HASH(minor_fault_count,       u32, u64);
BPF_HASH(major_fault_elapsed_ns,  u32, u64);
BPF_HASH(minor_fault_elapsed_ns,  u32, u64);

// ── Readahead 카운트: PID 키 ──
BPF_HASH(readahead_count,         u32, u64);

static __always_inline u32 get_pid() {
    u64 id = bpf_get_current_pid_tgid();
    return id >> 32; // tgid
}

// ===== USDT: logic_start =====
int trace_logic_start(struct pt_regs *ctx) {
    struct event_t e = {};
    
    u64 now = bpf_ktime_get_ns();
    u32 pid = get_pid();

    // phase on + 이전 잔여값 초기화
    phase_ts_pid.update(&pid, &now);
    major_fault_count.delete(&pid);
    minor_fault_count.delete(&pid);
    major_fault_elapsed_ns.delete(&pid);
    minor_fault_elapsed_ns.delete(&pid);
    readahead_count.delete(&pid);
    pagefault_start_ns.delete(&pid);
    
    
    // 이벤트 알림(시작)
    e.ts = bpf_ktime_get_ns();
    e.pid = pid;
    e.kind = 0;
    events.perf_submit(ctx, &e, sizeof(e));
    return 0;
}

// ===== USDT: logic_end =====
int trace_logic_end(struct pt_regs *ctx) {
    struct event_t e = {};
    u64 now = bpf_ktime_get_ns();
    u32 pid = get_pid();

    // phase 이름 읽기(USDT arg0: const char*)
    u64 addr = 0;
    bpf_usdt_readarg(1, ctx, &addr);
    bpf_probe_read_user_str(e.phase, sizeof(e.phase), (void *)addr);

    // phase off
    phase_ts_pid.delete(&pid);
    
    // 이벤트 알림(끝)
    e.ts = now;
    e.pid = pid;
    e.kind = 1;
    events.perf_submit(ctx, &e, sizeof(e));
    return 0;
}

// ── VM_FAULT_* 플래그 (커널 버전에 따라 상수값 다를 수 있음) ──
#define VM_FAULT_MAJOR   4
#define VM_FAULT_NOPAGE  256
#define VM_FAULT_LOCKED  512
#define VM_FAULT_RETRY   1024

// ===== kprobe: handle_mm_fault (진입) =====
int kprobe__handle_mm_fault(struct pt_regs *ctx) {
    u32 pid = get_pid();

    // phase 중이 아닐 때는 무시
    if (!phase_ts_pid.lookup(&pid)) return 0;

    // PF 처리 시작 시각 기억
    u64 now = bpf_ktime_get_ns();
    pagefault_start_ns.update(&pid, &now);
    return 0;
}

// ===== kretprobe: handle_mm_fault (복귀) =====
int kretprobe__handle_mm_fault(struct pt_regs *ctx) {
    u32 pid = get_pid();

    // 시작 시각이 없으면 스킵
    u64 *startp = pagefault_start_ns.lookup(&pid);
    if (!startp) return 0;

    // 경과 ns 계산 + 시작 시각 삭제
    u64 end = bpf_ktime_get_ns();
    u64 delta = end - *startp;
    pagefault_start_ns.delete(&pid);

    // 반환 플래그로 major/minor 버킷 분기
    long retval = PT_REGS_RC(ctx);
    u64 zero = 0;

    if ((retval & VM_FAULT_MAJOR) != 0) {
        u64 *c = major_fault_count.lookup_or_init(&pid, &zero);
        (*c)++;
        u64 *t = major_fault_elapsed_ns.lookup_or_init(&pid, &zero);
        *t += delta;
    } else {
        // non-major(RETTRY/NOPAGE 포함)는 minor 버킷으로 합산
        u64 *c = minor_fault_count.lookup_or_init(&pid, &zero);
        (*c)++;
        u64 *t = minor_fault_elapsed_ns.lookup_or_init(&pid, &zero);
        *t += delta;
    }
    return 0;
}

// ===== kprobe: ondemand_readahead =====
int kprobe__ondemand_readahead(struct pt_regs *ctx) {
    u32 pid = get_pid();

    // phase 중일 때만 카운트
    if (!phase_ts_pid.lookup(&pid)) return 0;

    u64 zero = 0;
    u64 *c = readahead_count.lookup_or_init(&pid, &zero);
    (*c)++;
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

def _read_and_clear_u64(tbl, pid):
    key = u32(pid)
    try:
        val = tbl[key].value
        del tbl[key]     # phase 경계에서 값 분리
        return val
    except KeyError:
        return 0
    
def on_event(cpu, data, size):
    global missing_start
    e = b["events"].event(data)
    if e.kind == 0:
        last_start_ns[e.pid] = e.ts
    else:
        name = (e.phase.decode("utf-8", "replace").rstrip("\x00") or "<unknown>")
        start = last_start_ns.get(e.pid)
        if start is None:
            # 비정상 순서 방어: start가 없으면 카운트만 증가
            missing_start += 1
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

        # ===== PID-based PageFault/ReadAhead 수거 =====
        major_cnt = _read_and_clear_u64(b["major_fault_count"], e.pid)
        minor_cnt = _read_and_clear_u64(b["minor_fault_count"], e.pid)
        major_ns  = _read_and_clear_u64(b["major_fault_elapsed_ns"], e.pid)
        minor_ns  = _read_and_clear_u64(b["minor_fault_elapsed_ns"], e.pid)
        ra_cnt  = _read_and_clear_u64(b["readahead_count"], e.pid)

        pf_ms        = int((major_ns + minor_ns) / 1e6)
        major_pf_ms  = int(major_ns / 1e6)
        minor_pf_ms  = int(minor_ns / 1e6)

        # ===== rec 생성 (PageFault/ReadAhead 필드 채움) =====
        wall_ms = int(dur_ns / 1e6)
        rec = {
            'phase': name,
            'pid'  : int(e.pid),
            'tid'  : 0,                 # 이번 라운드는 pid 기준이므로 0 유지
            'wall_ms': wall_ms,

            # PF 채움
            'pf_ms': pf_ms,
            'major_pf_ms': major_pf_ms,
            'minor_pf_ms': minor_pf_ms,
            'major_fault_count': int(major_cnt),
            'minor_fault_count': int(minor_cnt),
            'major_fault_elapsed_ns': int(major_ns),
            'minor_fault_elapsed_ns': int(minor_ns),
            'readahead_count': int(ra_cnt),

            # 아직 미집계 항목은 0 placeholder
            'total_non_io_ms': 0,
            'total_io_ms': 0,
            'read_ms': 0,
            'write_ms': 0,
            'sys_read_count': 0,
            'sys_write_count': 0,
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

    # ======== (NEW) Print per-phase breakdowns in sequence ========
    if phase_records:
        print("\n\n===== Phase Breakdown (per occurrence) =====")
        for rec in phase_records:
            print_phase_breakdown(rec)

    # Diagnostics
    if last_start_ns or missing_start:
        print("\n-- Diagnostics --")
        if last_start_ns:
            print(f"in-flight phases without END : {len(last_start_ns)}")
        if missing_start:
            print(f"END without prior START      : {missing_start}")

    print("=====================================\n")

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
