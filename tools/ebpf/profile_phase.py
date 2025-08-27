#!/usr/bin/env python3
from bcc import BPF, USDT
import atexit, signal, sys, re, os


binary_path = "./bin/text_generator_main"
script_dir = os.path.dirname(os.path.abspath(__file__))

# ======== Config ========
SHOW_SEQUENCE = False  # 시퀀스 테이블이 필요할 때만 True

# ======== eBPF text ========
# Load EBPF code from external file
ebpf_path = os.path.join(script_dir, "bpfc_profile_phase.c")
with open(ebpf_path, "r") as f:
    BPF_TEXT = f.read()


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
_nat_tok = re.compile(r"\d+|\D+")


def _naturalsort_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in _nat_tok.findall(s)]


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

    print(f"{main_major_ns} {main_minor_ns} {main_minor_retry_ns}")

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
            items = sorted(cnt.keys(), key=_naturalsort_key)  # 자연 정렬

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
