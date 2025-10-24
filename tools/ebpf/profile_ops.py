#!/usr/bin/env python3
from bcc import BPF, USDT
import atexit, signal, sys, re, os

binary_path = "bin/cmt_generator"
# binary_path = "bin/text_generator_main_mmap"
# binary_path = "bin/text_generator_main"
# binary_path = "tools/bin/benchmark_model"  # 추적할 바이너리 경로

script_dir = os.path.dirname(os.path.abspath(__file__))

# ======== Config ========
SHOW_SEQUENCE = False  # 시퀀스 테이블이 필요할 때만 True

# ======== eBPF text ========
# Load EBPF code from external file
ebpf_path = os.path.join(script_dir, "bpfc_profile_ops.c")
with open(ebpf_path, "r") as f:
    BPF_TEXT = f.read()


# ======== Dtypes ======
import ctypes
from collections import defaultdict, deque
from dataclasses import dataclass, field, fields as _dc_fields
from typing import Any, Dict, Union, List

u32 = ctypes.c_uint
u64 = ctypes.c_ulonglong


@dataclass
class OpsRaw:
    pid: int
    ops_name: str
    op_index: int
    obj_index: int
    start_ns: int
    end_ns: int
    pf_vals: Dict[str, Any] = field(default_factory=dict)
    rw_vals: Dict[str, Any] = field(default_factory=dict)
    blk_vals: Dict[str, Any] = field(default_factory=dict)
    sched_vals: Dict[str, Any] = field(default_factory=dict)
    io_wall_time_ns: int = 0

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "OpsRaw":
        kw = {}
        for f in _dc_fields(OpsRaw):
            key = f.name
            if key == "ops_name" and "ops" in d:
                kw[key] = d.get("ops")
            else:
                kw[key] = d.get(key, getattr(f, "default", None))
        return OpsRaw(**kw)


@dataclass
class OpsRecord:
    ops: str = "<unknown>"
    pid: int = 0
    tid: int = 0
    op_index: int = 0
    obj_index: int = 0
    wall_clock_time_ns: int = 0
    wall_clock_time_us: int = 0
    single_thread_non_io_handle_time_us: int = 0
    single_thread_io_handle_time_us: int = 0
    single_thread_read_sys_time_us: int = 0
    single_thread_write_sys_time_us: int = 0
    single_thread_pf_time_us: int = 0
    single_thread_major_pf_time_us: int = 0
    single_thread_minor_pf_time_us: int = 0
    single_thread_minor_retry_pf_time_us: int = 0
    total_sys_read_time_us: int = 0
    total_sys_write_time_us: int = 0
    total_pf_time_us: int = 0
    total_major_pf_time_us: int = 0
    total_minor_pf_time_us: int = 0
    total_minor_retry_pf_time_us: int = 0
    avg_major_pf_time_us: int = 0
    avg_minor_pf_time_us: int = 0
    avg_minor_retry_pf_time_us: int = 0
    total_major_pf_count: int = 0
    total_minor_pf_count: int = 0
    total_minor_pf_retry_count: int = 0
    readahead_count: int = 0
    total_sys_read_count: int = 0
    total_sys_write_count: int = 0
    avg_sys_read_time_us: int = 0
    avg_sys_write_time_us: int = 0
    total_cpu_runtime_us: int = 0
    total_block_io_time_us: int = 0
    total_block_io_count: int = 0
    total_block_io_read_bytes: int = 0
    total_block_io_write_bytes: int = 0
    avg_block_io_time_us: int = 0
    avg_block_io_read_bytes: int = 0
    avg_block_io_write_bytes: int = 0
    wall_block_io_time_us: int = 0
    io_concurrency_factor: float = 0.0
    block_io_time_ratio: float = 0.0
    cpu_parallelism_factor: float = 0.0


    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "OpsRecord":
        kw = {}
        for f in _dc_fields(OpsRecord):
            kw[f.name] = d.get(f.name, getattr(f, "default", None))
        return OpsRecord(**kw)


# ======== IO Interval 처리 추가 ========
# IO interval 버퍼 (from profile_phase_io_backup.py)
interval_buf = defaultdict(deque)  # pid -> deque[(start_ns, end_ns, bytes, op)]


def on_interval_event(cpu, data, size):
    """IO interval 이벤트 핸들러 - interval_buf에 저장"""
    global interval_buf
    
    iv = b["intervals"].event(data)
    pid = int((iv.pid_tgid >> 32) & 0xFFFFFFFF)
    start = int(iv.start_ns)
    end = int(iv.end_ns)
    op_str = iv.op.decode("latin-1") if isinstance(iv.op, bytes) else str(iv.op)
    
    interval_buf[pid].append((start, end, int(iv.bytes), op_str))


def compute_io_wall_and_total(pid: int, ops_start_ns: int, ops_end_ns: int):
    """Ops 창 내에서 IO interval을 병합하여 wall time 계산"""
    global interval_buf

    buf = interval_buf.get(pid)
    if not buf:
        return 0, 0, 0

    # Ops 창 내의 interval들을 클리핑하고 수집
    items = []
    total_ns = 0
    for s, e, _bytes, _op in list(buf):
        if s > ops_end_ns:
            break  # 미래 interval들
        if s < ops_start_ns:
            s = ops_start_ns
        if e <= s:
            continue
        if e > ops_end_ns:
            e = ops_end_ns
        if e <= s:
            continue
        items.append((s, e))
        total_ns += e - s

    if not items:
        return 0, 0, 0

    # Interval 병합 (시작 시간으로 정렬 후 유니온 계산)
    items.sort(key=lambda x: x[0])
    merged_ns = 0
    cs, ce = items[0]
    for s, e in items[1:]:
        if s <= ce:
            if e > ce:
                ce = e
        else:
            merged_ns += ce - cs
            cs, ce = s, e
    merged_ns += ce - cs

    return merged_ns, total_ns, len(items)


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

def _read_sched_sum_and_clear(sched_map, pid):
    return _read_stat_generic(sched_map, pid, ["runtime_ns", "wait_ns"])

def _read_blockstat_sum_and_clear(blockstat_map, pid):
    return _read_stat_generic(
        blockstat_map,
        pid,
        ["time_ns", "count", "read_bytes", "write_bytes"],
    )

def _capture_ops_raw_record(
    pid: int, ops_name: str, op_index: int, obj_index: int, start_ns: int, end_ns: int
) -> OpsRaw:
    """Capture raw kernel stats for an ops without heavy calculation.

    Returns an OpsRaw containing raw counters and timestamps. This function is
    intentionally lightweight so it can be called from the perf event handler.
    """
    # pf_vals = _read_pfstat_sum_and_clear(b["pfstat"], pid)
    # rw_vals = _read_rwstat_sum_and_clear(b["rwstat"], pid)
    blk_vals = _read_blockstat_sum_and_clear(b["blockstat"], pid)
    sched_vals = _read_sched_sum_and_clear(b["schedstat"], pid)

    # 새로 추가: I/O wall time 계산
    io_wall_ns, total_io_ns, count_used = compute_io_wall_and_total(pid, start_ns, end_ns)

    # interval_buf에서 이 ops에서 사용된 interval들 정리
    # ops가 끝나면 관련 I/O도 모두 끝난 것으로 간주하고 버퍼를 비웁니다.
    if pid in interval_buf:
        del interval_buf[pid]

    # print(f"Captured ops: pid={pid}, ops={ops_name}, op_index={op_index}, obj_index={obj_index}, start_ns={start_ns}, end_ns={end_ns}")
    return OpsRaw(
        pid=pid,
        ops_name=ops_name,
        op_index=op_index,
        obj_index=obj_index,
        start_ns=start_ns,
        end_ns=end_ns,
        # pf_vals=pf_vals,
        # rw_vals=rw_vals,
        blk_vals=blk_vals,
        sched_vals=sched_vals,
        io_wall_time_ns=int(io_wall_ns),  # 계산된 IO wall time 설정
    )


def on_ops_event(cpu, data, size):
    global missing_start
    e = b["events"].event(data)
    if e.kind == 0:
        last_start_ns[e.pid] = e.ts  # e.pid = tgid
        return

    # kind == 1 (end)
    name = e.phase.decode("utf-8", "replace").rstrip("\x00") or "<unknown>"
    op_index = int(e.op_index)
    obj_index = int(e.obj_index)
    start = last_start_ns.get(e.pid)
    if start is None:
        missing_start += 1  # 비정상 순서 방어
        return

    # finalize duration and clear in-flight start
    dur_ns = e.ts - start
    del last_start_ns[e.pid]

    # Create operator key for aggregation (operator_name[op_index,obj_index])
    op_key = f"{name}[{op_index},{obj_index}]"

    # 집계(요약 표용)
    sum_ns[op_key] += dur_ns
    cnt[op_key] += 1
    if dur_ns < min_ns[op_key]:
        min_ns[op_key] = dur_ns
    if dur_ns > max_ns[op_key]:
        max_ns[op_key] = dur_ns
    if SHOW_SEQUENCE:
        seq.append((op_key, dur_ns / 1e6))  # ms

    # Build structured ops record via a single function
    raw = _capture_ops_raw_record(e.pid, name, op_index, obj_index, start, e.ts)
    ops_raw_records.append(raw)



def _generate_record(raw_rec: OpsRaw) -> OpsRecord:
    """
    Build final report record from raw capture data.
    """
    pid = raw_rec.pid
    ops_name = raw_rec.ops_name
    op_index = raw_rec.op_index
    obj_index = raw_rec.obj_index
    start_ns = raw_rec.start_ns
    end_ns = raw_rec.end_ns
    pf_vals = raw_rec.pf_vals
    rw_vals = raw_rec.rw_vals
    blk_vals = raw_rec.blk_vals
    sched_vals = raw_rec.sched_vals
    wall_block_io_time_ns = raw_rec.io_wall_time_ns

    # Base metrics
    wall_clock_time_ns = int(end_ns - start_ns)
    wall_clock_time_us = int(wall_clock_time_ns / 1e3)
    wall_clock_time_ms = int(wall_clock_time_ns / 1e6)

    # # Pagefault metrics
    # total_major_pf_count = int(pf_vals["major_cnt"])
    # total_minor_pf_count = int(pf_vals["minor_cnt"])
    # total_minor_pf_retry_count = int(pf_vals["minor_retry_cnt"])

    # readahead_count = int(pf_vals["readahead_cnt"])

    # total_major_pf_time_ns = int(pf_vals["major_ns"])
    # total_minor_pf_time_ns = int(pf_vals["minor_ns"])
    # total_minor_retry_pf_time_ns = int(pf_vals["minor_retry_ns"])

    # total_major_pf_time_ms = int(total_major_pf_time_ns // 1e6)
    # total_minor_pf_time_ms = int(total_minor_pf_time_ns // 1e6)
    # total_minor_retry_pf_time_ms = int(total_minor_retry_pf_time_ns // 1e6)

    # total_pf_time_ms = int(total_major_pf_time_ms + total_minor_pf_time_ms + total_minor_retry_pf_time_ms)

    # Use base thread (smallest tid) attribution provided by _read_pfstat_sum_and_clear
    # single_thread_major_pf_time_ns = int(pf_vals["single_thread_major_ns"])
    # single_thread_minor_pf_time_ns = int(pf_vals["single_thread_minor_ns"])
    # single_thread_minor_retry_pf_time_ns = int(pf_vals["single_thread_minor_retry_ns"])

    # Wall time for page faults
    # single_thread_major_pf_time_ms = int(single_thread_major_pf_time_ns // 1e6)
    # single_thread_minor_pf_time_ms = int(single_thread_minor_pf_time_ns // 1e6)
    # single_thread_minor_retry_pf_time_ms = int(single_thread_minor_retry_pf_time_ns // 1e6)

    # single_thread_pf_time_ms = (single_thread_major_pf_time_ms + single_thread_minor_pf_time_ms + single_thread_minor_retry_pf_time_ms)

    # avg_major_pf_time_us = int((total_major_pf_time_ns // (total_major_pf_count if total_major_pf_count else 1)) // 1000)
    # avg_minor_pf_time_us = int((total_minor_pf_time_ns // (total_minor_pf_count if total_minor_pf_count else 1)) // 1000)
    # avg_minor_retry_pf_time_us = int((total_minor_retry_pf_time_ns // (total_minor_pf_retry_count if total_minor_pf_retry_count else 1)) // 1000)

    # Syscall metrics
    # total_sys_read_ns = rw_vals["read_ns"]
    # total_sys_write_ns = rw_vals["write_ns"]
    # total_sys_read_count = rw_vals["read_cnt"]
    # total_sys_write_count = rw_vals["write_cnt"]

    # total_sys_read_time_ms = int(total_sys_read_ns // 1e6)
    # total_sys_write_time_ms = int(total_sys_write_ns // 1e6)

    #TODO: BPFC 코드와 이것과 연동시켜야함, 
    single_thread_read_sys_time_ms = 0
    single_thread_write_sys_time_ms = 0

    # avg_sys_read_time_us = int((rw_vals["read_ns"] // (total_sys_read_count if total_sys_read_count else 1)) // 1e3)
    # avg_sys_write_time_us = int((rw_vals["write_ns"] // (total_sys_write_count if total_sys_write_count else 1)) // 1e3)

    # Block IO
    total_block_io_time_ns = int(blk_vals["time_ns"])
    total_block_io_time_us = int(blk_vals["time_ns"] / 1e3)
    total_block_io_time_ms = int(blk_vals["time_ns"] / 1e6)
    total_block_io_count = int(blk_vals["count"])
    total_block_io_read_bytes = int(blk_vals["read_bytes"])
    total_block_io_write_bytes = int(blk_vals["write_bytes"])

    avg_block_io_time_us = int(_safe_div(total_block_io_time_us, total_block_io_count))
    avg_block_io_read_bytes = int(_safe_div(total_block_io_read_bytes, total_block_io_count))
    avg_block_io_write_bytes = int(_safe_div(total_block_io_write_bytes, total_block_io_count))

    # CPU
    total_cpu_runtime_ns = int(sched_vals["runtime_ns"])
    total_cpu_runtime_us = int(sched_vals["runtime_ns"] // 1e3)
    total_cpu_runtime_ms = int(sched_vals["runtime_ns"] // 1e6)
    total_cpu_waittime_us = int(sched_vals["wait_ns"] // 1e3)

    # Compute derived
    # single_thread_io_handle_time_ms = single_thread_pf_time_ms + single_thread_read_sys_time_ms + single_thread_write_sys_time_ms
    # single_thread_non_io_handle_time_ms = wall_clock_time_ms - single_thread_io_handle_time_ms
    
    wall_block_io_time_us = int(wall_block_io_time_ns / 1e3)

    cpu_parallelism_factor = total_cpu_runtime_ns / wall_clock_time_ns if wall_clock_time_ns else 0.0
    io_concurrency_factor = (total_block_io_time_ns / wall_block_io_time_ns) if wall_block_io_time_ns else 0.0
    block_io_time_ratio = (wall_block_io_time_ns / wall_clock_time_ns) * 100 if wall_clock_time_ns else 0.0

    rec = OpsRecord(
        ops=ops_name,
        pid=int(pid),
        tid=0,
        op_index=op_index,
        obj_index=obj_index,
        wall_clock_time_ns=wall_clock_time_ns,
        wall_clock_time_us=wall_clock_time_us,
            
        total_cpu_runtime_us=total_cpu_runtime_us,
        
        total_block_io_time_us=total_block_io_time_us,
        total_block_io_count=total_block_io_count,
        total_block_io_read_bytes=total_block_io_read_bytes,
        total_block_io_write_bytes=total_block_io_write_bytes,
        avg_block_io_time_us=avg_block_io_time_us,
        avg_block_io_read_bytes=avg_block_io_read_bytes,
        avg_block_io_write_bytes=avg_block_io_write_bytes,

        wall_block_io_time_us=wall_block_io_time_us,
        cpu_parallelism_factor=cpu_parallelism_factor,
        io_concurrency_factor=io_concurrency_factor,
        block_io_time_ratio=block_io_time_ratio,
    )
    return rec


def _print_ops_breakdown(rec: OpsRecord,index:int):
    """Prints the breakdown of a specific ops using OpsRecord attributes.

    This simplifies printing by assuming callers pass an OpsRecord built by
    `_generate_record`.
    """

    print("-------------------------------------------")
    print(f"[{index}] Operator {rec.ops}[{rec.op_index},{rec.obj_index}] elapsed: {rec.wall_clock_time_ns} (ns)")
    print("")
    print("-- I/O Stats --")
    print(f" Total Block I/O Time                       : {rec.total_block_io_time_us} (us)")
    print(f" Total Block I/O Count                      : {rec.total_block_io_count} (cnt)")
    print(f" Total Block I/O Read Bytes                 : {rec.total_block_io_read_bytes} (bytes)")
    print(f" Avg Block I/O Time                         : {rec.avg_block_io_time_us} (us)")
    print(f" Avg Block I/O Read Bytes                   : {rec.avg_block_io_read_bytes} (bytes)")
    print("")
    print("-- Derived Metrics --")
    print(f" Total CPU Runtime                          : {rec.total_cpu_runtime_us} (us)")
    print(f" Total Block IO Time                        : {rec.total_block_io_time_us} (us)")
    print("")
    print(f" Wall Clock Time                            : {rec.wall_clock_time_us} (us)")
    print(f" Wall Block IO Time                         : {rec.wall_block_io_time_us} (us)")
    print("")
    print(f" CPU Parallelism Factor                     : {rec.cpu_parallelism_factor:.3f} (CPU cores equivalent) (Total CPU / Wall Clock Time)")
    print(f" IO Concurrency Factor                      : {rec.io_concurrency_factor:.3f} (Concurrent IO operations) (Total Block IO / Wall Block IO Time)")
    print(f" IO Time Ratio                              : {rec.block_io_time_ratio:.3f} (%) (Wall IO Time / Wall Clock Time)")
    print("")
    


def print_report():
    print("\n===== Ops Report (start–stop) =====")

    if SHOW_SEQUENCE and seq:
        print("\n-- Sequence (ms) --")
        for name, ms in seq:
            print(f"{name:>50s} : {ms:9.3f}")

    if cnt:
        print("\n-- Summary by Ops --")

        if SHOW_SEQUENCE and seq:
            items = list(dict.fromkeys(name for name, _ in seq))  # 등장 순서
        else:
            items = sorted(cnt.keys(), key=_naturalsort_key)  # 자연 정렬

        print(
            f"{'Ops':50s}  {'Count':>5s}  {'Avg(ms)':>10s}  {'Min(ms)':>10s}  {'Max(ms)':>10s}  {'Total(ms)':>12s}"
        )
        for k in items:
            avg_ms = (sum_ns[k] / cnt[k]) / 1e6
            min_ms_v = min_ns[k] / 1e6
            max_ms_v = max_ns[k] / 1e6
            tot_ms = sum_ns[k] / 1e6
            print(
                f"{k:50s}  {cnt[k]:5d}  {avg_ms:10.3f}  {min_ms_v:10.3f}  {max_ms_v:10.3f}  {tot_ms:12.3f}"
            )

        # Top-K by Avg(ms) with Ratio & Cum(%)
        TOPK = 7
        avg_map = {ops: sum_ns[ops] / cnt[ops] / 1e6 for ops in cnt}
        sorted_by_avg = sorted(avg_map.items(), key=lambda x: x[1], reverse=True)
        total_avg_sum = sum(v for _, v in sorted_by_avg)

        print(f"\n-- Top-{TOPK} Ops by Avg(ms) --")
        print(f"{'Ops':<50} {'Avg(ms)':>12} {'Ratio(%)':>12} {'Cum(%)':>12}")

        cum = 0.0
        for ops, avg in sorted_by_avg[:TOPK]:
            ratio = 100.0 * avg / total_avg_sum if total_avg_sum > 0 else 0.0
            cum += ratio
            print(f"{ops:<50} {avg:12.3f} {ratio:12.1f} {cum:12.1f}")

    # Diagnostics
    if last_start_ns or missing_start:
        print("\n-- Diagnostics --")
        if last_start_ns:
            print(f"in-flight ops without END : {len(last_start_ns)}")
        if missing_start:
            print(f"END without prior START      : {missing_start}")
    print("=====================================\n")

    #  Print per-ops breakdowns in sequence
    # if ops_raw_records:
    #     print("\n\n===== ops Breakdown (per occurrence) =====")
    #     print(f"Total occurrences recorded: {len(ops_raw_records)}")
    #     for i, raw_record in enumerate(ops_raw_records):
    #         record = _generate_record(raw_record)
    #         _print_ops_breakdown(record,i)


# ======== Setup / Main ========
if __name__ == "__main__":

    # --- Attach USDT ---
    usdt = USDT(path=binary_path)
    usdt.enable_probe_or_bail("text_gen:ops_start", "trace_ops_start")
    usdt.enable_probe_or_bail("text_gen:ops_check", "trace_ops_check")

    b = BPF(text=BPF_TEXT, usdt_contexts=[usdt])

    # --- Aggregation (ops-level: start–stop) ---
    last_start_ns = {}  # pid -> start_ns

    # 누적: ops 이름별 합/횟수/최소/최대
    sum_ns = defaultdict(int)
    cnt = defaultdict(int)
    min_ns = defaultdict(lambda: 1 << 62)
    max_ns = defaultdict(int)
    seq = []  # (ops, duration_ms) 발생 순서 기록 (옵션)
    missing_start = 0

    # (NEW) per-occurrence breakdown records
    ops_raw_records = []

    # Buffer 등록: perf(events) + ring(intervals)
    b["events"].open_ring_buffer(on_ops_event)
    b["intervals"].open_ring_buffer(on_interval_event)

    # Exit hooks
    atexit.register(print_report)

    # Start tracing
    print("Tracing USDT probes and IO intervals... Ctrl-C to stop.")
    try:
        while True:
            # Poll both: perf (ops events) and ring (I/O intervals)
            b.ring_buffer_poll(timeout=0)
            b.ring_buffer_poll(timeout=0)
    except KeyboardInterrupt:
        pass
