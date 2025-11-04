from __future__ import annotations

import ctypes
import errno
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Protocol, Sequence, Tuple


_LIBC = ctypes.CDLL(None, use_errno=True)
_LIBC.posix_memalign.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_size_t,
    ctypes.c_size_t,
]
_LIBC.posix_memalign.restype = ctypes.c_int
_LIBC.free.argtypes = [ctypes.c_void_p]
_LIBC.free.restype = None
_LIBC.pread.argtypes = [
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_longlong,
]
_LIBC.pread.restype = ctypes.c_ssize_t

from .__planner_data_structures__ import (
    PrefetchPlan,
    PrefetchPlanEntry,
    WeightChunkInfo,
)
from .strategy_base import ChunkKey, PlanningContext, PlanningStrategy


class IoTimeEstimator(Protocol):
    def estimate(
        self,
        mode: str,
        chunks: Sequence[WeightChunkInfo],
        *,
        gap_bytes: int = 0,
    ) -> float:
        """Return the I/O time in milliseconds for loading the provided chunks."""
        ...


@dataclass
class BandwidthIoTimeEstimator:
    bandwidth_bytes_per_sec: float
    fixed_overhead_ms: float = 0.0

    def estimate(
        self,
        mode: str,
        chunks: Sequence[WeightChunkInfo],
        *,
        gap_bytes: int = 0,
    ) -> float:
        del mode  # Mode-specific handling can be added later.
        total_bytes = sum(chunk.aligned_size for chunk in chunks)
        if total_bytes <= 0:
            return self.fixed_overhead_ms
        transfer_ms = (total_bytes / max(self.bandwidth_bytes_per_sec, 1)) * 1000.0
        return self.fixed_overhead_ms + transfer_ms


@dataclass
class MeasuredIoTimeEstimator:
    measurements: Mapping[ChunkKey, float]
    fallback: IoTimeEstimator

    def estimate(
        self,
        mode: str,
        chunks: Sequence[WeightChunkInfo],
        *,
        gap_bytes: int = 0,
    ) -> float:
        total = 0.0
        for chunk in chunks:
            key = (mode, chunk.chunk_index, chunk.origin_offset)
            measured = self.measurements.get(key)
            if measured is None:
                return self.fallback.estimate(mode, chunks, gap_bytes=gap_bytes)
            total += measured
        return total


@dataclass
class DirectIoTimeEstimator:
    """Estimate I/O latency by issuing real direct-I/O reads."""

    fallback: IoTimeEstimator
    threads: int = 4
    block_size_bytes: int = 512 * 1024
    buffer_alignment: int = 4096
    cache_results: bool = True
    _cache: Dict[int, float] = field(default_factory=dict, init=False)

    def estimate(
        self,
        mode: str,
        chunks: Sequence[WeightChunkInfo],
        *,
        gap_bytes: int = 0,
    ) -> float:
        total_bytes = sum(chunk.aligned_size for chunk in chunks) + max(gap_bytes, 0)
        if total_bytes <= 0:
            return 0.0

        padded_bytes = self._align_up(
            max(total_bytes, self.block_size_bytes), self.block_size_bytes
        )

        if self.cache_results and padded_bytes in self._cache:
            return self._cache[padded_bytes]

        try:
            measurement_ms = self._measure_direct_io(padded_bytes)
        except Exception:
            return self.fallback.estimate(mode, chunks, gap_bytes=gap_bytes)

        if self.cache_results:
            self._cache[padded_bytes] = measurement_ms
        return measurement_ms

    def _measure_direct_io(self, size_bytes: int) -> float:
        temp_file: Optional[str] = None
        fd: Optional[int] = None
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                temp_file = tmp.name

            create_fd = os.open(temp_file, os.O_RDWR | os.O_CREAT | os.O_TRUNC)
            try:
                if hasattr(os, "posix_fallocate"):
                    os.posix_fallocate(create_fd, 0, size_bytes)
                else:
                    self._write_zeros(create_fd, size_bytes)
                os.fsync(create_fd)
            finally:
                os.close(create_fd)

            direct_flag = getattr(os, "O_DIRECT", None)
            if direct_flag is None:
                raise OSError(errno.ENOTSUP, "O_DIRECT not supported on this platform")
            fd = os.open(temp_file, os.O_RDONLY | direct_flag)

            segments = self._build_segments(size_bytes)
            start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=len(segments)) as executor:
                futures = [
                    executor.submit(self._read_segment, fd, offset, length)
                    for offset, length in segments
                ]
                for future in futures:
                    future.result()
            end = time.perf_counter()
            return (end - start) * 1000.0
        finally:
            if fd is not None:
                os.close(fd)
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except OSError:
                    pass

    def _write_zeros(self, fd: int, size_bytes: int) -> None:
        chunk = b"\0" * min(self.block_size_bytes, 1024 * 1024)
        remaining = size_bytes
        while remaining > 0:
            to_write = min(len(chunk), remaining)
            os.write(fd, chunk[:to_write])
            remaining -= to_write

    def _build_segments(self, size_bytes: int) -> Tuple[Tuple[int, int], ...]:
        if size_bytes <= 0:
            return ((0, 0),)

        max_segments = max(1, size_bytes // self.block_size_bytes)
        effective_threads = max(1, min(self.threads, max_segments))
        base = max(self.block_size_bytes, size_bytes // effective_threads)
        segment_size = self._align_up(base, self.block_size_bytes)
        segments: List[Tuple[int, int]] = []
        offset = 0
        for i in range(effective_threads):
            if offset >= size_bytes:
                break
            remaining = size_bytes - offset
            length = segment_size if i < effective_threads - 1 else remaining
            length = min(remaining, self._align_up(length, self.block_size_bytes))
            segments.append((offset, length))
            offset += length
        if not segments:
            segments.append((0, size_bytes))
        return tuple(segments)

    def _read_segment(self, fd: int, offset: int, length: int) -> None:
        block = self.block_size_bytes
        buf_ptr = ctypes.c_void_p()
        result = _LIBC.posix_memalign(
            ctypes.byref(buf_ptr), self.buffer_alignment, block
        )
        if result != 0:
            raise OSError(result, "posix_memalign failed")
        try:
            current_offset = offset
            end_offset = offset + length
            while current_offset < end_offset:
                bytes_to_read = block
                read_bytes = _LIBC.pread(
                    fd, buf_ptr, bytes_to_read, ctypes.c_longlong(current_offset)
                )
                if read_bytes < 0:
                    err = ctypes.get_errno()
                    raise OSError(err, os.strerror(err))
                if read_bytes == 0:
                    break
                current_offset += read_bytes
        finally:
            _LIBC.free(buf_ptr)

    def _align_up(self, value: int, alignment: int) -> int:
        if alignment <= 0:
            return value
        remainder = value % alignment
        if remainder == 0:
            return value
        return value + (alignment - remainder)
