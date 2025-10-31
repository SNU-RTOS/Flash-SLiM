from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

from .__planner_data_structures__ import PrefetchPlan, PrefetchPlanEntry, WeightChunkInfo
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

