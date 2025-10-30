"""Re-chunking planning strategy with overlap-aware grouping of weight chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Protocol, Sequence

from .__planner_data_structures import PrefetchPlan, PrefetchPlanEntry, WeightChunkInfo
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
class RechunkPlanningStrategy(PlanningStrategy):
    max_buffer_size: int
    io_estimator: IoTimeEstimator
    default_compute_ms: float = 0.0
    allow_stall: bool = False

    def build(self, context: PlanningContext) -> PrefetchPlan:
        self._validate_chunks(context)
        plan = PrefetchPlan(metadata=dict(context.metadata))
        io_order = 0
        for mode, chunks in context.weight_chunks.items():
            ordered_chunks = _sort_chunks(chunks)
            logical_groups = self._rechunk_mode(mode, ordered_chunks, context)
            plan.plan_entries[mode] = []
            for logical_index, group in enumerate(logical_groups):
                chunk_payload = self._materialize_group_payload(
                    mode, logical_index, group, context.chunk_lookup
                )
                compute_ms = self._sum_compute_time(mode, group, context.profile_stats)
                entry = PrefetchPlanEntry(
                    mode=mode,
                    chunk_data=chunk_payload,
                    io_order=io_order,
                    avg_compute_time=compute_ms,
                )
                plan.plan_entries[mode].append(entry)
                io_order += 1
        return plan

    def _rechunk_mode(
        self,
        mode: str,
        chunks: Sequence[WeightChunkInfo],
        context: PlanningContext,
    ) -> List[List[WeightChunkInfo]]:
        if not chunks:
            return []

        result: List[List[WeightChunkInfo]] = []
        index = 0
        using: List[WeightChunkInfo] = [chunks[index]]
        index += 1

        while index < len(chunks):
            loading: List[WeightChunkInfo] = []
            loading_size = 0
            using_compute_ms = self._sum_compute_time(mode, using, context.profile_stats)
            using_size = _sum_aligned_size(using)

            while index < len(chunks):
                candidate = chunks[index]
                candidate_size = candidate.aligned_size
                next_head_size = chunks[index + 1].aligned_size if (index + 1) < len(chunks) else 0
                mem_after = using_size + loading_size + candidate_size + next_head_size
                if mem_after > self.max_buffer_size:
                    break

                tentative_loading = loading + [candidate]
                gap_bytes = _compute_gap_bytes(loading or using, candidate)
                io_ms = self.io_estimator.estimate(mode, tentative_loading, gap_bytes=gap_bytes)

                if using_compute_ms < io_ms:
                    break

                loading.append(candidate)
                loading_size += candidate_size
                index += 1

            if not loading:
                result.append(using)
                if index >= len(chunks):
                    using = []
                    break
                using = [chunks[index]]
                index += 1
                continue

            result.append(using)
            using = loading
            # Reset loop to evaluate a fresh loading window.

        if using:
            result.append(using)

        return result

    def _sum_compute_time(
        self,
        mode: str,
        chunks: Iterable[WeightChunkInfo],
        profile_stats: Mapping[ChunkKey, float],
    ) -> float:
        total = 0.0
        for chunk in chunks:
            key = (mode, chunk.chunk_index, chunk.origin_offset)
            total += profile_stats.get(key, self.default_compute_ms)
        return total

    def _materialize_group_payload(
        self,
        mode: str,
        logical_index: int,
        group: Sequence[WeightChunkInfo],
        chunk_lookup: Mapping[ChunkKey, Dict[str, object]],
    ) -> Dict[str, object]:
        first = group[0]
        weights_id = {chunk.weights_id for chunk in group}
        prefetch_modes = {chunk.prefetch_mode for chunk in group}
        prefetch_mode_strs = {chunk.prefetch_mode_str for chunk in group}
        if len(weights_id) != 1 or len(prefetch_modes) != 1 or len(prefetch_mode_strs) != 1:
            raise ValueError("Re-chunking requires uniform weights_id and prefetch_mode within a group")

        aggregate_payload = {
            "chunk_index": logical_index,
            "aligned_offset": first.aligned_offset,
            "offset_adjust": first.offset_adjust,
            "aligned_size": _sum_aligned_size(group),
            "origin_offset": first.origin_offset,
            "origin_size": sum(chunk.origin_size for chunk in group),
            "weights_id": first.weights_id,
            "prefetch_mode": first.prefetch_mode,
            "prefetch_mode_str": first.prefetch_mode_str,
            "rechunked_span": [
                {
                    "chunk_index": chunk.chunk_index,
                    "aligned_offset": chunk.aligned_offset,
                    "aligned_size": chunk.aligned_size,
                    "origin_offset": chunk.origin_offset,
                    "origin_size": chunk.origin_size,
                }
                for chunk in group
            ],
            "estimated_io_time_ms": self.io_estimator.estimate(mode, group),
        }

        # Preserve any additional keys from the first raw payload.
        raw_payload = chunk_lookup.get((mode, first.chunk_index, first.origin_offset))
        if raw_payload:
            for key, value in raw_payload.items():
                aggregate_payload.setdefault(key, value)

        return aggregate_payload

    def _validate_chunks(self, context: PlanningContext) -> None:
        if self.max_buffer_size <= 0:
            raise ValueError("max_buffer_size must be positive")
        for mode, chunks in context.weight_chunks.items():
            for chunk in chunks:
                if chunk.aligned_size > self.max_buffer_size:
                    raise ValueError(
                        f"Chunk {mode}[{chunk.chunk_index}] size {chunk.aligned_size} exceeds buffer {self.max_buffer_size}"
                    )


def _compute_gap_bytes(existing: Sequence[WeightChunkInfo], candidate: WeightChunkInfo) -> int:
    last = existing[-1]
    end_offset = last.aligned_offset + last.aligned_size
    if candidate.aligned_offset <= end_offset:
        return 0
    return candidate.aligned_offset - end_offset


def _sort_chunks(chunks: Sequence[WeightChunkInfo]) -> List[WeightChunkInfo]:
    return sorted(chunks, key=lambda c: (c.aligned_offset, c.chunk_index))


def _sum_aligned_size(chunks: Iterable[WeightChunkInfo]) -> int:
    return sum(chunk.aligned_size for chunk in chunks)
