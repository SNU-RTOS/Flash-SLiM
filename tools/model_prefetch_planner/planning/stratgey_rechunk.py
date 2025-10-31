"""Re-chunking planning strategy with overlap-aware grouping of weight chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Protocol, Sequence

from .__planner_data_structures__ import PrefetchPlan, PrefetchPlanEntry, WeightChunkInfo
from .strategy_base import ChunkKey, PlanningContext, PlanningStrategy
from .io_estimator import IoTimeEstimator

from .common import _sort_chunks, _print_chunk_list, _compute_gap_bytes, _sum_aligned_size



@dataclass
class RechunkPlanningStrategy(PlanningStrategy):
    max_buffer_size: int
    io_estimator: IoTimeEstimator
    default_compute_ms: float = 0.0

    def build(self, context: PlanningContext) -> PrefetchPlan:
        self._validate_chunks(context)
        plan = PrefetchPlan(metadata=dict(context.metadata))
        io_order = 0
        for mode, chunks in context.weight_chunks.items():
            ordered_chunks = _sort_chunks(chunks)
            # compute logical groups, but do NOT mutate original chunk_index values.
            # We will emit one PrefetchPlanEntry per original o-chunk, preserving chunk_index,
            # and encode re-chunking in the entry's io_order (group-triggered prefetch order).
            logical_groups = self._rechunk_mode(mode, ordered_chunks, context)
            plan.plan_entries[mode] = []
            # For group k, we intend to prefetch that group's chunks during execution of group k-1.
            # So we set io_order for chunks in group k to max(0, k-1).
            for group_idx, group in enumerate(logical_groups):
                # encode logical re-chunk group id into io_order so groups are
                # distinguishable in the plan. Previously this used a shifted
                # mapping (max(0, k-1)) to indicate prefetch timing; for
                # clarity we now emit the logical group index directly.
                prefetch_io_order = group_idx
                for chunk in group:
                    # materialize per-chunk payload from lookup if present, else reconstruct
                    lookup_key = (mode, chunk.chunk_index, chunk.origin_offset)
                    chunk_payload = context.chunk_lookup.get(lookup_key)
                    if chunk_payload is None:
                        chunk_payload = {
                            "chunk_index": chunk.chunk_index,
                            "aligned_offset": chunk.aligned_offset,
                            "offset_adjust": chunk.offset_adjust,
                            "aligned_size": chunk.aligned_size,
                            "origin_offset": chunk.origin_offset,
                            "origin_size": chunk.origin_size,
                            "weights_id": chunk.weights_id,
                            "prefetch_mode": chunk.prefetch_mode,
                            "prefetch_mode_str": chunk.prefetch_mode_str,
                        }

                    # compute per-chunk avg_compute_time
                    key = (mode, chunk.chunk_index, chunk.origin_offset)
                    compute_ms = context.profile_stats.get(key, self.default_compute_ms)

                    entry = PrefetchPlanEntry(
                        mode=mode,
                        chunk_data=dict(chunk_payload),
                        io_order=prefetch_io_order,
                        avg_compute_time=compute_ms,
                    )
                    plan.plan_entries[mode].append(entry)
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
                # enforce contiguity invariant: candidate must be contiguous in origin space
                last_chunk = loading[-1] if loading else using[-1]
                expected_origin = last_chunk.origin_offset + last_chunk.origin_size
                if candidate.origin_offset != expected_origin:
                    # not contiguous in origin; stop adding to this loading window so that
                    # the candidate will begin a new logical group
                    break

                candidate_size = candidate.aligned_size
                next_head_size = chunks[index + 1].aligned_size if (index + 1) < len(chunks) else 0
                mem_after = using_size + loading_size + candidate_size + next_head_size
                if mem_after > self.max_buffer_size:
                    break

                # Only enforce the memory invariant. I/O-time-based checks removed to simplify
                # the algorithm per request — accept candidate if memory allows.
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

    
    def _validate_chunks(self, context: PlanningContext) -> None:
        if self.max_buffer_size <= 0:
            raise ValueError("max_buffer_size must be positive")
        for mode, chunks in context.weight_chunks.items():
            for chunk in chunks:
                if chunk.aligned_size > self.max_buffer_size:
                    raise ValueError(
                        f"Chunk {mode}[{chunk.chunk_index}] size {chunk.aligned_size} exceeds buffer {self.max_buffer_size}"
                    )



