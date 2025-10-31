"""Re-chunking planning strategy with overlap-aware grouping of weight chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Protocol, Sequence

from .__planner_data_structures__ import PrefetchPlan, PrefetchPlanEntry, WeightChunkInfo
from .strategy_base import ChunkKey, PlanningContext, PlanningStrategy
from .io_estimator import IoTimeEstimator

from .common import sort_chunk_list, print_chunk_list, _compute_gap_bytes, sum_chunk_aligned_size, sum_chunk_compute_time



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
            ordered_chunks = sort_chunk_list(chunks)
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
        using_chunks: List[WeightChunkInfo] = [chunks[index]]

        index += 1

        while index < len(chunks):
            loading_chunks: List[WeightChunkInfo] = []

            using_chunks_size = sum_chunk_aligned_size(using_chunks)
            loading_chunks_size = 0
            candidate_chunk_size = 0
            next_head_chunk_size = 0

            using_compute_ms = sum_chunk_compute_time(mode, using_chunks, context.profile_stats)

            while index < len(chunks):
                # Get the candidate chunk
                candidate_chunk = chunks[index]

                # enforce contiguity invariant: candidate must be contiguous in origin space
                last_chunk = loading_chunks[-1] if loading_chunks else using_chunks[-1]
                expected_origin = last_chunk.origin_offset + last_chunk.origin_size
                if candidate_chunk.origin_offset != expected_origin:
                    # not contiguous in origin; stop adding to this loading set so that the candidate will begin a new logical chunk-group
                    break

                # Evaluate memory usage if we add the candidate chunk
                candidate_chunk_size = candidate_chunk.aligned_size

                # Get the next chunk size
                next_head_chunk_size = chunks[index + 1].aligned_size if (index + 1) < len(chunks) else 0

                # Evaluate projected peak buffer usage (size) if we include the candidate.
                #
                # Why we include the "next head" here:
                # The strategy implements an explicit compute⇄I/O overlap model: while the
                # runtime performs computation on the current `using` chunks, we issue
                # prefetch I/O for the next io_order (the next logical I/O chunk-group). That
                # means the planner must reserve buffer space not only for the chunks
                # we're currently loading but also for the immediate next group's chunk
                # that we intend to prefetch during the compute phase. Including the
                # next head in this projected peak calculation ensures we are conservative
                # and avoid buffer exhaustion when the prefetch for the next io_order
                # is issued.
                #
                # The variable name `prefetch_reserved_buffer_bytes` reflects that this
                # value is a reservation for prefetching (a future/peak estimate,
                # not the instantaneous usage). We include the next-head so that the
                # planner reserves space to prefetch the next io_order's chunk while
                # the current `using` chunks are being computed.
                prefetch_reserved_buffer_size = (
                    using_chunks_size + loading_chunks_size + candidate_chunk_size + next_head_chunk_size
                )
                if prefetch_reserved_buffer_size > self.max_buffer_size:
                    # Exceeded memory limit; stop adding to this loading set so that
                    # the candidate will begin a new logical chunk-group
                    break

                # Only enforce the memory invariant. I/O-time-based checks removed to simplify
                # the algorithm per request — accept candidate if memory allows.
                loading_chunks.append(candidate_chunk)
                loading_chunks_size += candidate_chunk_size
                index += 1

            if not loading_chunks:
                result.append(using_chunks)
                if index >= len(chunks):
                    using_chunks = []
                    break
                using_chunks = [chunks[index]]
                index += 1
                continue

            result.append(using_chunks)
            using_chunks = loading_chunks
            # Reset loop to evaluate a fresh loading set.

        if using_chunks:
            result.append(using_chunks)

        return result


    
    def _validate_chunks(self, context: PlanningContext) -> None:
        if self.max_buffer_size <= 0:
            raise ValueError("max_buffer_size must be positive")
        for mode, chunks in context.weight_chunks.items():
            for chunk in chunks:
                if chunk.aligned_size > self.max_buffer_size:
                    raise ValueError(
                        f"Chunk {mode}[{chunk.chunk_index}] size {chunk.aligned_size} exceeds buffer {self.max_buffer_size}"
                    )



