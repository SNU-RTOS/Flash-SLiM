"""Re-chunking planning strategy with overlap-aware grouping of weight chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

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
            logical_groups, group_io_times = self._rechunk_mode(mode, ordered_chunks, context)
            plan.plan_entries[mode] = []
            # For group k, we intend to prefetch that group's chunks during execution of group k-1.
            # So we set io_order for chunks in group k to max(0, k-1).
            for group_idx, group in enumerate(logical_groups):
                group_io_time = group_io_times[group_idx] if group_idx < len(group_io_times) else None
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

                    if group_io_time is not None:
                        chunk_payload["estimated_io_time_ms"] = group_io_time

                    # compute per-chunk avg_compute_time
                    key = (mode, chunk.chunk_index, chunk.origin_offset)
                    compute_ms = context.profile_stats.get(key, self.default_compute_ms)

                    entry = PrefetchPlanEntry(
                        mode=mode,
                        chunk_data=dict(chunk_payload),
                        io_order=prefetch_io_order,
                        avg_compute_time=compute_ms,
                        estimated_io_time_ms=group_io_time,
                    )
                    plan.plan_entries[mode].append(entry)
        return plan


    def _check_chunk_contiguity(
        self, 
        prefetching_chunk_group: List[WeightChunkInfo], 
        active_chunk_group: List[WeightChunkInfo], 
        candidate_chunk: WeightChunkInfo
    ) -> bool:
        '''
        Check if the candidate chunk is contiguous in origin space with respect to
        the last chunk in either the loading_chunks or using_chunks.
        This ensures that chunks grouped together maintain contiguity in their
        original data layout.
        '''
        # enforce contiguity invariant: candidate must be contiguous in origin space
        last_chunk = prefetching_chunk_group[-1] if prefetching_chunk_group else active_chunk_group[-1]
        expected_origin_offset = last_chunk.origin_offset + last_chunk.origin_size
        
        if candidate_chunk.origin_offset != expected_origin_offset:
        # not contiguous in origin; stop adding to this loading set so that the candidate will begin a new logical chunk-group
            return False
        else:
            return True

    def _rechunk_mode(
        self,
        mode: str,
        chunks: Sequence[WeightChunkInfo],
        context: PlanningContext,
    ) -> Tuple[List[List[WeightChunkInfo]], List[float]]:
        
        if not chunks:
            return [], []

        # List of grouped (rechunked) chunks to return
        committed_chunk_groups: List[List[WeightChunkInfo]] = []
        committed_group_io_times: List[float] = []
        
        # Start with the first chunk as the initial using_chunks
        index = 0
        compute_chunk_group: List[WeightChunkInfo] = [chunks[index]]
        index += 1

        # Loop until all chunks are processed
        while index < len(chunks):
            prefetch_chunk_group: List[WeightChunkInfo] = []

            compute_chunk_group_size = sum_chunk_aligned_size(compute_chunk_group)
            prefetch_chunk_group_size = 0
            candidate_chunk_size = 0
            next_candidate_chunk_size = 0

            compute_chunk_group_elapsed_ms = sum_chunk_compute_time(mode, 
                                                                    compute_chunk_group, 
                                                                    context.profile_stats)

            while index < len(chunks):
                # Get the candidate chunk
                candidate_chunk = chunks[index]

                ### Invariant 1. Check chunk memory contiguity ###
                if not self._check_chunk_contiguity(prefetch_chunk_group, 
                                                    compute_chunk_group, 
                                                    candidate_chunk):
                    break

                # Get the candidate chunk size
                candidate_chunk_size = candidate_chunk.aligned_size

                # Get the next candidate chunk size
                next_candidate_chunk_size = chunks[index + 1].aligned_size if (index + 1) < len(chunks) else 0
                
                ### Invariant 2. Check io time hiding condition (group compute_time > group io_time) ###
                io_estimate_ms = self._estimate_prefetch_io_time(
                    mode,
                    compute_chunk_group,
                    prefetch_chunk_group,
                    candidate_chunk,
                )
                if io_estimate_ms is not None and compute_chunk_group_elapsed_ms < io_estimate_ms:
                    break

                ### Invariant 3. Check peak memory usage ###
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
                    compute_chunk_group_size + 
                    prefetch_chunk_group_size + 
                    candidate_chunk_size + 
                    next_candidate_chunk_size
                )
                if prefetch_reserved_buffer_size > self.max_buffer_size:
                    # Exceeded memory limit; stop adding to this loading set so that
                    # the candidate will begin a new logical chunk-group
                    break

                # Passed all invariants -> include the candidate chunk to prefetching chunk group
                prefetch_chunk_group.append(candidate_chunk)
                prefetch_chunk_group_size += candidate_chunk_size
                index += 1

            # If no chunks were added to prefetching_chunk_group, we must forcibly
            # start a new group with the next chunk to avoid infinite loop.
            if not prefetch_chunk_group:
                committed_chunk_groups.append(compute_chunk_group)
                committed_group_io_times.append(self._estimate_group_io_time(mode, compute_chunk_group))
                if index >= len(chunks):
                    compute_chunk_group = []
                    break
                compute_chunk_group = [chunks[index]]
                index += 1
                continue

            # Completed the current loading chunk; commits the current using chunk to plan and
            # start a new using chunk from the loading chunk.
            committed_chunk_groups.append(compute_chunk_group)
            committed_group_io_times.append(self._estimate_group_io_time(mode, compute_chunk_group))
            compute_chunk_group = prefetch_chunk_group
            # Rechunk loop to evaluate a fresh loading chunk.

        if compute_chunk_group:
            committed_chunk_groups.append(compute_chunk_group)
            committed_group_io_times.append(self._estimate_group_io_time(mode, compute_chunk_group))

        return committed_chunk_groups, committed_group_io_times


    def _estimate_prefetch_io_time(
        self,
        mode: str,
        compute_chunk_group: List[WeightChunkInfo],
        prefetch_chunk_group: List[WeightChunkInfo],
        candidate_chunk: WeightChunkInfo,
    ) -> Optional[float]:
        existing = prefetch_chunk_group if prefetch_chunk_group else compute_chunk_group
        if not existing:
            return None
        gap_bytes = _compute_gap_bytes(existing, candidate_chunk)
        try:
            chunks_for_estimate = list(prefetch_chunk_group)
            chunks_for_estimate.append(candidate_chunk)
            estimate = self.io_estimator.estimate(mode, chunks_for_estimate, gap_bytes=gap_bytes)
        except Exception:
            return None
        return float(estimate)

    def _estimate_group_io_time(
        self,
        mode: str,
        chunk_group: List[WeightChunkInfo],
    ) -> float:
        if not chunk_group:
            return 0.0
        try:
            return float(self.io_estimator.estimate(mode, chunk_group, gap_bytes=0))
        except Exception:
            return 0.0


    
    def _validate_chunks(self, context: PlanningContext) -> None:
        if self.max_buffer_size <= 0:
            raise ValueError("max_buffer_size must be positive")
        for mode, chunks in context.weight_chunks.items():
            for chunk in chunks:
                if chunk.aligned_size > self.max_buffer_size:
                    raise ValueError(
                        f"Chunk {mode}[{chunk.chunk_index}] size {chunk.aligned_size} exceeds buffer {self.max_buffer_size}"
                    )



