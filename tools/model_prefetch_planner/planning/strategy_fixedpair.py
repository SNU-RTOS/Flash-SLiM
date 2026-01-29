from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .__planner_data_structures__ import PrefetchPlan, PrefetchPlanEntry, WeightChunkInfo
from .strategy_base import PlanningContext, PlanningStrategy
from .io_estimator import IoTimeEstimator
from .common import (
    sort_chunk_list,
    sum_chunk_compute_time,
    resolve_chunk_payload,
)


@dataclass
class FixedPairDecodePlanningStrategy(PlanningStrategy):
    """
    Testing-only planning strategy:
      - DECODE: force each logical group to contain exactly 2 original chunks (except last may be 1).
      - PREFILL: keep the existing rechunk behavior (default), or optionally pair as well.

    Emits one PrefetchPlanEntry per original chunk, preserving chunk_index.
    Encodes logical group id into io_order (0,1,2,...).
    """
    max_buffer_size: int
    io_estimator: IoTimeEstimator
    default_compute_ms: float = 0.0

    decode_group_size: int = 2
    pair_prefill_too: bool = False  # set True if you also want PREFILL to be fixed-pair

    def build(self, context: PlanningContext) -> PrefetchPlan:
        self._validate_chunks(context)

        plan = PrefetchPlan(metadata=dict(context.metadata))

        for mode, chunks in context.weight_chunks.items():
            ordered_chunks = sort_chunk_list(chunks)

            if mode == "DECODE":
                logical_groups, group_io_times = self._group_by_fixed_count(
                    mode, ordered_chunks, group_size=self.decode_group_size
                )
            else:
                if self.pair_prefill_too:
                    logical_groups, group_io_times = self._group_by_fixed_count(
                        mode, ordered_chunks, group_size=self.decode_group_size
                    )
                else:
                    # Fall back to your current rechunking policy for non-DECODE.
                    # If you want, you can import and call your existing RechunkPlanningStrategy here
                    # instead of copying its logic.
                    logical_groups, group_io_times = self._keep_original_grouping(
                        mode, ordered_chunks
                    )

            plan.plan_entries[mode] = []

            for group_idx, group in enumerate(logical_groups):
                group_io_time = group_io_times[group_idx] if group_idx < len(group_io_times) else None
                prefetch_io_order = group_idx  # group id

                for chunk in group:
                    chunk_payload = resolve_chunk_payload(context.chunk_lookup, mode, chunk)
                    key = (mode, chunk.chunk_index, chunk.origin_offset)
                    compute_ms = context.profile_stats.get(key, self.default_compute_ms)

                    plan.plan_entries[mode].append(
                        PrefetchPlanEntry(
                            mode=mode,
                            chunk_data=chunk_payload,
                            io_order=prefetch_io_order,
                            avg_compute_time=compute_ms,
                            estimated_io_time_ms=group_io_time,
                        )
                    )

        return plan

    # -----------------------------
    # Grouping policies
    # -----------------------------
    def _group_by_fixed_count(
        self,
        mode: str,
        chunks: Sequence[WeightChunkInfo],
        group_size: int,
    ) -> Tuple[List[List[WeightChunkInfo]], List[float]]:
        if not chunks:
            return [], []
        if group_size <= 0:
            raise ValueError("group_size must be positive")

        groups: List[List[WeightChunkInfo]] = []
        io_times: List[float] = []

        i = 0
        while i < len(chunks):
            group = list(chunks[i : i + group_size])
            i += group_size

            groups.append(group)
            io_times.append(self._estimate_group_io_time(mode, group))

        return groups, io_times

    def _keep_original_grouping(
        self,
        mode: str,
        chunks: Sequence[WeightChunkInfo],
    ) -> Tuple[List[List[WeightChunkInfo]], List[float]]:
        """
        Minimal "do nothing" fallback: each original chunk becomes its own group.
        If you want the exact previous behavior, replace this with a call to your
        existing RechunkPlanningStrategy._rechunk_mode logic.
        """
        groups = [[c] for c in chunks]
        io_times = [self._estimate_group_io_time(mode, g) for g in groups]
        return groups, io_times

    # -----------------------------
    # I/O estimate helpers
    # -----------------------------
    def _estimate_group_io_time(self, mode: str, chunk_group: List[WeightChunkInfo]) -> float:
        if not chunk_group:
            return 0.0
        try:
            return float(self.io_estimator.estimate(mode, chunk_group, gap_bytes=0))
        except Exception:
            return 0.0

    # -----------------------------
    # Validation
    # -----------------------------
    def _validate_chunks(self, context: PlanningContext) -> None:
        if self.max_buffer_size <= 0:
            raise ValueError("max_buffer_size must be positive")
        for mode, chunks in context.weight_chunks.items():
            for chunk in chunks:
                if chunk.aligned_size > self.max_buffer_size:
                    raise ValueError(
                        f"Chunk {mode}[{chunk.chunk_index}] size {chunk.aligned_size} exceeds buffer {self.max_buffer_size}"
                    )
