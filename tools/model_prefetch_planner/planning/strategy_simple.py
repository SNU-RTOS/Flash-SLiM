"""Simple planning strategy that mirrors the original per-chunk ordering."""

from __future__ import annotations

from typing import Dict, List, Mapping

from .__planner_data_structures__ import PrefetchPlan, PrefetchPlanEntry, WeightChunkInfo
from .strategy_base import ChunkKey, PlanningContext, PlanningStrategy


def _sort_chunks(chunks: List[WeightChunkInfo]) -> List[WeightChunkInfo]:
    return sorted(chunks, key=lambda c: (c.aligned_offset, c.chunk_index))


def _materialize_chunk_payload(
    chunk_lookup: Mapping[ChunkKey, Dict[str, object]], mode: str, chunk: WeightChunkInfo
) -> Dict[str, object]:
    key = (mode, chunk.chunk_index, chunk.origin_offset)
    chunk_payload = chunk_lookup.get(key)
    if chunk_payload is not None:
        return dict(chunk_payload)
    # Fallback when original payload is unavailable.
    return {
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


class SimplePlanningStrategy(PlanningStrategy):
    """Replicates the legacy planning behavior without re-chunking."""

    def build(self, context: PlanningContext) -> PrefetchPlan:
        plan = PrefetchPlan(metadata=dict(context.metadata))
        io_order = 0
        for mode, chunks in context.weight_chunks.items():
            ordered_chunks = _sort_chunks(chunks)
            plan.plan_entries[mode] = []
            for chunk in ordered_chunks:
                chunk_payload = _materialize_chunk_payload(context.chunk_lookup, mode, chunk)
                plan.plan_entries[mode].append(
                    PrefetchPlanEntry(mode=mode, chunk_data=chunk_payload, io_order=io_order)
                )
                io_order += 1
        return plan
