from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Protocol, Sequence

from .__planner_data_structures__ import PrefetchPlan, PrefetchPlanEntry, WeightChunkInfo
from .strategy_base import ChunkKey, PlanningContext, PlanningStrategy

def _sort_chunks(chunks: Sequence[WeightChunkInfo]) -> List[WeightChunkInfo]:
    return sorted(chunks, key=lambda c: (c.chunk_index))

def _print_chunk_list(chunks: Iterable[WeightChunkInfo]) -> None:
    for chunk in chunks:
        print(f"  Chunk(index={chunk.chunk_index}, offset={chunk.aligned_offset}, size={chunk.aligned_size})")

def _compute_gap_bytes(existing: Sequence[WeightChunkInfo], candidate: WeightChunkInfo) -> int:
    last = existing[-1]
    end_offset = last.aligned_offset + last.aligned_size
    if candidate.aligned_offset <= end_offset:
        return 0
    return candidate.aligned_offset - end_offset

def _sum_aligned_size(chunks: Iterable[WeightChunkInfo]) -> int:
    return sum(chunk.aligned_size for chunk in chunks)