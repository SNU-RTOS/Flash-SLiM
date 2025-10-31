from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Protocol, Sequence, Optional

from .__planner_data_structures__ import PrefetchPlan, PrefetchPlanEntry, WeightChunkInfo
from .strategy_base import ChunkKey, PlanningContext, PlanningStrategy

def sort_chunk_list(chunks: Sequence[WeightChunkInfo]) -> List[WeightChunkInfo]:
    return sorted(chunks, key=lambda c: (c.chunk_index))

def print_chunk_list(chunks: Iterable[WeightChunkInfo]) -> None:
    for chunk in chunks:
        print(f"  Chunk(index={chunk.chunk_index}, offset={chunk.aligned_offset}, size={chunk.aligned_size})")

def _compute_gap_bytes(existing: Sequence[WeightChunkInfo], candidate: WeightChunkInfo) -> int:
    last = existing[-1]
    end_offset = last.aligned_offset + last.aligned_size
    if candidate.aligned_offset <= end_offset:
        return 0
    return candidate.aligned_offset - end_offset

def sum_chunk_aligned_size(chunks: Iterable[WeightChunkInfo]) -> int:
    return sum(chunk.aligned_size for chunk in chunks)

default_compute_ms = 0.0

def sum_chunk_compute_time(
        mode: str,
        chunks: Iterable[WeightChunkInfo],
        profile_stats: Mapping[ChunkKey, float],
    ) -> float:
        total = 0.0
        for chunk in chunks:
            key = (mode, chunk.chunk_index, chunk.origin_offset)
            total += profile_stats.get(key, default_compute_ms)
        return total
    
def _coerce_to_float(value: Optional[object], default: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return default
    return default


def _coerce_to_int(value: Optional[object], default: int) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return default
    return default
    