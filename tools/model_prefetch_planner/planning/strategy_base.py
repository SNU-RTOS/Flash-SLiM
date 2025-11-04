"""Base interfaces and context objects for planning strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Protocol, Tuple

from .__planner_data_structures__ import PrefetchPlan, WeightChunkInfo

ChunkKey = Tuple[str, int, int]


@dataclass
class PlanningContext:
    metadata: Dict[str, object]
    weight_chunks: Mapping[str, List[WeightChunkInfo]]
    chunk_lookup: Mapping[ChunkKey, Mapping[str, object]]
    profile_stats: Mapping[ChunkKey, float]


class PlanningStrategy(Protocol):
    def build(self, context: PlanningContext) -> PrefetchPlan:
        """Generate a prefetch plan for the provided context."""
        ...
