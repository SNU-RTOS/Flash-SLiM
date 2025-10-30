"""Dataclasses shared across planning strategies."""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class WeightChunkInfo:
    chunk_index: int
    aligned_offset: int
    offset_adjust: int
    aligned_size: int
    origin_offset: int
    origin_size: int
    weights_id: int
    prefetch_mode: int
    prefetch_mode_str: str


@dataclass
class PrefetchPlanEntry:
    mode: str
    chunk_data: Dict[str, Any]
    io_order: int
    avg_compute_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        ordered_entry = OrderedDict()
        for key, value in self.chunk_data.items():
            ordered_entry[key] = value
        ordered_entry["avg_compute_time"] = self.avg_compute_time
        ordered_entry["io_order"] = self.io_order
        return ordered_entry

    @property
    def chunk_index(self) -> int:
        return self.chunk_data.get("chunk_index", -1)

    @property
    def origin_offset(self) -> int:
        return self.chunk_data.get("origin_offset", -1)


@dataclass
class PrefetchPlan:
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)
    plan_entries: Dict[str, List[PrefetchPlanEntry]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        metadata_snapshot = dict(self.metadata)
        metadata_snapshot["prefetch_plan_version"] = self.version
        return {
            "metadata": metadata_snapshot,
            "weight_chunks": {
                mode: [entry.to_dict() for entry in entries]
                for mode, entries in self.plan_entries.items()
            },
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
