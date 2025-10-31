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
        # Build per-mode, per-io_order summary for quick inspection without
        # mutating the immutable `weight_chunks` list. This provides how much
        # data will be prefetched for each logical I/O group.
        prefetch_summary: Dict[str, Dict[str, Any]] = {}
        for mode, entries in self.plan_entries.items():
            groups: Dict[int, Dict[str, Any]] = {}
            for entry in entries:
                ed = entry.to_dict()
                io_order = ed.get("io_order", 0)
                grp = groups.setdefault(
                    io_order,
                    {
                        "chunk_count": 0,
                        "total_aligned_size": 0,
                        "total_origin_size": 0,
                        "chunk_indices": [],
                        "start_origin_offset": None,
                        "total_avg_compute_time": 0.0,
                    },
                )
                grp["chunk_count"] += 1
                aligned = ed.get("aligned_size") or 0
                origin_offset = ed.get("origin_offset")
                origin_size = ed.get("origin_size") or 0
                # ensure ints and update totals
                try:
                    a = int(aligned)
                    grp["total_aligned_size"] += a
                except Exception:
                    pass
                try:
                    o_size = int(origin_size)
                    grp["total_origin_size"] += o_size
                except Exception:
                    pass
                # update start_origin_offset to the smallest origin_offset seen in group
                try:
                    if origin_offset is not None:
                        o_off = int(origin_offset)
                        if grp["start_origin_offset"] is None or o_off < grp["start_origin_offset"]:
                            grp["start_origin_offset"] = o_off
                except Exception:
                    pass
                # accumulate per-entry avg_compute_time (treat missing as 0)
                try:
                    avg = ed.get("avg_compute_time")
                    if avg is None:
                        avg_val = 0.0
                    else:
                        avg_val = float(avg)
                    grp["total_avg_compute_time"] += avg_val
                except Exception:
                    pass

                grp["chunk_indices"].append(ed.get("chunk_index"))

            # convert group keys to strings for JSON-friendly stable ordering
            prefetch_summary[mode] = {str(k): v for k, v in sorted(groups.items())}

        return {
            "metadata": metadata_snapshot,
            "weight_chunks": {
                mode: [entry.to_dict() for entry in entries]
                for mode, entries in self.plan_entries.items()
            },
            "prefetch_plan": prefetch_summary,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
