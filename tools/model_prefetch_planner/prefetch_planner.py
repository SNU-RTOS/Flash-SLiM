#!/usr/bin/env python3
# prefetch_planner.py
# Prototype version of Prefetch Planner
# Author: GeonHa Park
# Description:
#   Generate a Prefetch Plan based on Weight Chunk Metadata Table (CMT)
#   following the architecture defined in Flash-SLiM Prefetch Planner Design.

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from planning.__planner_data_structures import PrefetchPlan, WeightChunkInfo
from planning.stratgey_rechunk import (
    BandwidthIoTimeEstimator,
    IoTimeEstimator,
    RechunkPlanningStrategy,
)
from planning.strategy_simple import SimplePlanningStrategy
from planning.strategy_base import ChunkKey, PlanningContext, PlanningStrategy


class PrefetchPlanner:
    def __init__(self, cmt_path: str):
        self.cmt_path = cmt_path
        self.metadata: Dict[str, object] = {}
        self.weight_chunks: Dict[str, List[WeightChunkInfo]] = {}
        self.chunk_lookup: Dict[ChunkKey, Dict[str, object]] = {}
        self.profile_stats: Dict[ChunkKey, float] = {}
        self.loaded_profile_files: List[str] = []

    def load_cmt(self) -> None:
        print(f"[PrefetchPlanner] Loading Weight Chunk Metadata Table from: {self.cmt_path}")
        with open(self._resolve_path(self.cmt_path), "r", encoding="utf-8") as source:
            cmt_data = json.load(source)

        self.metadata = cmt_data.get("metadata", {}) or {}
        raw_chunks = cmt_data.get("weight_chunks", {}) or {}

        for mode, chunks in raw_chunks.items():
            typed_chunks: List[WeightChunkInfo] = []
            for chunk in chunks:
                chunk_payload = {k: v for k, v in chunk.items() if k != "managed_buffer_index"}
                info = WeightChunkInfo(**chunk_payload)
                typed_chunks.append(info)
                self.chunk_lookup[(mode, info.chunk_index, info.origin_offset)] = dict(chunk)
            self.weight_chunks[mode] = typed_chunks

        print(f"[PrefetchPlanner] Loaded {sum(len(v) for v in self.weight_chunks.values())} chunks")

    def load_profile_logs(self, pattern: str = "bpf_profile_ops_results_*") -> None:
        print(f"[PrefetchPlanner] Loading profiling logs with pattern: {pattern}")
        base_dir = os.path.dirname(self.cmt_path) or os.getcwd()
        glob_path = os.path.join(base_dir, pattern)
        profile_files = sorted(glob.glob(glob_path))
        if not profile_files:
            print("[PrefetchPlanner] No profiling logs found; avg_compute_time will be left as null")
            return

        aggregator: Dict[ChunkKey, List[float]] = defaultdict(list)
        ops_pattern = re.compile(r"^(?P<mode>[A-Z_]+)\[(?P<chunk>\d+),(?P<offset>\d+)\]")

        for profile_file in profile_files:
            print(f"[PrefetchPlanner] Parsing profiling log: {profile_file}")
            self.loaded_profile_files.append(profile_file)
            with open(profile_file, "r", encoding="utf-8") as source:
                for line in source:
                    line = line.strip()
                    if not line or line.startswith("=") or line.startswith("--") or line.startswith("Ops"):
                        continue
                    parts = line.split()
                    if not parts:
                        continue
                    match = ops_pattern.match(parts[0])
                    if not match or len(parts) < 6 or not parts[1].isdigit():
                        continue
                    mode = match.group("mode")
                    chunk_idx = int(match.group("chunk"))
                    origin_offset = int(match.group("offset"))
                    try:
                        avg_ms = float(parts[2])
                    except ValueError:
                        continue
                    aggregator[(mode, chunk_idx, origin_offset)].append(avg_ms)

        self.profile_stats = {
            key: statistics.mean(values)
            for key, values in aggregator.items()
        }
        print(f"[PrefetchPlanner] Loaded profiling stats for {len(self.profile_stats)} chunks")

    # def integrate_profiling_data(self, plan: PrefetchPlan) -> None:
    #     if not self.profile_stats:
    #         print("[PrefetchPlanner] No profiling data to integrate; skipping avg_compute_time enrichment")
    #         return

    #     print("[PrefetchPlanner] Integrating profiling data into prefetch plan")
    #     for mode, entries in plan.plan_entries.items():
    #         for entry in entries:
    #             if entry.avg_compute_time is not None:
    #                 continue
    #             key = (mode, entry.chunk_index, entry.origin_offset)
    #             if key in self.profile_stats:
    #                 entry.avg_compute_time = self.profile_stats[key]

    #     print("[PrefetchPlanner] Profiling data integration complete")

    def build_context(self) -> PlanningContext:
        return PlanningContext(
            metadata=dict(self.metadata),
            weight_chunks=self.weight_chunks,
            chunk_lookup=self.chunk_lookup,
            profile_stats=self.profile_stats,
        )
        
    def save_prefetch_plan(self, plan: PrefetchPlan, output_path: str) -> None:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as sink:
            json.dump(plan.to_dict(), sink, indent=2)
        print(f"[PrefetchPlanner] Prefetch plan saved to {output_path}")

        
    def _resolve_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        base_dir = os.path.dirname(self.cmt_path) or os.getcwd()
        return os.path.join(base_dir, path)


def create_strategy(args: argparse.Namespace, planner: PrefetchPlanner) -> PlanningStrategy:
    strategy_id = args.strategy.lower()
    if strategy_id == "simple":
        return SimplePlanningStrategy()
    if strategy_id == "rechunk":
        buffer_size = _resolve_buffer_size(planner)
        if buffer_size <= 0:
            raise ValueError("Re-chunk strategy requires a positive max_buffer_size in metadata")
        io_estimator = _build_io_estimator(planner)
        return RechunkPlanningStrategy(
            max_buffer_size=buffer_size,
            io_estimator=io_estimator,
            default_compute_ms=_resolve_default_compute_ms(planner),
            allow_stall=_resolve_allow_stall(planner),
        )
    raise ValueError(f"Unknown planning strategy: {args.strategy}")


def _build_io_estimator(planner: PrefetchPlanner) -> IoTimeEstimator:
    bandwidth = max(
        _coerce_to_float(planner.metadata.get("io_bandwidth_bytes_per_s"), 1_000_000_000.0),
        1.0,
    )
    base_estimator = BandwidthIoTimeEstimator(
        bandwidth_bytes_per_sec=bandwidth,
        fixed_overhead_ms=max(
            _coerce_to_float(planner.metadata.get("io_fixed_overhead_ms"), 0.0),
            0.0,
        ),
    )
    return base_estimator


def _resolve_buffer_size(planner: PrefetchPlanner) -> int:
    # Support multiple possible metadata keys (legacy and current)
    
    key = "weight_chunk_buffer_size"
    if key in planner.metadata:
            return _coerce_to_int(planner.metadata.get(key), 0)
    return 0


def _resolve_default_compute_ms(planner: PrefetchPlanner) -> float:
    return _coerce_to_float(planner.metadata.get("default_compute_ms"), 0.0)


def _resolve_allow_stall(planner: PrefetchPlanner) -> bool:
    value = planner.metadata.get("allow_stall")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return False


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Prefetch Plan from Weight Chunk Metadata Table",
    )
    parser.add_argument(
        "--cmt",
        default="weight_chunks_metadata_table.json",
        help="Path to weight chunks metadata file in JSON format",
    )
    parser.add_argument(
        "--output",
        default="prefetch_plan.json",
        help="Path to save the generated prefetch plan",
    )
    parser.add_argument(
        "--profile-pattern",
        default="bpf_profile_ops_results_*",
        help="Glob pattern (relative to the CMT directory) for profiling logs",
    )
    parser.add_argument(
        "--strategy",
        choices=["simple", "rechunk"],
        default="simple",
        help="Planning strategy to apply",
    )
    args = parser.parse_args()

    planner = PrefetchPlanner(args.cmt)
    planner.load_cmt()
    planner.load_profile_logs(args.profile_pattern)

    context = planner.build_context()
    strategy = create_strategy(args, planner)
    
    plan = strategy.build(context)
    planner.save_prefetch_plan(plan, args.output)


if __name__ == "__main__":
    main()
