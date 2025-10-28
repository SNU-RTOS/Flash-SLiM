#!/usr/bin/env python3
# prefetch_planner.py
# Prototype version of Prefetch Planner
# Author: GeonHa Park
# Description:
#   Generate a Prefetch Plan based on Weight Chunk Metadata Table (CMT)
#   following the architecture defined in Flash-SLiM Prefetch Planner Design.

import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import os
import glob
import re
from collections import defaultdict, OrderedDict
import statistics


@dataclass
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
        return self.chunk_data["chunk_index"]

    @property
    def origin_offset(self) -> int:
        return self.chunk_data["origin_offset"]


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
            "prefetch_plan": {
                mode: [entry.to_dict() for entry in entries]
                for mode, entries in self.plan_entries.items()
            }
        }


class PrefetchPlanner:
    def __init__(self, cmt_path: str):
        self.cmt_path = cmt_path
        self.metadata = {}
        self.weight_chunks: Dict[str, List[WeightChunkInfo]] = {}
        self.raw_weight_chunks: Dict[str, List[Dict[str, Any]]] = {}
        self.chunk_lookup: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
        self.profile_stats: Dict[Tuple[str, int, int], float] = {}
        self.loaded_profile_files: List[str] = []

    def load_cmt(self):
        print(f"[PrefetchPlanner] Loading Weight Chunk Metadata Table from: {self.cmt_path}")
        with open(self.cmt_path, "r") as f:
            cmt_data = json.load(f)
        self.metadata = cmt_data.get("metadata", {})
        raw_chunks = cmt_data.get("weight_chunks", {})
        for mode, chunks in raw_chunks.items():
            dataclass_chunks: List[WeightChunkInfo] = []
            raw_chunk_snapshots: List[Dict[str, Any]] = []
            for chunk in chunks:
                chunk_copy = dict(chunk)
                raw_chunk_snapshots.append(chunk_copy)
                chunk_payload = {k: v for k, v in chunk.items() if k != "managed_buffer_index"}
                dataclass_chunks.append(WeightChunkInfo(**chunk_payload))
                lookup_key = (mode, chunk["chunk_index"], chunk["origin_offset"])
                self.chunk_lookup[lookup_key] = chunk_copy
            self.weight_chunks[mode] = dataclass_chunks
            self.raw_weight_chunks[mode] = raw_chunk_snapshots
        print(f"[PrefetchPlanner] Loaded {sum(len(v) for v in self.weight_chunks.values())} chunks")

    def load_profile_logs(self, pattern: str = "bpf_profile_ops_results_*"):
        print(f"[PrefetchPlanner] Loading profiling logs with pattern: {pattern}")
        base_dir = os.path.dirname(self.cmt_path) or os.getcwd()
        glob_path = os.path.join(base_dir, pattern)
        profile_files = sorted(glob.glob(glob_path))
        if not profile_files:
            print("[PrefetchPlanner] No profiling logs found; avg_compute_time will be left as null")
            return

        aggregator: Dict[Tuple[str, int, int], List[float]] = defaultdict(list)
        ops_pattern = re.compile(r"^(?P<mode>[A-Z_]+)\[(?P<chunk>\d+),(?P<offset>\d+)\]")

        for profile_file in profile_files:
            print(f"[PrefetchPlanner] Parsing profiling log: {profile_file}")
            self.loaded_profile_files.append(profile_file)
            with open(profile_file, "r") as f:
                for line in f:
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
                    key = (mode, chunk_idx, origin_offset)
                    aggregator[key].append(avg_ms)

        self.profile_stats = {
            key: statistics.mean(values)
            for key, values in aggregator.items()
        }
        print(f"[PrefetchPlanner] Loaded profiling stats for {len(self.profile_stats)} chunks")

    def generate_partitioned_chunks(self):
        print("[PrefetchPlanner] Generating partitioned weight chunks (logical grouping)")
        # In the real implementation, partitioning may depend on aligned_size thresholds or operators
        partitions = {}
        for mode, chunks in self.weight_chunks.items():
            partitions[mode] = sorted(chunks, key=lambda x: x.aligned_offset)
        return partitions

    def generate_prefetch_plan(self, partitions: Dict[str, List[WeightChunkInfo]]):
        print("[PrefetchPlanner] Generating prefetch plan (logical order)")
        plan = PrefetchPlan(metadata=dict(self.metadata))
        io_order = 0
        for mode, chunks in partitions.items():
            plan.plan_entries[mode] = []
            for chunk in chunks:
                lookup_key = (mode, chunk.chunk_index, chunk.origin_offset)
                chunk_payload = self.chunk_lookup.get(lookup_key)
                if chunk_payload is None:
                    # Fallback to reconstructed payload if lookup fails
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
                plan.plan_entries[mode].append(
                    PrefetchPlanEntry(
                        mode=mode,
                        chunk_data=dict(chunk_payload),
                        io_order=io_order
                    )
                )
                io_order += 1
        return plan

    def integrate_profiling_data(self, plan: PrefetchPlan):
        if not self.profile_stats:
            print("[PrefetchPlanner] No profiling data to integrate; skipping avg_compute_time enrichment")
            return

        print("[PrefetchPlanner] Integrating profiling data into prefetch plan")
        for mode, entries in plan.plan_entries.items():
            for entry in entries:
                key = (mode, entry.chunk_index, entry.origin_offset)
                if key in self.profile_stats:
                    entry.avg_compute_time = self.profile_stats[key]

        print("[PrefetchPlanner] Profiling data integration complete")

    def save_prefetch_plan(self, plan: PrefetchPlan, output_path: str):
        # Create parent directories only if a directory component exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(plan.to_dict(), f, indent=2)
        print(f"[PrefetchPlanner] Prefetch plan saved to {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate Prefetch Plan from Weight Chunk Metadata Table")
    parser.add_argument("--cmt", required=False, help="Path to weight chunks metadata file in json format, which is output from cmt_generator",default="weight_chunks_metadata_table.json")
    parser.add_argument("--output", required=False, help="Path to save prefetch_plan in json format",default="prefetch_plan.json")
    parser.add_argument("--profile-pattern", required=False, help="Glob pattern (relative to the CMT directory) for profiling logs", default="bpf_profile_ops_results_*")
    args = parser.parse_args()

    planner = PrefetchPlanner(args.cmt)
    planner.load_cmt()
    planner.load_profile_logs(args.profile_pattern)
    partitions = planner.generate_partitioned_chunks()
    plan = planner.generate_prefetch_plan(partitions)
    planner.integrate_profiling_data(plan)
    planner.save_prefetch_plan(plan, args.output)


if __name__ == "__main__":
    main()
