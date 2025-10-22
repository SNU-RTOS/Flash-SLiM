#!/usr/bin/env python3
# prefetch_planner.py
# Prototype version of Prefetch Planner
# Author: GeonHa Park
# Description:
#   Generate a Prefetch Plan based on Weight Chunk Metadata Table (CMT)
#   following the architecture defined in Flash-SLiM Prefetch Planner Design.

import json
from dataclasses import dataclass, field
from typing import List, Dict, Any
import os


@dataclass
class WeightChunkInfo:
    chunk_index: int
    aligned_offset: int
    offset_adjust: int
    aligned_size: int
    origin_offset: int
    origin_size: int
    managed_buffer_index: int
    weights_id: int
    prefetch_mode: int
    prefetch_mode_str: str


@dataclass
class PrefetchPlanEntry:
    chunk_index: int
    mode: str
    aligned_offset: int
    aligned_size: int
    io_order: int
    # TODO: add profiling latency or dependency hints later
    score: float = 0.0


@dataclass
class PrefetchPlan:
    version: str = "1.0"
    plan_entries: List[PrefetchPlanEntry] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "prefetch_plan": [entry.__dict__ for entry in self.plan_entries]
        }


class PrefetchPlanner:
    def __init__(self, cmt_path: str):
        self.cmt_path = cmt_path
        self.metadata = {}
        self.weight_chunks: Dict[str, List[WeightChunkInfo]] = {}

    def load_cmt(self):
        print(f"[PrefetchPlanner] Loading Weight Chunk Metadata Table from: {self.cmt_path}")
        with open(self.cmt_path, "r") as f:
            cmt_data = json.load(f)
        self.metadata = cmt_data.get("metadata", {})
        raw_chunks = cmt_data.get("weight_chunks", {})
        for mode, chunks in raw_chunks.items():
            self.weight_chunks[mode] = [WeightChunkInfo(**c) for c in chunks]
        print(f"[PrefetchPlanner] Loaded {sum(len(v) for v in self.weight_chunks.values())} chunks")

    def generate_partitioned_chunks(self):
        print("[PrefetchPlanner] Generating partitioned weight chunks (logical grouping)")
        # In the real implementation, partitioning may depend on aligned_size thresholds or operators
        partitions = {}
        for mode, chunks in self.weight_chunks.items():
            partitions[mode] = sorted(chunks, key=lambda x: x.aligned_offset)
        return partitions

    def generate_prefetch_plan(self, partitions: Dict[str, List[WeightChunkInfo]]):
        print("[PrefetchPlanner] Generating prefetch plan (logical order)")
        plan = PrefetchPlan()
        io_order = 0
        for mode, chunks in partitions.items():
            for chunk in chunks:
                entry = PrefetchPlanEntry(
                    chunk_index=chunk.chunk_index,
                    mode=mode,
                    aligned_offset=chunk.aligned_offset,
                    aligned_size=chunk.aligned_size,
                    io_order=io_order
                )
                plan.plan_entries.append(entry)
                io_order += 1
        return plan

    def integrate_profiling_data(self, plan: PrefetchPlan):
        print("[PrefetchPlanner] Integrating profiling data (template only)")
        # TODO: integrate eBPF profiling data or operator latency mapping here
        pass

    def save_prefetch_plan(self, plan: PrefetchPlan, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(plan.to_dict(), f, indent=2)
        print(f"[PrefetchPlanner] Prefetch plan saved to {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate Prefetch Plan from Weight Chunk Metadata Table")
    parser.add_argument("--cmt", required=True, help="Path to weight chunks metadata file in json format, which is output from cmt_generator",default="weight_chunks_metadata.json")
    parser.add_argument("--output", required=True, help="Path to save prefetch_plan in json format",default="prefetch_plan.json")
    args = parser.parse_args()

    planner = PrefetchPlanner(args.cmt)
    planner.load_cmt()
    partitions = planner.generate_partitioned_chunks()
    plan = planner.generate_prefetch_plan(partitions)
    planner.integrate_profiling_data(plan)
    planner.save_prefetch_plan(plan, args.output)


if __name__ == "__main__":
    main()
