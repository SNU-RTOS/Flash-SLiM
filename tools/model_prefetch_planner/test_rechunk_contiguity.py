from planning.__planner_data_structures__ import WeightChunkInfo
from planning.strategy_base import PlanningContext
from planning.stratgey_rechunk import RechunkPlanningStrategy
from planning.io_estimator import BandwidthIoTimeEstimator
from planning.common import _sort_chunks, _print_chunk_list, _compute_gap_bytes, _sum_aligned_size

# Construct two non-contiguous chunks (as in your example)
chunk0 = WeightChunkInfo(
    chunk_index=0,
    aligned_offset=3223400448,
    offset_adjust=3200,
    aligned_size=15794176,
    origin_offset=3223403520,
    origin_size=15790080,
    weights_id=198,
    prefetch_mode=0,
    prefetch_mode_str="PREFILL",
)

chunk1 = WeightChunkInfo(
    chunk_index=1,
    aligned_offset=15790080,
    offset_adjust=128,
    aligned_size=9478144,
    origin_offset=15790080,
    origin_size=9474048,
    weights_id=197,
    prefetch_mode=0,
    prefetch_mode_str="PREFILL",
)

context = PlanningContext(
    metadata={"weight_chunk_buffer_size": 1024 * 1024 * 1024},
    weight_chunks={"PREFILL": [chunk0, chunk1]},
    chunk_lookup={},
    profile_stats={},
)

strategy = RechunkPlanningStrategy(
    max_buffer_size=context.metadata["weight_chunk_buffer_size"],
    io_estimator=BandwidthIoTimeEstimator(bandwidth_bytes_per_sec=1e9),
    default_compute_ms=1.0,
)

plan = strategy.build(context)

import json
print(json.dumps(plan.to_dict(), indent=2))
