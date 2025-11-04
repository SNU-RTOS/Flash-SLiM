"""Simple planning strategy that mirrors the original per-chunk ordering."""

from __future__ import annotations

from .__planner_data_structures__ import PrefetchPlan, PrefetchPlanEntry
from .strategy_base import PlanningContext, PlanningStrategy
from .common import sort_chunk_list, resolve_chunk_payload


class SimplePlanningStrategy(PlanningStrategy):
    """Replicates the legacy planning behavior without re-chunking."""

    def build(self, context: PlanningContext) -> PrefetchPlan:
        plan = PrefetchPlan(metadata=dict(context.metadata))
        for mode, chunks in context.weight_chunks.items():
            ordered_chunks = sort_chunk_list(chunks)
            plan.plan_entries[mode] = []
            for io_order, chunk in enumerate(ordered_chunks):
                chunk_payload = resolve_chunk_payload(context.chunk_lookup, mode, chunk)
                plan.plan_entries[mode].append(
                    PrefetchPlanEntry(
                        mode=mode, chunk_data=chunk_payload, io_order=io_order
                    )
                )
        return plan
