from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .__planner_data_structures__ import PrefetchPlan, PrefetchPlanEntry, WeightChunkInfo
from .strategy_base import PlanningContext, PlanningStrategy
from .io_estimator import IoTimeEstimator
from .common import sort_chunk_list, resolve_chunk_payload


@dataclass
class SizeOnlyStrategy(PlanningStrategy):
    """
    Center-out segmentation with alternating target capacities around a pivot chunk.

    It enforces a circular adjacency constraint:
      size(g_t) + size(g_(t+1 mod N)) <= max_buffer_size  for all t
    including the wrap-around pair (last + first).

    If rotation alone cannot satisfy circularity, it tries to fix ONLY the wrap-around
    violation by splitting either the first or last group into two groups along chunk
    boundaries (no partial chunk splitting). This matches:
      "if first and last are both ~y, split either endpoint group if it has 2+ chunks".
    """

    max_buffer_size: int
    io_estimator: IoTimeEstimator
    default_compute_ms: float = 0.0

    # Pivot selection: if None, pick largest chunk in DECODE
    pivot_chunk_index: Optional[int] = None

    # When packing rings, pick left neighbor first if True, else right first
    prefer_left: bool = True

    # Non-DECODE handling
    keep_non_decode_singletons: bool = True

    # If True, after splitting endpoint to fix wrap-around, we also re-check full circularity.
    # (Recommended True)
    validate_all_edges_after_fix: bool = True

    def build(self, context: PlanningContext) -> PrefetchPlan:
        self._validate_chunks(context)
        plan = PrefetchPlan(metadata=dict(context.metadata))

        for mode, chunks in context.weight_chunks.items():
            ordered = sort_chunk_list(chunks)

            if mode == "DECODE":
                groups, group_io_times = self._group_decode_center_out_circular(mode, ordered)
            else:
                if self.keep_non_decode_singletons:
                    groups = [[c] for c in ordered]
                else:
                    groups = [[c] for c in ordered]
                group_io_times = [self._estimate_group_io_time(mode, g) for g in groups]

            plan.plan_entries[mode] = []
            for group_idx, group in enumerate(groups):
                group_io_time = group_io_times[group_idx] if group_idx < len(group_io_times) else None
                for chunk in group:
                    chunk_payload = resolve_chunk_payload(context.chunk_lookup, mode, chunk)
                    key = (mode, chunk.chunk_index, chunk.origin_offset)
                    compute_ms = context.profile_stats.get(key, self.default_compute_ms)

                    plan.plan_entries[mode].append(
                        PrefetchPlanEntry(
                            mode=mode,
                            chunk_data=chunk_payload,
                            io_order=group_idx,  # group id
                            avg_compute_time=compute_ms,
                            estimated_io_time_ms=group_io_time,
                        )
                    )

        return plan

    # ---------------------------------------------------------------------
    # DECODE grouping: center-out alternating budgets + circular handling
    # ---------------------------------------------------------------------
    def _group_decode_center_out_circular(
        self,
        mode: str,
        chunks: Sequence[WeightChunkInfo],
    ) -> Tuple[List[List[WeightChunkInfo]], List[float]]:
        if not chunks:
            return [], []

        # 1) Build center-out groups (position based), then convert to chunk groups
        groups, io_times = self._group_center_out_alternating(mode, chunks)

        # 2) Order groups by first chunk position so they execute sequentially
        zipped = list(zip(groups, io_times))
        zipped.sort(key=lambda t: self._group_first_pos(chunks, t[0]))
        groups = [g for g, _ in zipped]
        io_times = [t for _, t in zipped]

        x = int(self.max_buffer_size)

        # 3) Try rotate to satisfy full circular adjacency
        rot = self._try_rotate_to_make_circular_ok(groups, io_times, x)
        if rot is not None:
            groups, io_times = rot
            if self.validate_all_edges_after_fix:
                self._assert_circular_ok(groups, x)
            return groups, io_times

        # 4) Rotation failed. You asked: "for now simply break one that harms circularity into two".
        # Here we focus on wrap-around (last + first). We split first or last group if multi-chunk.
        groups, io_times = self._fix_wraparound_by_splitting_endpoints(mode, groups, io_times, x)

        # 5) After splitting, rotation should typically succeed. Try again.
        rot2 = self._try_rotate_to_make_circular_ok(groups, io_times, x)
        if rot2 is None:
            # If you only care about wrap-around, you could return groups here.
            # But generally you want all circular edges to be valid.
            raise ValueError(
                "Still cannot satisfy circular adjacency after endpoint split. "
                "This means at least one internal adjacent pair also exceeds the buffer, "
                "or endpoint splitting could not create a feasible boundary."
            )

        groups, io_times = rot2
        if self.validate_all_edges_after_fix:
            self._assert_circular_ok(groups, x)
        return groups, io_times

    def _group_center_out_alternating(
        self,
        mode: str,
        chunks: Sequence[WeightChunkInfo],
    ) -> Tuple[List[List[WeightChunkInfo]], List[float]]:
        """
        Center-out rings around pivot with alternating target budgets:
          Group 0: pivot alone (size y)
          Group 1: pack neighbors to ~ (x - y)
          Group 2: pack next neighbors to ~ y
          Group 3: pack next neighbors to ~ (x - y)
          ...

        This is a size-driven packer for testing. It does not check overlap hiding.
        """
        pivot_pos = self._select_pivot_pos(mode, chunks)
        pivot = chunks[pivot_pos]

        x = int(self.max_buffer_size)
        y = int(pivot.aligned_size)
        if y > x:
            raise ValueError(f"Pivot chunk size y={y} exceeds max_buffer_size x={x}")

        cap_small = x - y  # target for odd groups

        groups_pos: List[List[int]] = []
        groups_pos.append([pivot_pos])

        left = pivot_pos - 1
        right = pivot_pos + 1
        group_id = 1

        while left >= 0 or right < len(chunks):
            target = cap_small if (group_id % 2 == 1) else y
            cur: List[int] = []
            cur_bytes = 0

            def try_take_from_left() -> bool:
                nonlocal left, cur_bytes
                if left < 0:
                    return False
                s = int(chunks[left].aligned_size)
                if cur and (cur_bytes + s > target):
                    return False
                cur.append(left)
                cur_bytes += s
                left -= 1
                return True

            def try_take_from_right() -> bool:
                nonlocal right, cur_bytes
                if right >= len(chunks):
                    return False
                s = int(chunks[right].aligned_size)
                if cur and (cur_bytes + s > target):
                    return False
                cur.append(right)
                cur_bytes += s
                right += 1
                return True

            while True:
                picked = (
                    try_take_from_left() or try_take_from_right()
                    if self.prefer_left
                    else try_take_from_right() or try_take_from_left()
                )
                if not picked:
                    break
                if cur_bytes == target:
                    break

            if not cur:
                # Avoid infinite loop: consume one remaining chunk
                if left >= 0:
                    cur = [left]
                    left -= 1
                elif right < len(chunks):
                    cur = [right]
                    right += 1
                else:
                    break

            groups_pos.append(cur)
            group_id += 1

        groups: List[List[WeightChunkInfo]] = []
        io_times: List[float] = []
        for pos_list in groups_pos:
            pos_sorted = sorted(pos_list)
            g = [chunks[p] for p in pos_sorted]
            groups.append(g)
            io_times.append(self._estimate_group_io_time(mode, g))

        return groups, io_times

    # ---------------------------------------------------------------------
    # Circular adjacency helpers
    # ---------------------------------------------------------------------
    def _group_bytes(self, g: List[WeightChunkInfo]) -> int:
        return sum(int(c.aligned_size) for c in g)

    def _all_adjacent_ok(self, groups: List[List[WeightChunkInfo]], x: int) -> bool:
        if not groups:
            return True
        sizes = [self._group_bytes(g) for g in groups]
        n = len(sizes)
        for t in range(n):
            if sizes[t] + sizes[(t + 1) % n] > x:
                return False
        return True

    def _assert_circular_ok(self, groups: List[List[WeightChunkInfo]], x: int) -> None:
        if not groups:
            return
        sizes = [self._group_bytes(g) for g in groups]
        n = len(sizes)
        for t in range(n):
            s = sizes[t] + sizes[(t + 1) % n]
            if s > x:
                raise ValueError(
                    f"Circular adjacency violation at {t}->{(t+1)%n}: "
                    f"{sizes[t]} + {sizes[(t+1)%n]} = {s} > {x}"
                )

    def _try_rotate_to_make_circular_ok(
        self,
        groups: List[List[WeightChunkInfo]],
        io_times: List[float],
        x: int,
    ) -> Optional[Tuple[List[List[WeightChunkInfo]], List[float]]]:
        n = len(groups)
        if n <= 1:
            return (groups, io_times)

        for r in range(n):
            rot_g = groups[r:] + groups[:r]
            rot_t = io_times[r:] + io_times[:r]
            if self._all_adjacent_ok(rot_g, x):
                return (rot_g, rot_t)
        return None

    # ---------------------------------------------------------------------
    # Endpoint split to fix wrap-around only (no partial chunk splitting)
    # ---------------------------------------------------------------------
    def _split_group_prefix_to_fit(
        self,
        g: List[WeightChunkInfo],
        prefix_cap: int,
    ) -> Optional[Tuple[List[WeightChunkInfo], List[WeightChunkInfo]]]:
        """
        Split g into (prefix, suffix) so that:
          size(prefix) <= prefix_cap
        and both parts non-empty. Choose the largest prefix that fits.
        """
        if len(g) < 2:
            return None

        acc = 0
        cut = 0
        for i, c in enumerate(g):
            s = int(c.aligned_size)
            if acc + s <= prefix_cap:
                acc += s
                cut = i + 1
            else:
                break

        if cut <= 0 or cut >= len(g):
            return None
        return g[:cut], g[cut:]

    def _split_group_suffix_to_fit(
        self,
        g: List[WeightChunkInfo],
        suffix_cap: int,
    ) -> Optional[Tuple[List[WeightChunkInfo], List[WeightChunkInfo]]]:
        """
        Split g into (prefix, suffix) so that:
          size(suffix) <= suffix_cap
        and both parts non-empty. Choose the largest suffix that fits.
        """
        if len(g) < 2:
            return None

        acc = 0
        cut = len(g)
        for i in range(len(g) - 1, -1, -1):
            s = int(g[i].aligned_size)
            if acc + s <= suffix_cap:
                acc += s
                cut = i
            else:
                break

        if cut <= 0 or cut >= len(g):
            return None
        return g[:cut], g[cut:]

    def _fix_wraparound_by_splitting_endpoints(
        self,
        mode: str,
        groups: List[List[WeightChunkInfo]],
        io_times: List[float],
        x: int,
    ) -> Tuple[List[List[WeightChunkInfo]], List[float]]:
        """
        Fix only the wrap-around pair (last + first) by splitting either first or last group
        into two groups, if feasible (group must have >= 2 chunks).

        After the split, we recompute IO times for the modified groups only.
        """
        if len(groups) <= 1:
            return groups, io_times

        first = groups[0]
        last = groups[-1]
        s_first = self._group_bytes(first)
        s_last = self._group_bytes(last)

        if s_first + s_last <= x:
            return groups, io_times

        cap_first = x - s_last   # need size(first_part) <= cap_first
        cap_last = x - s_first   # need size(last_part)  <= cap_last

        # Try split FIRST so that last + first_a <= x
        split_first = self._split_group_prefix_to_fit(first, prefix_cap=cap_first)
        if split_first is not None:
            first_a, first_b = split_first
            new_groups = [first_a, first_b] + groups[1:]

            new_io_times = (
                [self._estimate_group_io_time(mode, first_a),
                 self._estimate_group_io_time(mode, first_b)]
                + io_times[1:]
            )
            return new_groups, new_io_times

        # Try split LAST so that last_b + first <= x
        split_last = self._split_group_suffix_to_fit(last, suffix_cap=cap_last)
        if split_last is not None:
            last_a, last_b = split_last
            new_groups = groups[:-1] + [last_a, last_b]

            new_io_times = (
                io_times[:-1]
                + [self._estimate_group_io_time(mode, last_a),
                   self._estimate_group_io_time(mode, last_b)]
            )
            return new_groups, new_io_times

        raise ValueError(
            f"Cannot fix wrap-around by splitting endpoints. "
            f"size(last)={s_last}, size(first)={s_first}, x={x}. "
            f"Need first or last segment to have >=2 chunks with some subset <= (x - other)."
        )

    # ---------------------------------------------------------------------
    # Misc helpers
    # ---------------------------------------------------------------------
    def _select_pivot_pos(self, mode: str, chunks: Sequence[WeightChunkInfo]) -> int:
        if self.pivot_chunk_index is None:
            return max(range(len(chunks)), key=lambda p: int(chunks[p].aligned_size))
        for p, c in enumerate(chunks):
            if c.chunk_index == self.pivot_chunk_index:
                return p
        raise ValueError(f"pivot_chunk_index {self.pivot_chunk_index} not found in {mode} chunks")

    def _group_first_pos(self, chunks: Sequence[WeightChunkInfo], g: List[WeightChunkInfo]) -> int:
        first = g[0]
        for i, c in enumerate(chunks):
            if c is first:
                return i
        for i, c in enumerate(chunks):
            if c.chunk_index == first.chunk_index and c.origin_offset == first.origin_offset:
                return i
        return 0

    def _estimate_group_io_time(self, mode: str, chunk_group: List[WeightChunkInfo]) -> float:
        if not chunk_group:
            return 0.0
        try:
            return float(self.io_estimator.estimate(mode, chunk_group, gap_bytes=0))
        except Exception:
            return 0.0

    def _validate_chunks(self, context: PlanningContext) -> None:
        if self.max_buffer_size <= 0:
            raise ValueError("max_buffer_size must be positive")
        for mode, chunks in context.weight_chunks.items():
            for chunk in chunks:
                if int(chunk.aligned_size) > int(self.max_buffer_size):
                    raise ValueError(
                        f"Chunk {mode}[{chunk.chunk_index}] size {chunk.aligned_size} exceeds buffer {self.max_buffer_size}"
                    )
