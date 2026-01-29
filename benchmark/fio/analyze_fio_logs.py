#!/usr/bin/env python3
"""
analyze_fio_logs.py

Recursively parses fio JSON logs under a root directory and compares configurations.

It builds:
  results[config_key][size_bytes] = {
      "n": int,
      "mean_ms": float,
      "std_ms": float,
      "values_ms": [float, ...]
  }

Logging (requested):
  - Prints a size-centric ranking:
      For each size (ascending), list configurations sorted by mean_ms (fastest -> slowest).

Plot:
  - x-axis: size (bytes, log2)
  - y-axis: mean latency (ms)
  - one curve per configuration (or limited with --max-curves)
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt


SIZE_RE = re.compile(r"^size_(\d+)$")
BS_RE = re.compile(r"^bs_(.+)$")
QD_RE = re.compile(r"^qd_(\d+)$")
DIRECT_RE = re.compile(r"^direct_(\d+)$")


@dataclass(frozen=True)
class ConfigKey:
    ioengine: str
    bs: str
    qd: int
    direct: int

    def label(self) -> str:
        return f"{self.ioengine} bs={self.bs} qd={self.qd} direct={self.direct}"


def human_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    f = float(n)
    u = 0
    while f >= 1024.0 and u < len(units) - 1:
        f /= 1024.0
        u += 1
    if u == 0:
        return f"{int(f)} {units[u]}"
    return f"{f:.2f} {units[u]}"


def is_exact_mib(n: int) -> bool:
    return n % (1024 * 1024) == 0


def extract_read_runtime_ms(fio_json: dict) -> float:
    # fio JSON: jobs[0].read.runtime (ms)
    jobs = fio_json.get("jobs", [])
    if not jobs:
        raise ValueError("missing 'jobs'")
    read_obj = jobs[0].get("read", None)
    if not isinstance(read_obj, dict):
        raise ValueError("missing 'jobs[0].read'")
    rt = read_obj.get("runtime", None)
    if rt is None:
        raise ValueError("missing 'jobs[0].read.runtime'")
    return float(rt)


def infer_size_bytes(p: Path) -> int:
    for part in p.parts[::-1]:
        m = SIZE_RE.match(part)
        if m:
            return int(m.group(1))
    raise ValueError(f"cannot infer size from path: {p}")


def infer_config(p: Path, root: Path) -> ConfigKey:
    rel = p.relative_to(root)
    parts = rel.parts

    bs: Optional[str] = None
    qd: Optional[int] = None
    direct: Optional[int] = None
    ioengine: Optional[str] = None

    for part in parts:
        m = BS_RE.match(part)
        if m:
            bs = m.group(1)
            continue
        m = QD_RE.match(part)
        if m:
            qd = int(m.group(1))
            continue
        m = DIRECT_RE.match(part)
        if m:
            direct = int(m.group(1))
            continue
        if SIZE_RE.match(part):
            continue
        if part.startswith("run_") or part.endswith(".json"):
            continue
        if ioengine is None:
            ioengine = part

    if ioengine is None:
        ioengine = "unknown"
    if bs is None:
        bs = "unknown"
    if qd is None:
        qd = -1
    if direct is None:
        direct = -1

    return ConfigKey(ioengine=ioengine, bs=bs, qd=qd, direct=direct)


def build_results(root: Path) -> Tuple[Dict[ConfigKey, Dict[int, dict]], int]:
    results: Dict[ConfigKey, Dict[int, dict]] = {}
    bad = 0

    files = sorted(root.rglob("run_*.json"))
    if not files:
        raise SystemExit(f"No run_*.json files found under: {root}")

    raw: Dict[ConfigKey, Dict[int, List[float]]] = {}

    for f in files:
        try:
            size_b = infer_size_bytes(f)
            cfg = infer_config(f, root)

            data = json.loads(f.read_text(encoding="utf-8"))
            rt_ms = extract_read_runtime_ms(data)

            raw.setdefault(cfg, {}).setdefault(size_b, []).append(rt_ms)
        except Exception:
            bad += 1

    for cfg, size_map in raw.items():
        results[cfg] = {}
        for size_b, vals in size_map.items():
            results[cfg][size_b] = {
                "n": len(vals),
                "mean_ms": mean(vals),
                "std_ms": pstdev(vals) if len(vals) > 1 else 0.0,
                "values_ms": vals,
            }

    return results, bad


def save_summary_json(results: Dict[ConfigKey, Dict[int, dict]], out_path: Path) -> None:
    obj = {}
    for cfg, size_map in results.items():
        obj[cfg.label()] = {
            str(size_b): {
                "n": st["n"],
                "mean_ms": st["mean_ms"],
                "std_ms": st["std_ms"],
            }
            for size_b, st in sorted(size_map.items())
        }
    out_path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def plot_results(
    results: Dict[ConfigKey, Dict[int, dict]],
    out_png: Path,
    max_curves: int,
    ylog: bool,
) -> None:
    cfgs = sorted(
        results.keys(),
        key=lambda c: (len(results[c]), c.label()),
        reverse=True,
    )
    if max_curves > 0:
        cfgs = cfgs[:max_curves]

    plt.figure()

    for cfg in cfgs:
        size_map = results[cfg]
        xs = sorted(size_map.keys())
        ys = [size_map[x]["mean_ms"] for x in xs]
        plt.plot(xs, ys, marker="o", label=cfg.label())

    plt.xscale("log", base=2)
    if ylog:
        plt.yscale("log")

    plt.xlabel("Read size (bytes, log2)")
    plt.ylabel("Mean latency (ms)")
    plt.title("fio latency vs size by configuration")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(fontsize="x-small", ncol=1, loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)


def print_rankings_by_size(results: Dict[ConfigKey, Dict[int, dict]], only_mib: bool) -> None:
    # Build set of all sizes observed
    all_sizes = set()
    for cfg, size_map in results.items():
        all_sizes.update(size_map.keys())

    for size_b in sorted(all_sizes):
        if only_mib and not is_exact_mib(size_b):
            continue

        # Collect (mean, cfg, stats) for this size where present
        rows = []
        for cfg, size_map in results.items():
            st = size_map.get(size_b)
            if st is None:
                continue
            rows.append((st["mean_ms"], cfg, st))

        if not rows:
            continue

        rows.sort(key=lambda t: t[0])  # fastest -> slowest

        print(f"\n=== SIZE {size_b} bytes ({human_bytes(size_b)}) : fastest -> slowest ===")
        print("rank,mean_ms,std_ms,n,config")
        for i, (m, cfg, st) in enumerate(rows, start=1):
            print(f"{i},{m:.3f},{st['std_ms']:.3f},{st['n']},\"{cfg.label()}\"")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log_root", help="Root directory that contains fio logs (macro output root)")
    ap.add_argument("--out-png", default="fio_config_compare.png", help="Output plot file (PNG)")
    ap.add_argument("--out-json", default="fio_config_summary.json", help="Output summary file (JSON)")
    ap.add_argument(
        "--max-curves",
        type=int,
        default=0,
        help="Limit number of plotted configurations (0 = plot all)",
    )
    ap.add_argument(
        "--ylog",
        action="store_true",
        help="Use log scale for y-axis (latency). Useful when curves vary widely.",
    )
    ap.add_argument(
        "--rank-only-mib",
        action="store_true",
        help="Only print rankings for sizes that are exact multiples of 1 MiB.",
    )
    args = ap.parse_args()

    root = Path(args.log_root)
    if not root.exists():
        raise SystemExit(f"log_root not found: {root}")

    results, bad = build_results(root)

    save_summary_json(results, Path(args.out_json))

    print(f"Parsed configurations: {len(results)}")
    if bad:
        print(f"Warning: {bad} files could not be parsed.")

    # Requested: sorted logging per size (fastest -> slowest)
    print_rankings_by_size(results, only_mib=args.rank_only_mib)

    plot_results(results, Path(args.out_png), max_curves=args.max_curves, ylog=args.ylog)
    print(f"\nWrote summary JSON: {args.out_json}")
    print(f"Wrote plot PNG: {args.out_png}")


if __name__ == "__main__":
    main()
