#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt


def human_bytes(n: int) -> str:
    # 1024-based units
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    f = float(n)
    u = 0
    while f >= 1024.0 and u < len(units) - 1:
        f /= 1024.0
        u += 1
    if u == 0:
        return f"{int(f)} {units[u]}"
    return f"{f:.2f} {units[u]}"


def extract_runtime_ms(fio_json: dict) -> float:
    # fio JSON commonly provides:
    # fio_json["jobs"][0]["read"]["runtime"]  (ms)
    jobs = fio_json.get("jobs", [])
    if not jobs:
        raise ValueError("missing 'jobs' in fio json")
    read_obj = jobs[0].get("read", {})
    rt = read_obj.get("runtime", None)
    if rt is None:
        raise ValueError("missing jobs[0].read.runtime in fio json")
    return float(rt)


def infer_size_from_path(p: Path) -> int:
    # expects .../size_<bytes>/run_XXX.json
    # fallback: parse any "size_<n>" component in parents
    for part in p.parts[::-1]:
        if part.startswith("size_"):
            return int(part.split("_", 1)[1])
    raise ValueError(f"cannot infer size from path: {p}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir", help="Directory produced by run_fio_sweep.sh")
    ap.add_argument("--out", default="fio_size_vs_time.png", help="Output plot filename (PNG)")
    ap.add_argument("--show", action="store_true", help="Show plot window")
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        raise SystemExit(f"Log dir not found: {log_dir}")

    files = sorted(log_dir.rglob("run_*.json"))
    if not files:
        raise SystemExit(f"No run_*.json files found under: {log_dir}")

    by_size = {}  # size_bytes -> list[runtime_ms]
    bad = 0

    for f in files:
        try:
            size = infer_size_from_path(f)
            data = json.loads(f.read_text(encoding="utf-8"))
            rt_ms = extract_runtime_ms(data)
            by_size.setdefault(size, []).append(rt_ms)
        except Exception:
            bad += 1

    sizes = sorted(by_size.keys())
    if not sizes:
        raise SystemExit("Parsed zero valid runs.")

    # Compute stats
    rows = []
    for s in sizes:
        rts = by_size[s]
        rows.append(
            {
                "size_bytes": s,
                "n": len(rts),
                "mean_ms": mean(rts),
                "std_ms": pstdev(rts) if len(rts) > 1 else 0.0,
            }
        )

    # Print a small summary
    print("size_bytes,size_human,n,mean_ms,std_ms")
    for r in rows:
        print(f"{r['size_bytes']},{human_bytes(r['size_bytes'])},{r['n']},{r['mean_ms']:.3f},{r['std_ms']:.3f}")
    if bad:
        print(f"\nWarning: {bad} files could not be parsed.")

    # Plot
    x = [r["size_bytes"] for r in rows]
    y = [r["mean_ms"] for r in rows]

    plt.figure()
    plt.plot(x, y, marker="o")
    # plt.xscale("log", base=2)
    # plt.yscale("log")  # often helpful; remove if you prefer linear y
    plt.xlabel("Read size (MiB)")
    plt.ylabel("Average read runtime (ms)")
    plt.title("fio direct read: size vs average runtime")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"\nSaved plot to: {args.out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
