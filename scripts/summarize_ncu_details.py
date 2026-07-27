#!/usr/bin/env python3
"""Tabulate the headline metrics from `ncu --page details --csv` exports.

`scripts/ncu_profile_fp6_gemm.sh` writes one details CSV per shard. Each CSV is
one row per metric with the full kernel name repeated on every row, which is
unreadable directly. This pulls the metrics that decide a memory-bound GEMM's
fate into one row per report, plus the warp stall breakdown.

Usage::

    python scripts/summarize_ncu_details.py /tmp/fp6_ncu_*/
    python scripts/summarize_ncu_details.py a.details.csv b.details.csv
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

# (column label, metric name as ncu spells it)
HEADLINE: tuple[tuple[str, str], ...] = (
    ("dur_us", "Duration [us]"),
    ("dram_%", "DRAM Throughput"),
    ("sm_%", "Compute (SM) Throughput"),
    ("mem_TB/s", "Memory Throughput"),
    ("occ_%", "Achieved Occupancy"),
    ("theo_occ_%", "Theoretical Occupancy"),
    ("grid", "Grid Size"),
    ("block", "Block Size"),
    ("waves/SM", "Waves Per SM"),
    ("regs", "Registers Per Thread"),
    ("smem_KB", "Dynamic Shared Memory Per Block"),
    ("lim_smem", "Block Limit Shared Mem"),
    ("lim_regs", "Block Limit Registers"),
    ("lim_warps", "Block Limit Warps"),
    ("L2_hit_%", "L2 Hit Rate"),
    ("L1_hit_%", "L1/TEX Hit Rate"),
)


def _read(path: pathlib.Path) -> tuple[dict[str, str], list[tuple[str, float]]]:
    """Return (metric -> value, sorted stall reasons)."""
    metrics: dict[str, str] = {}
    stalls: list[tuple[str, float]] = []
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("Metric Name") or "").strip()
            value = (row.get("Metric Value") or "").strip()
            if not name:
                continue
            # Duplicate metric names appear in several sections (e.g. "Memory
            # Throughput" in both SOL and Memory Workload) with different units;
            # keep the first, which is the SOL percentage form.
            metrics.setdefault(name, value)
            # ncu picks the unit per magnitude, so a slow kernel reports
            # Gbyte/s where a fast one reports Tbyte/s. Normalize, or the
            # column silently blanks out on exactly the regressions we care
            # about most.
            unit = (row.get("Metric Unit") or "").strip()
            # Same magnitude-dependent unit choice bites Duration: a decode
            # kernel reports usecond, a prefill kernel at the same shape family
            # reports msecond, and reading the raw number makes a 6.5 ms GEMM
            # look like 6.5 us.
            if name == "Duration":
                scale = {
                    "nsecond": 1e-3,
                    "ns": 1e-3,
                    "usecond": 1.0,
                    "us": 1.0,
                    "msecond": 1e3,
                    "ms": 1e3,
                    "second": 1e6,
                    "s": 1e6,
                }.get(unit)
                if scale is not None:
                    try:
                        metrics["Duration [us]"] = (
                            f"{float(value.replace(',', '')) * scale:.2f}"
                        )
                    except ValueError:
                        pass
            if name == "Memory Throughput" and unit in ("Tbyte/s", "Gbyte/s"):
                try:
                    tb = float(value.replace(",", ""))
                except ValueError:
                    continue
                if unit == "Gbyte/s":
                    tb /= 1000.0
                metrics["Memory Throughput [TB/s]"] = f"{tb:.2f}"
            if name.startswith("Stall "):
                try:
                    stalls.append((name[6:], float(value.replace(",", ""))))
                except ValueError:
                    pass
    stalls.sort(key=lambda kv: kv[1], reverse=True)
    return metrics, stalls


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--stalls", type=int, default=6, help="Top stall reasons")
    args = ap.parse_args()

    files: list[pathlib.Path] = []
    for raw in args.paths:
        p = pathlib.Path(raw)
        files.extend(sorted(p.rglob("*.details.csv")) if p.is_dir() else [p])
    if not files:
        sys.exit("no *.details.csv found")

    labels = [lbl for lbl, _ in HEADLINE]
    width = max(len(f.name.replace(".details.csv", "")) for f in files)
    print(f"{'report':{width}s} " + " ".join(f"{l:>10s}" for l in labels))
    all_stalls: list[tuple[str, list[tuple[str, float]]]] = []
    for f in files:
        metrics, stalls = _read(f)
        cells = []
        for _, metric in HEADLINE:
            key = (
                "Memory Throughput [TB/s]"
                if metric == "Memory Throughput"
                else metric
            )
            cells.append(f"{metrics.get(key, '-'):>10s}")
        print(f"{f.name.replace('.details.csv', ''):{width}s} " + " ".join(cells))
        all_stalls.append((f.name.replace(".details.csv", ""), stalls))

    for name, stalls in all_stalls:
        if not stalls:
            continue
        print(f"\n{name} - top warp stall reasons (cycles/instruction):")
        for reason, val in stalls[: args.stalls]:
            print(f"  {val:8.2f}  {reason}")


if __name__ == "__main__":
    main()
