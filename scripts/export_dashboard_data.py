#!/usr/bin/env python3
"""
export_dashboard_data.py — Build the committed data set the deployed dashboard reads.

The full quantile-mapping cache is about 700 MB, almost all of it in the 20
per-pixel percentile tables. That cannot live in a git repository, but it is
not all needed: the five aggregate tables total about 11 MB and drive three of
the dashboard's five views on their own.

This script copies the aggregates plus a chosen subset of the percentile tables
into ``data/deploy/``, which is committed and is the directory the dashboard
falls back to when the full cache is absent (as it is on Streamlit Cloud).

Default subset: the ``year`` scheme only, giving roughly 26 MB in total. The
point-level views (Scatter, and Maps at "Selected window" scope) then offer
annual windows; every other view keeps all four window lengths, because the
aggregates were computed over the full grid.

Usage
-----
    python3 scripts/export_dashboard_data.py
    python3 scripts/export_dashboard_data.py --schemes year quarter
    python3 scripts/export_dashboard_data.py --schemes ""      # aggregates only
"""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import qm_metrics as qmm              # noqa: E402
from src import quantile_windows as qw         # noqa: E402
from src.predictors import PREDICTOR_NAMES     # noqa: E402

QM_DIR = ROOT / "data" / "cache" / "qm"
DEPLOY_DIR = ROOT / "data" / "deploy"

#: Small tables that are always copied.
ALWAYS = list(qmm.METRIC_FILES.values()) + [
    "static_pixels.parquet",
    "lapse_rates.parquet",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--schemes", nargs="*", default=["year"],
        help="window schemes whose percentile tables to include "
             f"(any of {qw.SCHEMES}); pass no values for aggregates only",
    )
    p.add_argument("--clean", action="store_true",
                   help="delete the deploy directory first")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    schemes = [s for s in args.schemes if s]
    bad = set(schemes) - set(qw.SCHEMES)
    if bad:
        raise SystemExit(f"Unknown scheme(s): {sorted(bad)}; expected {qw.SCHEMES}")

    if not QM_DIR.exists():
        raise SystemExit(
            f"No cache at {QM_DIR}. Run scripts/run_quantile_mapping.py first."
        )

    if args.clean and DEPLOY_DIR.exists():
        shutil.rmtree(DEPLOY_DIR)
    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)

    copied, total = [], 0
    missing = []

    for name in ALWAYS:
        src = QM_DIR / name
        if not src.exists():
            missing.append(name)
            continue
        shutil.copy2(src, DEPLOY_DIR / name)
        copied.append((name, src.stat().st_size))
        total += src.stat().st_size

    for predictor in PREDICTOR_NAMES:
        for scheme in schemes:
            src = qmm.pct_table_path(QM_DIR, predictor, scheme)
            if not src.exists():
                missing.append(src.name)
                continue
            shutil.copy2(src, DEPLOY_DIR / src.name)
            copied.append((src.name, src.stat().st_size))
            total += src.stat().st_size

    for name, size in copied:
        print(f"  {size / 1048576:8.2f} MB  {name}")
    print(f"\n  {total / 1048576:8.2f} MB  total in {DEPLOY_DIR}")

    if missing:
        print(f"\n  warning: {len(missing)} file(s) not found in the cache:")
        for name in missing:
            print(f"    - {name}")

    included = ", ".join(schemes) if schemes else "none"
    print(f"\nPercentile tables included: {included}")
    print("Point-level views in the deployed app are limited to these windows;")
    print("every other view keeps all four, from the aggregates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
