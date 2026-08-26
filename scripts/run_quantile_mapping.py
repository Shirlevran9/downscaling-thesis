#!/usr/bin/env python3
"""
run_quantile_mapping.py — Run the full quantile-mapping hyper-parameter grid.

Builds the five predictor fields, then for every (predictor, window scheme)
pair writes a wide percentile table, and finally the aggregate tables the
notebook and dashboard read.

    5 predictors x 4 window schemes = 20 percentile tables
    each scored at 5 percentiles                 = 100 combinations

Everything is cached under ``data/cache/``.  Re-running is cheap: a percentile
table that already exists is skipped unless ``--force`` is given.

Usage
-----
    python3 scripts/run_quantile_mapping.py
    python3 scripts/run_quantile_mapping.py --force
    python3 scripts/run_quantile_mapping.py --predictors bilinear knn4
    python3 scripts/run_quantile_mapping.py --aggregates-only
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data_io as dio                     # noqa: E402
from src import qm_metrics as qmm                  # noqa: E402
from src import quantile_windows as qw             # noqa: E402
from src import spatial_ops as sops                # noqa: E402
from src.predictors import PREDICTOR_NAMES, build_all_predictors  # noqa: E402

DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
QM_DIR = CACHE_DIR / "qm"
REGION = dict(south=24, north=38, west=30, east=38)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true",
                   help="rebuild predictor and percentile caches from scratch")
    p.add_argument("--predictors", nargs="+", default=PREDICTOR_NAMES,
                   choices=PREDICTOR_NAMES,
                   help="subset of predictors to process")
    p.add_argument("--schemes", nargs="+", default=qw.SCHEMES,
                   choices=qw.SCHEMES,
                   help="subset of window schemes to process")
    p.add_argument("--aggregates-only", action="store_true",
                   help="skip table building; recompute the aggregates only")
    return p.parse_args()


def load_inputs() -> dict:
    """Load, align and convert both datasets, and derive the static per-pixel table."""
    era5_files = sorted(DATA_DIR.glob("t2m_ERA5land_daily_*.nc"))
    cmip_file = next(DATA_DIR.glob("tas_day_*.nc"))
    if not era5_files:
        raise FileNotFoundError(f"No ERA5-Land files found in {DATA_DIR}")

    print(f"Loading {len(era5_files)} ERA5-Land files …", flush=True)
    era5_ds = dio.load_era5_land(era5_files, region=REGION)
    print(f"Loading CMIP6 {cmip_file.name} …", flush=True)
    cmip_ds = dio.load_cmip6(cmip_file, region=REGION, pad_lat=1.0, pad_lon=1.5)

    era5_ds, cmip_ds, shared_dates = dio.align_calendars(era5_ds, cmip_ds)
    era5_temp = dio.to_celsius(era5_ds["t2m"]).load()
    cmip_tas = dio.to_celsius(cmip_ds["tas"]).load()
    print(f"Aligned on {len(shared_dates)} shared days "
          f"({shared_dates[0]} … {shared_dates[-1]})", flush=True)

    era5_lats = era5_temp.latitude.values
    era5_lons = era5_temp.longitude.values
    lat_name = "lat" if "lat" in cmip_tas.dims else "latitude"
    lon_name = "lon" if "lon" in cmip_tas.dims else "longitude"
    cmip_lats = cmip_tas[lat_name].values
    cmip_lons = cmip_tas[lon_name].values

    land_mask_2d = sops.compute_land_mask(era5_temp)
    n_land = int(land_mask_2d.sum())
    print(f"Grid {len(era5_lats)}x{len(era5_lons)}; {n_land:,} land pixels "
          f"({100 * n_land / land_mask_2d.size:.1f}%)", flush=True)

    return dict(
        era5_temp=era5_temp, cmip_tas=cmip_tas, shared_dates=shared_dates,
        era5_lats=era5_lats, era5_lons=era5_lons,
        cmip_lats=cmip_lats, cmip_lons=cmip_lons,
        land_mask_2d=land_mask_2d,
    )


def build_static_cols(inp: dict, elev_da, dz_da) -> pd.DataFrame:
    """Per-pixel attributes merged into every percentile table.

    Columns: ``lat``, ``lon``, ``elevation`` (m), ``dz`` (sub-grid terrain, m)
    and ``sea_fraction`` of the parent CMIP6 cell.  ``lat``/``lon`` are cast to
    float32 so the merge keys match the percentile tables exactly.
    """
    assign = sops.assign_era5_to_cmip_cells(
        inp["era5_lats"], inp["era5_lons"],
        inp["cmip_lats"], inp["cmip_lons"], inp["land_mask_2d"],
    )
    cell_stats = sops.compute_cell_sea_fraction(
        inp["era5_lats"], inp["era5_lons"],
        inp["cmip_lats"], inp["cmip_lons"], inp["land_mask_2d"],
    )
    assign = assign.merge(
        cell_stats[["cmip_lat", "cmip_lon", "sea_fraction"]],
        on=["cmip_lat", "cmip_lon"], how="left",
    )

    rows, cols = np.where(inp["land_mask_2d"])
    static = pd.DataFrame(
        {
            "lat": inp["era5_lats"][rows].astype(np.float32),
            "lon": inp["era5_lons"][cols].astype(np.float32),
            "elevation": elev_da.values[rows, cols].astype(np.float32),
            "dz": dz_da.values[rows, cols].astype(np.float32),
        }
    )
    # assign is built from the same np.where ordering, so the columns line up.
    static["cmip_lat"] = assign["cmip_lat"].to_numpy(np.float32)
    static["cmip_lon"] = assign["cmip_lon"].to_numpy(np.float32)
    static["sea_fraction"] = assign["sea_fraction"].to_numpy(np.float32)
    return static


def main() -> int:
    args = parse_args()
    QM_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    inp = load_inputs()

    print("\nBuilding predictor fields …", flush=True)
    fields, meta = build_all_predictors(
        cmip_tas=inp["cmip_tas"],
        era5_temp=inp["era5_temp"],
        land_mask_2d=inp["land_mask_2d"],
        region=REGION,
        dates=inp["shared_dates"],
        cache_dir=CACHE_DIR,
        force=args.force,
    )

    static = build_static_cols(inp, meta["elevation"], meta["dz"])
    static.to_parquet(QM_DIR / "static_pixels.parquet", index=False)

    lapse_df = pd.DataFrame(
        [
            {"season": k, "gamma_per_m": meta["lapse_rates"][k],
             "gamma_per_km": 1000 * meta["lapse_rates"][k],
             "n": meta["lapse_rates"][f"n_{k}"]}
            for k in ("all", "DJF", "MAM", "JJA", "SON")
        ]
    )
    lapse_df.to_parquet(QM_DIR / "lapse_rates.parquet", index=False)

    static_merge = static[["lat", "lon", "elevation", "dz", "sea_fraction"]]

    if not args.aggregates_only:
        print("\nBuilding percentile tables …", flush=True)
        for predictor in args.predictors:
            for scheme in args.schemes:
                path = qmm.pct_table_path(QM_DIR, predictor, scheme)
                if path.exists() and not args.force:
                    print(f"  · {path.name} exists — skipped", flush=True)
                    continue
                t0 = time.time()
                table = qw.build_percentile_table(
                    era5_temp=inp["era5_temp"],
                    predictor_da=fields[predictor],
                    land_mask_2d=inp["land_mask_2d"],
                    dates=inp["shared_dates"],
                    scheme=scheme,
                    percentiles=qw.PERCENTILES,
                    static_cols=static_merge,
                )
                table.to_parquet(path, index=False)
                print(f"  ✓ {path.name}  {len(table):,} rows  "
                      f"({time.time() - t0:.1f} s)", flush=True)
                del table

    print("\nBuilding aggregate tables …", flush=True)
    qmm.build_all_aggregates(
        qm_dir=QM_DIR,
        predictors=args.predictors,
        schemes=args.schemes,
        percentiles=qw.PERCENTILES,
    )

    print(f"\nDone in {(time.time() - t_start) / 60:.1f} min. "
          f"Output in {QM_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
