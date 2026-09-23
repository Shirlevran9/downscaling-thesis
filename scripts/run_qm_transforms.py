#!/usr/bin/env python3
"""
run_qm_transforms.py — Fit and cross-validate the quantile-mapping transforms.

One predictor (bilinear), one distribution window (the meteorological season),
eight transforms, leave-one-season-year-out cross-validation.  Everything is
written to ``data/cache/qm/transforms/`` as parquet, and the notebook reads
those aggregates rather than refitting anything.

This is **not** the same experiment as ``run_quantile_mapping.py``, which stays
in place.  That script compares five *predictors* across four window lengths and
fits no correction; this one compares eight *transforms* on a single predictor
and scores them out of sample.  It also depends on that script having run, since
it reuses ``static_pixels.parquet``.

Usage
-----
    python3.10 scripts/run_qm_transforms.py
    python3.10 scripts/run_qm_transforms.py --pixel-stride 10 --methods quant linear
    python3.10 scripts/run_qm_transforms.py --force
    python3.10 scripts/run_qm_transforms.py --aggregates-only

Use ``python3.10``, not ``python3``: the latter has no xarray on this machine.
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import qm_cv as qcv                              # noqa: E402
from src import qm_eval as qev                            # noqa: E402
from src import qm_inputs as qin                          # noqa: E402
from src import qm_nodes as qn                            # noqa: E402
from src import quantile_windows as qw                    # noqa: E402
from src.qm_transforms import RAW_METHOD, TRANSFORMS      # noqa: E402

DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
QM_DIR = CACHE_DIR / "qm"
OUT_DIR = QM_DIR / "transforms"

#: Land pixels expected on a full run; guards against a silent data change.
EXPECTED_LAND_PIXELS = 7683

log = logging.getLogger("run_qm_transforms")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--methods", nargs="+", default=list(TRANSFORMS),
                   choices=list(TRANSFORMS),
                   help="transforms to run; 'raw' is always included")
    p.add_argument("--n-nodes", type=int, default=99,
                   help="q-q nodes used to fit (default 99)")
    p.add_argument("--extrap", default="boe", choices=["boe", "clip", "native"],
                   help="rule for values outside the training range")
    p.add_argument("--min-days", type=int, default=qn.MIN_TRAIN_DAYS,
                   help="minimum training days for a pixel to be fitted")
    p.add_argument("--percentiles", nargs="+", type=int, default=list(qw.PERCENTILES),
                   help="percentiles to evaluate")
    p.add_argument("--seasons", nargs="+", default=list(qn.SEASONS),
                   choices=list(qn.SEASONS), help="seasons to run")
    p.add_argument("--pixel-stride", type=int, default=1,
                   help="keep every Nth land pixel (dev mode)")
    p.add_argument("--max-pixels", type=int, default=0,
                   help="cap the pixel count after striding (0 = no cap)")
    p.add_argument("--sample-pixels", type=int, default=qcv.SAMPLE_PIXELS,
                   help="pixels sampled when choosing a hyper-parameter")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-hyper-search", action="store_true",
                   help="use each transform's default hyper-parameters")
    p.add_argument("--force", action="store_true",
                   help="refit methods whose output already exists")
    p.add_argument("--aggregates-only", action="store_true",
                   help="skip fitting; rebuild the aggregate tables only")
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def config_of(args: argparse.Namespace, n_pixels: int) -> dict:
    """The configuration that the artefacts on disk must agree with."""
    return {
        "predictor": "bilinear",
        "scheme": qw.SEASON_SCHEME,
        "n_nodes": int(args.n_nodes),
        "extrap": args.extrap,
        "min_days": int(args.min_days),
        "percentiles": [int(q) for q in args.percentiles],
        "seasons": list(args.seasons),
        "pixel_stride": int(args.pixel_stride),
        "max_pixels": int(args.max_pixels),
        "n_pixels": int(n_pixels),
        "hyper_search": not args.no_hyper_search,
        "sample_pixels": int(args.sample_pixels),
        "seed": int(args.seed),
    }


def check_manifest(out_dir: Path, cfg: dict, force: bool) -> None:
    """Refuse to mix artefacts built under different settings.

    Without this a ``--pixel-stride 10`` development run and a full run would
    quietly coexist in the same directory, and the aggregates would be built
    from a mixture of the two.
    """
    path = out_dir / "qq_manifest.json"
    if not path.exists():
        return
    old = json.loads(path.read_text()).get("config", {})
    diff = {k: (old.get(k), v) for k, v in cfg.items() if old.get(k) != v}
    if not diff:
        return
    lines = [f"  {k}: on disk {a!r}, requested {b!r}" for k, (a, b) in diff.items()]
    msg = ("Existing artefacts were built with a different configuration:\n"
           + "\n".join(lines))
    if force:
        log.warning("%s\nOverwriting because --force was given.", msg)
        return
    raise SystemExit(msg + "\n\nRe-run with --force to overwrite them.")


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname).1s %(message)s", stream=sys.stdout,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = list(dict.fromkeys([RAW_METHOD] + list(args.methods)))
    qs = [int(q) for q in args.percentiles]
    t_start = time.time()

    # ---- inputs ---------------------------------------------------------
    inp = qin.load_aligned_inputs(DATA_DIR)
    bilinear = qin.load_bilinear_predictor(
        CACHE_DIR, inp["cmip_tas"], inp["era5_lats"], inp["era5_lons"],
        inp["shared_dates"],
    )

    pixels = qn.select_pixels(
        inp["land_mask_2d"], inp["era5_lats"], inp["era5_lons"],
        stride=args.pixel_stride, max_pixels=args.max_pixels, seed=args.seed,
    )
    if args.pixel_stride == 1 and not args.max_pixels:
        if len(pixels) != EXPECTED_LAND_PIXELS:
            log.warning(
                "Full run selected %d land pixels, expected %d. The land mask "
                "has changed, so joins against static_pixels.parquet may fail.",
                len(pixels), EXPECTED_LAND_PIXELS,
            )
    cfg = config_of(args, len(pixels))
    check_manifest(out_dir, cfg, args.force)
    pixels.to_parquet(out_dir / "qq_pixels.parquet", index=False)

    # ---- the percentile table for this scheme ---------------------------
    pct_path = qm_pct_path()
    if not pct_path.exists():
        log.info("Building %s …", pct_path.name)
        static = None
        static_path = QM_DIR / "static_pixels.parquet"
        if static_path.exists():
            static = pd.read_parquet(static_path)
        else:
            log.warning(
                "%s not found; the percentile table will carry no elevation or "
                "sea-fraction columns. Run scripts/run_quantile_mapping.py to "
                "create it.", static_path.name,
            )
        qw.build_percentile_table(
            inp["era5_temp"], bilinear, inp["land_mask_2d"],
            list(inp["shared_dates"]), qw.SEASON_SCHEME, qs, static_cols=static,
        ).to_parquet(pct_path, index=False)

    # ---- per-season daily matrices, shared by every method --------------
    daily = qn.build_daily_seasons(
        inp["era5_temp"], bilinear, list(inp["shared_dates"]), pixels,
        seasons=args.seasons,
    )
    nodes_by_fold = qcv.build_fold_nodes(daily, n_nodes=args.n_nodes)

    # ---- fit ------------------------------------------------------------
    timings: dict[str, float] = {}
    if not args.aggregates_only:
        for method in methods:
            pred_path = out_dir / f"qq_pred_{method}.parquet"
            if pred_path.exists() and not args.force:
                log.info("%s: cached, skipping.", method)
                continue
            t0 = time.time()
            pred, diag, hyper = qcv.run_method(
                method, daily, pixels,
                n_nodes=args.n_nodes, extrap=args.extrap, percentiles=qs,
                sample_pixels=args.sample_pixels, seed=args.seed,
                search_hyper=not args.no_hyper_search,
                nodes_by_fold=nodes_by_fold,
            )
            pred.to_parquet(pred_path, index=False)
            diag.to_parquet(out_dir / f"qq_fit_diag_{method}.parquet", index=False)
            if len(hyper):
                hyper.to_parquet(out_dir / f"qq_hyper_{method}.parquet", index=False)
            timings[method] = time.time() - t0
            log.info("%s: %.1f s", method, timings[method])

    # ---- aggregates (always rebuilt; they are cheap) --------------------
    have = [m for m in methods if (out_dir / f"qq_pred_{m}.parquet").exists()]
    if not have:
        raise SystemExit("No prediction tables found; nothing to aggregate.")

    longs, diags = [], []
    for m in have:
        longs.append(qev.melt_predictions(
            pd.read_parquet(out_dir / f"qq_pred_{m}.parquet"), qs
        ))
        dp = out_dir / f"qq_fit_diag_{m}.parquet"
        if dp.exists():
            diags.append(pd.read_parquet(dp))
    long = pd.concat(longs, ignore_index=True)
    diag = pd.concat(diags, ignore_index=True) if diags else None

    # The identity trap: quantile mapping reproduces the training percentiles
    # exactly, so a zero error would mean the fold split leaked.
    fitted = long[long["method"] != RAW_METHOD]
    if len(fitted):
        worst = fitted.groupby("method", observed=True)["bias"].apply(
            lambda s: s.abs().mean()
        )
        if (worst < 1e-6).any():
            raise SystemExit(
                "A fitted method has essentially zero out-of-sample error:\n"
                f"{worst.to_string()}\n"
                "That means the held-out year reached the training set."
            )

    floor = qev.observed_percentile_sd(long)
    floor.to_parquet(out_dir / "qq_obs_floor.parquet", index=False)

    pixel_err = qev.per_pixel_error(long)
    pixel_err = pixel_err.merge(
        floor[["pix_id", "season", "percentile", "obs_mad", "obs_sd"]],
        on=["pix_id", "season", "percentile"], how="left",
    )
    pixel_err.to_parquet(out_dir / "qq_per_pixel_error.parquet", index=False)

    summary = qev.method_summary(long, floor, diag)
    summary.to_parquet(out_dir / "qq_metrics_summary.parquet", index=False)

    qev.daily_moments(daily, pixels).to_parquet(
        out_dir / "qq_daily_moments.parquet", index=False
    )

    # ---- manifest -------------------------------------------------------
    (out_dir / "qq_manifest.json").write_text(json.dumps({
        "config": cfg,
        "methods": have,
        "hyper_grids": {m: TRANSFORMS[m].hp_grid for m in have},
        "seconds_per_method": {k: round(v, 1) for k, v in timings.items()},
        "total_seconds": round(time.time() - t_start, 1),
        "git_sha": _git_sha(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "created_utc": pd.Timestamp.utcnow().isoformat(),
    }, indent=2))

    # ---- report ---------------------------------------------------------
    print(f"\nWrote {len(have)} method(s) to {out_dir}")
    overall = (summary.groupby("method", observed=True)
               .agg(mae=("mae", "mean"), bias=("bias", "mean"),
                    mae_floor=("mae_floor", "mean"))
               .sort_values("mae"))
    overall["mae_excess"] = overall["mae"] - overall["mae_floor"]
    print("\nMean over seasons and percentiles (°C):")
    print(overall.round(4).to_string())
    print(f"\nTotal {(time.time() - t_start) / 60:.1f} min")


def qm_pct_path() -> Path:
    return QM_DIR / f"pct_bilinear_{qw.SEASON_SCHEME}.parquet"


if __name__ == "__main__":
    main()
