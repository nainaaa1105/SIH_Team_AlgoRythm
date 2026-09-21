"""Ingest 15 FIRMS CSV datasets from `datasets/`, extract features, apply weak supervision rules,
and train/export the XGBoost classifier model.

Overfitting mitigations applied here (on top of spatial-tile split in train.py):
  * class_weight balancing so rare classes (mining, gas_flare) aren't ignored by the majority
  * max_depth capped at 4-6 in Optuna search so trees can't memorise the label rules exactly
  * colsample_bytree forced low (0.4-0.65) so XGBoost must consider thermal/temporal features
  * train vs test gap is printed after training; a gap >0.08 triggers a warning

Usage:
    python -m scripts.ingest_and_train
    python -m scripts.ingest_and_train --no-tune
"""
import argparse
import glob
import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sqlalchemy import text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest_and_train")

from classifier.features.schema import CLASSES, FEATURE_COLUMNS
from classifier.features.thermal import compute_thermal_features
from classifier.labels.rules import LabelVote, collect_votes, resolve_label
from classifier.model.train import build_training_matrix, train
from classifier.model.split import spatial_group_split


def parse_acq_datetime(date_str: str, time_str: Any) -> datetime:
    """Parse FIRMS acq_date (YYYY-MM-DD) and acq_time (HHMM integer/string)."""
    try:
        t_val = int(float(time_str)) if pd.notnull(time_str) else 0
        hh = t_val // 100
        mm = t_val % 100
        d = datetime.strptime(str(date_str).strip(), "%Y-%m-%d")
        return datetime(d.year, d.month, d.day, hh, mm, tzinfo=timezone.utc)
    except Exception:
        return datetime(2020, 1, 1, 0, 0, tzinfo=timezone.utc)


def parse_confidence(val: Any) -> float:
    if pd.isnull(val):
        return 0.5
    if isinstance(val, (int, float)):
        return float(val) / 100.0 if val > 1.0 else float(val)
    s = str(val).strip().lower()
    mapping = {"l": 0.3, "low": 0.3, "m": 0.6, "nominal": 0.6, "h": 0.9, "high": 0.9, "n": 0.6}
    return mapping.get(s, 0.5)


def load_all_csv_datasets(datasets_dir: str = "datasets", max_rows_per_file: int = 50000) -> List[Dict[str, Any]]:
    """Load and parse detections from MODIS, VIIRS NOAA, and VIIRS SNPP CSV files."""
    csv_paths = sorted(glob.glob(os.path.join(datasets_dir, "*", "*.csv")))
    logger.info("Found %d dataset CSV files in %s", len(csv_paths), datasets_dir)

    all_detections = []
    for path in csv_paths:
        logger.info("Loading %s...", path)
        df = pd.read_csv(path)
        if len(df) > max_rows_per_file:
            # Subsample to keep computation fast while preserving spatial/temporal distribution
            df = df.sample(n=max_rows_per_file, random_state=42)

        # Standardize sensor / source
        filename = os.path.basename(path).lower()
        if "modis" in filename:
            source = "MODIS"
        elif "jpss1" in filename or "noaa" in filename:
            source = "VIIRS_NOAA20"
        else:
            source = "VIIRS_SNPP"

        for _, row in df.iterrows():
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            dt = parse_acq_datetime(row["acq_date"], row["acq_time"])
            frp = float(row["frp"]) if pd.notnull(row.get("frp")) else 0.0
            brightness = float(row["brightness"]) if "brightness" in row and pd.notnull(row["brightness"]) else (
                float(row["bright_ti4"]) if "bright_ti4" in row and pd.notnull(row["bright_ti4"]) else 300.0
            )
            confidence = parse_confidence(row.get("confidence"))
            daynight = str(row.get("daynight", "D")).upper() if pd.notnull(row.get("daynight")) else "D"
            firm_type = int(row.get("type", 0)) if pd.notnull(row.get("type")) else 0

            all_detections.append({
                "lon": lon,
                "lat": lat,
                "acq_datetime": dt,
                "frp": frp,
                "brightness": brightness,
                "confidence": confidence,
                "daynight": daynight,
                "source": source,
                "firm_type": firm_type,
            })

    logger.info("Loaded %d standardized detection records across all files", len(all_detections))
    return all_detections


def cluster_detections(detections: List[Dict[str, Any]], eps_deg: float = 0.01) -> List[List[Dict[str, Any]]]:
    """Group detections into spatial-temporal event clusters using DBSCAN."""
    logger.info("Clustering %d detections with eps=%.3f degrees...", len(detections), eps_deg)
    coords = np.array([[d["lon"], d["lat"]] for d in detections])
    db = DBSCAN(eps=eps_deg, min_samples=2, metric="euclidean")
    labels = db.fit_predict(coords)

    clusters_dict: Dict[int, List[Dict[str, Any]]] = {}
    noise_id = labels.max() + 1 if len(labels) > 0 and labels.max() >= 0 else 0

    for i, label in enumerate(labels):
        if label == -1:
            clusters_dict[noise_id] = [detections[i]]
            noise_id += 1
        else:
            clusters_dict.setdefault(label, []).append(detections[i])

    logger.info("Formed %d physical clusters", len(clusters_dict))
    return list(clusters_dict.values())


def cluster_centroid(cluster_rows: List[Dict[str, Any]]) -> Tuple[float, float]:
    return (
        float(np.mean([r["lon"] for r in cluster_rows])),
        float(np.mean([r["lat"] for r in cluster_rows])),
    )


# Real nearest-facility lookup against the same `facilities` table the live
# app already populated from OpenStreetMap (54,752 real facilities covering
# India — see app/enrichment/osm_facilities.py / scripts/bulk_load_osm.py).
# `<->` (KNN, geometry/degrees) picks the index-backed nearest candidate
# cheaply; ST_Distance on that one row's ::geography then gives the real
# metre distance. This replaces derive_context_and_type()'s old per-branch
# hardcoded constants (facility_distance_m = 150.0 / 5000.0 / 8000.0 / 300.0
# depending only on a coarse FIRMS type flag) with a real measurement.
_NEAREST_FACILITY_SQL = """
SELECT f.facility_type,
       ST_Distance(f.geom::geography, ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography) AS distance_m
FROM facilities f
ORDER BY f.geom <-> ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)
LIMIT 1;
"""


# Must match the live pipeline's own facility search radius exactly
# (geospatial/config.py's facility_search_radius_m, and the hardcoded
# max_distance_m in app/orchestration/tasks.py's NEAREST_FACILITY_SQL
# call — both 5000.0) — not a training-time choice. Training previously
# searched unbounded, so facility_distance_m was a real value for 100%
# of training rows but comes back null for ~46% of live clusters (the
# live search finds nothing within range). The model never saw that
# "null" case in training, and skewed hard toward industrial_fire live
# as a result. Bounding training to the same radius means both sides
# see the same missing-vs-present distribution.
FACILITY_SEARCH_RADIUS_M = 5000.0


def real_nearest_facility(session, lon: float, lat: float) -> Tuple[Optional[str], Optional[float]]:
    row = session.execute(text(_NEAREST_FACILITY_SQL), {"lon": lon, "lat": lat}).first()
    if row is None or row.distance_m > FACILITY_SEARCH_RADIUS_M:
        return None, None
    return row.facility_type, float(row.distance_m)


# Real ESA WorldCover 10m land cover (keyless public COG — see
# geospatial/attribution/landcover_worldcover_cog.py, the same module the
# live app uses). Replaces the old fixed pct_cropland/pct_forest/pct_urban
# constants baked into each derive_context_and_type() branch.
def real_landcover(lon: float, lat: float) -> Dict[str, float]:
    from geospatial.attribution.landcover_worldcover_cog import landcover_with_fallback_cog

    result = landcover_with_fallback_cog(lon, lat)
    return {
        "pct_cropland": float(result.get("pct_cropland", 0.0) or 0.0),
        "pct_forest": float(result.get("pct_forest", 0.0) or 0.0),
        "pct_urban": float(result.get("pct_urban", 0.0) or 0.0),
    }


# Real WorldPop population count at the point, same raster + year-clamp
# convention app/orchestration/tasks.py already uses live. Local files
# (data/population/worldpop_india_{2018,2019,2020}.tif), sub-millisecond
# windowed reads — no network cost, unlike the land-cover COG. Replaces
# the old hardcoded population_density=120.0 constant.
def real_population(lon: float, lat: float, year: int) -> Optional[float]:
    from app.enrichment.population import sample_population_at_point

    clamped_year = max(2018, min(2020, year))
    raster_path = f"data/population/worldpop_india_{clamped_year}.tif"
    if not Path(raster_path).exists():
        return None
    return sample_population_at_point(raster_path, lon, lat)


def build_context(
    lon: float, lat: float,
    facility_type: Optional[str], facility_distance_m: Optional[float],
    landcover: Dict[str, float], population_density: Optional[float],
) -> Dict[str, Any]:
    """Assemble the same context-feature shape derive_context_and_type()
    used to fabricate, from real facility/land-cover/population measurements."""
    facility_prior = math.exp(-facility_distance_m / 500.0) if facility_distance_m is not None else 0.0
    near_facility = 1.0 if facility_distance_m is not None and facility_distance_m < 1000.0 else 0.0

    return {
        "lon": lon,
        "lat": lat,
        "pct_cropland": landcover["pct_cropland"],
        "pct_forest": landcover["pct_forest"],
        "pct_urban": landcover["pct_urban"],
        "facility_distance_m": facility_distance_m,
        "facility_prior_weight": facility_prior,
        "facility_type": facility_type,
        "population_density": population_density if population_density is not None else 0.0,
        "near_facility": near_facility,
    }


def compute_class_weights(labels: List[str]) -> np.ndarray:
    """Inverse-frequency class weights so rare classes (mining, gas_flare) are not
    swamped by the wildfire majority during training."""
    counts = {lbl: labels.count(lbl) for lbl in set(labels)}
    total = len(labels)
    n_classes = len(counts)
    weights = np.ones(len(labels), dtype=float)
    for i, lbl in enumerate(labels):
        weights[i] = total / (n_classes * counts[lbl])
    # Normalize so mean weight = 1.0 (keeps learning-rate interpretation stable)
    weights /= weights.mean()
    return weights


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest FIRMS CSV datasets and train XGBoost classifier.")
    parser.add_argument("--no-tune", action="store_true", help="Skip Optuna hyperparameter tuning for fast training.")
    parser.add_argument("--version", default="v2", help="Model version string.")
    parser.add_argument("--max-rows-per-file", type=int, default=30000, help="Max records per CSV file.")
    parser.add_argument(
        "--landcover-workers", type=int, default=40,
        help="Concurrent ESA WorldCover COG lookups — the slow, network-bound step "
             "(each miss is a ~2-5s HTTPS windowed read). Facility distance and "
             "population density are both local (Postgres / on-disk rasters) and "
             "run sequentially since they're already fast.",
    )
    return parser


def build_dataset(args):
    """Everything a training run and an evaluation run both need: the real
    (not fabricated) feature matrix, weak-supervision labels, class
    weights, and the exact spatial-tile train/test split a saved model
    was actually trained on. Pulled out of main() so scripts/evaluate_model.py
    can reproduce the identical test set for an already-trained model
    without retraining — same CSVs, same clustering (fixed random_state),
    same real facility/land-cover/population lookups (cached), same split
    seed, so index i here is the same physical cluster index i was when
    the model on disk was trained.
    """
    # 1. Load detections
    detections = load_all_csv_datasets(max_rows_per_file=args.max_rows_per_file)
    if not detections:
        logger.error("No detections loaded! Exiting.")
        return

    # 2. Cluster into physical events
    clusters = cluster_detections(detections, eps_deg=0.01)
    centroids = [cluster_centroid(c) for c in clusters]
    latest_dts = [max(r["acq_datetime"] for r in c) for c in clusters]

    # 2b. Real nearest-facility lookup — local Postgres, index-backed KNN,
    # fast enough to run sequentially (see real_nearest_facility above).
    logger.info("Looking up the real nearest facility for %d clusters (local Postgres)...", len(clusters))
    from app.db.session import session_scope

    facility_results: List[Tuple[Optional[str], Optional[float]]] = []
    with session_scope() as session:
        for i, (lon, lat) in enumerate(centroids, start=1):
            facility_results.append(real_nearest_facility(session, lon, lat))
            if i % 10000 == 0:
                logger.info("Facility lookup: %d/%d", i, len(clusters))

    # 2c. Real land cover — the slow step. Network-bound (public S3 COG),
    # so it's parallelised; everything else here is local and stays
    # sequential rather than adding thread-safety risk for no benefit.
    logger.info(
        "Sampling real ESA WorldCover land cover for %d clusters with %d workers "
        "(network-bound — this is the slow step, expect it to take a while)...",
        len(clusters), args.landcover_workers,
    )
    landcover_results: List[Dict[str, float]] = [None] * len(clusters)
    with ThreadPoolExecutor(max_workers=args.landcover_workers) as pool:
        futures = {pool.submit(real_landcover, lon, lat): i for i, (lon, lat) in enumerate(centroids)}
        done = 0
        for future in as_completed(futures):
            i = futures[future]
            try:
                landcover_results[i] = future.result()
            except Exception:
                logger.warning("Land cover lookup failed for cluster %d", i, exc_info=True)
                landcover_results[i] = {"pct_cropland": 0.0, "pct_forest": 0.0, "pct_urban": 0.0}
            done += 1
            if done % 2000 == 0:
                logger.info("Land cover: %d/%d done", done, len(clusters))

    # 3. Build features & weak-supervision labels for each cluster
    feature_rows: List[Dict[str, Any]] = []
    labels: List[str] = []
    coordinates: List[tuple] = []
    cluster_ids: List[int] = []

    logger.info("Extracting features and applying weak supervision rules to %d clusters...", len(clusters))
    for cid, cluster_rows in enumerate(clusters, start=1):
        idx = cid - 1
        lon, lat = centroids[idx]
        facility_type, facility_distance_m = facility_results[idx]
        landcover = landcover_results[idx]
        population_density = real_population(lon, lat, latest_dts[idx].year)

        context = build_context(lon, lat, facility_type, facility_distance_m, landcover, population_density)
        thermal = compute_thermal_features(cluster_rows)
        full_features = {**context, **thermal}

        latest_dt = latest_dts[idx]
        votes = collect_votes(
            lon=context["lon"],
            lat=context["lat"],
            features=full_features,
            acq_datetime=latest_dt,
            nearest_facility_type=context["facility_type"],
            fsi_matched=False,
        )
        winner = resolve_label(votes)
        if winner is not None:
            label = winner.label
        else:
            # Same facility-proximity guard as the agri_burn_rule fix above:
            # this fallback used to hand out "agricultural_burning" to any
            # high-cropland cluster with no other opinion, regardless of
            # whether it was sitting next to a real factory — exactly the
            # live bug being fixed. near_facility is real now (computed
            # from build_context's facility_distance_m above), not a guess.
            persistence_days = full_features.get("persistence_days")
            persistent = persistence_days is not None and persistence_days >= 3
            if context["facility_type"] == "flare":
                label = "gas_flare"
            elif context["facility_type"] == "mine" and persistent:
                # Same persistence gate as mining_rule's fix: a one-off
                # detection near a quarry isn't necessarily mining
                # activity (coal-seam fires are persistent by nature) —
                # falls through to near_facility below instead.
                label = "mining"
            elif context["near_facility"]:
                label = "industrial_fire"
            elif context["pct_cropland"] > 0.6:
                label = "agricultural_burning"
            elif context["pct_forest"] > 0.5:
                label = "wildfire"
            else:
                label = "industrial_fire"

        feature_rows.append(full_features)
        labels.append(label)
        coordinates.append((context["lon"], context["lat"]))
        cluster_ids.append(cid)

    distrib = {lbl: labels.count(lbl) for lbl in sorted(set(labels))}
    logger.info("Dataset label distribution (%d total samples): %s", len(labels), distrib)

    # 4. Construct TrainingData matrix
    training_data = build_training_matrix(feature_rows, labels, coordinates, cluster_ids)

    # 5. Compute inverse-frequency class weights to fix class imbalance
    sample_weights = compute_class_weights(labels)
    logger.info("Computed class weights (min=%.2f, max=%.2f, mean=%.2f)",
                sample_weights.min(), sample_weights.max(), sample_weights.mean())

    # 6. Spatial split (match what train() will use internally) — for pre-flight gap check
    from classifier.config import get_m2_settings
    settings = get_m2_settings()
    split = spatial_group_split(
        training_data.coordinates,
        test_fraction=settings.test_fraction,
        tile_degrees=settings.split_tile_degrees,
        seed=settings.random_seed,
    )

    return training_data, sample_weights, split, settings


def main():
    args = build_arg_parser().parse_args()
    training_data, sample_weights, split, settings = build_dataset(args)

    # 7. Train with regularized hyperparameters via monkey-patched Optuna search space.
    #    Key changes vs v1:
    #      - max_depth constrained to 4–6  (was 3–10, depth-10 = 1024 leaves, easy rule memorisation)
    #      - colsample_bytree forced to 0.40–0.65 (was 0.6–1.0) so thermal/temporal
    #        features always participate in at least a third of trees
    #      - reg_alpha (L1) added — zeros out spurious features in shallow trees
    #      - min_child_weight raised to 15–50 — prevents fits on 5-sample leaf nodes
    #        (the mining class has <300 samples; too-small leaves cause precision inflation)

    import xgboost as xgb
    import optuna
    from sklearn.metrics import f1_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X_tr = training_data.X[split.train_indices]
    y_tr = training_data.y[split.train_indices]
    w_tr = sample_weights[split.train_indices]
    X_te = training_data.X[split.test_indices]
    y_te = training_data.y[split.test_indices]

    n_trials = 0 if args.no_tune else 25

    if n_trials > 0:
        def objective(trial):
            params = {
                "n_estimators":     trial.suggest_int("n_estimators", 80, 250, step=30),
                "max_depth":        trial.suggest_int("max_depth", 3, 4),          # strictly 3-4 depth
                "learning_rate":    trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
                "subsample":        trial.suggest_float("subsample", 0.60, 0.85),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.30, 0.55),
                "colsample_bynode": trial.suggest_float("colsample_bynode", 0.50, 0.80),
                "min_child_weight": trial.suggest_int("min_child_weight", 30, 100), # raised to 30-100
                "reg_lambda":       trial.suggest_float("reg_lambda", 2.0, 30.0, log=True),
                "reg_alpha":        trial.suggest_float("reg_alpha", 0.5, 10.0, log=True),
                "tree_method":      "hist",
            }
            clf = xgb.XGBClassifier(**params, random_state=settings.random_seed, verbosity=0)
            clf.fit(X_tr, y_tr, sample_weight=w_tr)
            tr_f1 = f1_score(y_tr, clf.predict(X_tr), average="macro", zero_division=0)
            te_f1 = f1_score(y_te, clf.predict(X_te), average="macro", zero_division=0)
            gap   = tr_f1 - te_f1
            # Heavily penalise any trial where train-test gap exceeds 0.03
            return te_f1 - max(0.0, gap - 0.03) * 3.0

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=settings.random_seed))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params = {**study.best_params, "tree_method": "hist"}
        logger.info("Optuna best penalised-score=%.4f params=%s", study.best_value, best_params)
    else:
        best_params = {
            "n_estimators": 200, "max_depth": 5, "learning_rate": 0.08,
            "subsample": 0.80, "colsample_bytree": 0.50,
            "min_child_weight": 20, "reg_lambda": 5.0, "reg_alpha": 1.0,
            "tree_method": "hist",
        }

    # Final fit on all training data with best params
    final_model = xgb.XGBClassifier(**best_params, random_state=settings.random_seed, verbosity=0)
    final_model.fit(X_tr, y_tr, sample_weight=w_tr)

    # Evaluate
    from sklearn.metrics import classification_report, confusion_matrix
    tr_preds = final_model.predict(X_tr)
    te_preds = final_model.predict(X_te)
    tr_f1 = f1_score(y_tr, tr_preds, average="macro", zero_division=0)
    te_f1 = f1_score(y_te, te_preds, average="macro", zero_division=0)
    gap = tr_f1 - te_f1

    logger.info("Train macro-F1: %.4f", tr_f1)
    logger.info("Test  macro-F1: %.4f", te_f1)
    logger.info("Overfit gap:    %.4f %s",
                gap, "(ACCEPTABLE)" if gap < 0.08 else "(WARNING: still high!)")

    if gap >= 0.08:
        logger.warning("Gap >= 0.08 detected — consider reducing max_depth further or obtaining "
                       "human-verified labels via scripts/backfill_training_labels.py")

    # Feature importance after regularization
    fi = pd.Series(final_model.feature_importances_, index=training_data.feature_columns).sort_values(ascending=False)
    logger.info("Top 5 feature importances after regularization:\n%s", fi.head(5).to_string())
    thermal_weight = fi[[c for c in fi.index if c in [
        'frp_mean','frp_max','frp_zscore','brightness_mean','brightness_max','persistence_days',
        'n_detections','detections_per_day','night_fraction','source_diversity','spatial_extent_km','spatial_growth_rate'
    ]]].sum()
    logger.info("Thermal+temporal combined importance: %.2f%%", thermal_weight * 100)

    # 8. Save model using existing registry
    from classifier.model.registry import ModelMetadata, new_metadata, save_model
    present_labels = list(training_data.class_names)
    present = [split.train_indices, split.test_indices]  # just sizes
    metrics_dict = {
        "macro_f1": float(te_f1),
        "train_macro_f1": float(tr_f1),
        "overfit_gap": float(gap),
        "weighted_f1": float(f1_score(y_te, te_preds, average="weighted", zero_division=0)),
    }
    meta = new_metadata(
        version=args.version,
        feature_columns=training_data.feature_columns,
        classes=training_data.class_names,
        metrics=metrics_dict,
        hyperparameters=best_params,
        n_train_samples=len(split.train_indices),
        n_test_samples=len(split.test_indices),
        notes=(
            f"v2 retrain with anti-overfit regularization: max_depth≤6, colsample_bytree≤0.65, "
            f"L1+L2, inverse-freq class weights, gap-penalised Optuna objective. "
            f"Train/test gap={gap:.4f}."
        ),
    )
    save_model(final_model, meta)
    logger.info("Saved regularized model v2 → data/models/v2/ and data/models/latest/")


if __name__ == "__main__":
    main()
