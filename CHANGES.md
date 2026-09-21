# SIH162 FireSight — Implementation Changelog

One unified project now — `app/`, `classifier/`, `geospatial/`, `imagery/`, `temporal/`, `gateway/`, and `static/` are top-level packages/dirs under this one root (see [README.md](README.md) for the map). This file merges what were originally six separate per-member changelogs, written as each part was built; a later section, **"Unification and post-hoc fixes"**, covers what changed when everything was merged into one project and when GEE/MOSDAC alternatives were added.

## Contents
1. [Member 1 — Data & Ingestion](#member-1-data--ingestion-engineer--implementation-changelog)
2. [Member 2 — ML / Classifier](#member-2-ml-engineer--implementation-changelog)
3. [Member 3 — GIS / Geospatial](#member-3-gis--geospatial-engineer--implementation-changelog)
4. [Member 4 — CV / Imagery](#member-4-cv--imagery-engineer--implementation-changelog)
5. [Member 5 — Novelty / Temporal](#member-5-novelty-modules--implementation-changelog)
6. [Member 6 — Frontend & Dashboard](#member-6-frontend--dashboard--implementation-changelog)
7. [Unification and post-hoc fixes](#unification-and-post-hoc-fixes)

---

## Member 1 (Data & Ingestion Engineer) — Implementation Changelog

Location: `app/`, `scripts/` (package `app`) — merged into the single project root; this file describes the work as originally done.
Nothing in this changelog was deployed anywhere — everything runs
locally only, per instructions. All work was implemented against the
plan agreed in this conversation (10-day plan → [[member1-data-ingestion-spec]]
/ `member1_data_ingestion_spec.md` in project memory) and cross-checked
against `SIH26162_team_task_division.md` and the architecture diagram
supplied earlier in this conversation.

## What was built

### 1. Database schema (Day 1 scope)
- `app/db/models.py` — SQLAlchemy + GeoAlchemy2 ORM models for the 5
  tables everyone else's work keys off: `hotspots`, `clusters`,
  `facilities`, `fingerprints`, `alerts` — matching the exact shape
  frozen in the team task-division doc.
- `app/db/migrations/versions/0001_initial_schema.py` — Alembic
  migration creating the PostGIS + TimescaleDB extensions, all 5 tables,
  spatial GIST indexes, and converting `hotspots` into a TimescaleDB
  hypertable partitioned on `acq_datetime`.
- Kept satellite provenance (`hotspots.source`) as a first-class column
  through dedup/clustering, per the explicit requirement that M2's
  evidence-weighting engine needs to know which sensor(s) backed each
  cluster.

### 2. Ingestion connectors (Day 2–4 scope)
| File | Source | Status |
|---|---|---|
| `app/ingestion/firms.py` | NASA FIRMS (VIIRS + MODIS) | **Fully implemented and unit-tested.** Real, documented API shape (`/api/area/csv/{key}/{source}/{bbox}/{days}`), retry with backoff, malformed-row skipping, plus `fetch_historical_window` for the 90-day backfill. |
| `app/ingestion/insat3ds.py` | INSAT-3DS via MOSDAC | **Best-effort scaffold, clearly flagged.** MOSDAC has no public documented REST API (session/order-based access). The download URL and NetCDF variable names are my best guess at the real product shape and are called out in the module docstring as needing confirmation once a team member has live MOSDAC access. Fails loudly (raises `Insat3dsClientError`) rather than silently mis-parsing if the variable names don't match. |
| `app/ingestion/sentinel3_frp.py` | Sentinel-3 SLSTR FRP (L2 land product) | **Implemented against the real Copernicus Data Space Ecosystem OData API** (OAuth2 client-credentials, product search, download, NetCDF parse of `FRP_MWIR`/`latitude`/`longitude`). This is the one genuinely confirmed as useful for M1 (see the earlier discussion) — an independent FRP retrieval, not just optical imagery. |
| `app/ingestion/himawari.py` | Himawari-8/9 | **Deliberately returns `[]`.** Raw AHI L1 data (on NOAA's public `noaa-himawari8` S3 bucket, which the module can list) is not itself a fire product — a real active-fire product would come from JAXA's P-Tree system, which needs separate registration. Rather than fabricate detections or silently no-op, the module says so in its docstring and logs, and is wired in as a distinct pipeline stage so swapping in a real feed later needs no changes elsewhere. |
| `app/ingestion/normalize.py` | — | Canonical schema + per-source normalizers (VIIRS categorical confidence → 0–1, MODIS numeric confidence → 0–1). Fully unit-tested. |
| `app/ingestion/dedup.py` | — | Exact + near-duplicate dedup, plus dedup against already-persisted DB rows. Fully unit-tested, including a regression test added after a bug was caught (see **Bugs found and fixed** below). |
| `app/ingestion/cluster.py` | — | DBSCAN clustering (eps=0.005°≈500m per the architecture spec), incremental reconciliation against existing active clusters so a persistent flare accumulates into the *same* `cluster_id` across polling cycles instead of spawning a new one every 30 minutes. Fully unit-tested. |
| `app/ingestion/cloud_gate.py` | Google Earth Engine | Cloud-fraction gate using Sentinel-2 `CLOUDY_PIXEL_PERCENTAGE`, plus the confidence-decay formula (linear decay, thermal signal never fully zeroed by cloud since VIIRS/MODIS/INSAT-3DS IR channels penetrate cloud reasonably well). Fully unit-tested. |

### 3. Enrichment (Day 6 scope)
- `app/enrichment/osm_facilities.py` — Overpass query builder for
  industrial tags (`landuse=industrial`, `power=plant`, refineries,
  quarries/mines, etc.), haversine distance, and the
  `prior_weight = base_weight * exp(-distance_m / 500)` proximity-scoring
  formula referenced in the SIH162 planning notes. Also provides a
  production-grade `ST_DWithin` SQL query (`NEAREST_FACILITY_SQL`) for
  fast facility attribution at the DB level.
- `app/enrichment/landcover.py` — MCD12Q1 IGBP-code classification into
  forest/cropland/urban percentages, sampled from a local GeoTIFF window
  around a cluster.
- `app/enrichment/population.py` — WorldPop raster point-sampling.
- `app/enrichment/flares.py` — GGFR flare-site CSV loader.
- All of the above have pure-logic unit tests where the logic doesn't
  require an actual raster/DB (classification math, distance/scoring
  math, Overpass query construction).

### 4. Orchestration (Day 6–7 scope)
- `app/orchestration/pipeline.py` — the full ingestion cycle: fetch all
  sources → in-memory dedup (incl. against already-persisted rows) →
  DBSCAN cluster + reconcile against existing clusters → persist via
  `INSERT ... ON CONFLICT DO NOTHING` (the DB's unique constraint is the
  authoritative dedup backstop) → cloud-fraction gate per cluster → fan
  out per-cluster Celery jobs.
- `app/orchestration/queue.py` / `tasks.py` — Celery app + the
  per-cluster task graph: M1's own `enrich_cluster` task (facility
  attribution + land cover + population, writing `fingerprints`) runs
  first, then `dispatch_downstream_jobs` fans out named stub tasks
  (`tasks.m2_build_feature_vector`, `tasks.m3_spatial_attribution`,
  `tasks.m4_imagery_features` — only if optically available —,
  `tasks.m5_rhythm_and_kalman`) so M2–M5 can register their real task
  bodies under those names without M1's dispatch code changing.
- `app/orchestration/scheduler.py` — APScheduler entry point, polls
  every `POLL_INTERVAL_MINUTES` (default 30).

### 5. API layer (Day 7–8 scope)
- `app/main.py` + `app/api/routes_{hotspots,clusters,facilities}.py` —
  FastAPI read endpoints (`GET /hotspots` with bbox/time/source filters,
  `GET /clusters`, `GET /clusters/{id}`, `GET /facilities`), CORS
  enabled, plus `POST /ingest/run` for a manual/demo trigger and
  `GET /health`. Verified with FastAPI's `TestClient` — all 6 routes
  register and `/health` returns 200 (see **Verification** below).

### 6. Bulk-load scripts (Day 1 / one-time base data)
- `scripts/bulk_load_osm.py` — Overpass fetch → `facilities` table.
- `scripts/bulk_load_gem.py` — GEM plant-tracker XLSX + GGFR flare CSV →
  `facilities` table (GEM has no public API; point it at manually
  downloaded XLSX files).
- `scripts/bulk_load_landcover.py` — kicks off a GEE batch export of
  MCD12Q1 for the India bbox (Earth Engine exports are async to
  Drive/Cloud Storage, not a direct download, so this starts the export
  and prints where to fetch the result from).
- `scripts/backfill_90day.py` — replays the FIRMS archive in ≤10-day
  windows (FIRMS' archive query cap) through the same
  dedup→cluster→persist path, for M2's training-data generation and
  pipeline load-testing. **Only FIRMS is replayed** — INSAT-3DS/Sentinel-3
  don't have an equally simple historical endpoint wired up yet.

### 7. Docker (authored, not run)
- `Dockerfile.api`, `Dockerfile.worker`, `docker-compose.yml` — Postgres
  (TimescaleDB image with PostGIS baked in), Redis, `api`, `worker`, and
  `scheduler` services. **Not started in this session** — per
  instructions, nothing was deployed or even `docker compose up`'d.

## Bugs found and fixed during self-review

Per the instruction to recheck and correct mistakes, three real bugs
were caught and fixed before finishing:

1. **Dead dedup-against-existing-DB-rows logic.** `run_ingestion_cycle`
   queried existing hotspot keys from the DB but never actually passed
   them into `full_dedup(...)`, and the key shapes wouldn't have matched
   even if it had (`load_existing_hotspot_keys` returned
   `(source, acq_datetime)` tuples; `dedup.full_dedup` expects
   `(source, round(lon,6), round(lat,6), acq_datetime)`). Fixed by
   adding a public `dedup.exact_key_for_db_row(...)` helper, having
   `load_existing_hotspot_keys` decode each row's PostGIS geometry and
   build matching keys, and actually threading `existing_keys` through
   to `full_dedup`. Added a regression test
   (`test_exact_key_for_db_row_matches_shape_of_exact_key_from_canonical_record`)
   so the two key-builders can't silently drift apart again.
2. **Falsy-zero confidence bug.** `persist_records_and_clusters` used
   `record.get("confidence") or 0.5` as the fallback for a missing
   confidence value — but a *legitimate* confidence of `0.0` is falsy in
   Python, so it would have been silently overwritten with `0.5`,
   corrupting real low-confidence detections. Fixed with an explicit
   `is None` check.
3. **Dead/wrong code in `enrich_cluster`.** The task pulled every row of
   `facilities` into Python with a placeholder `"lat": None` field that
   was never used for anything (the variable was computed and
   discarded). Replaced with the actual `ST_DWithin` nearest-facility SQL
   query plus real land-cover/population sampling calls, so the task
   does what its docstring claims instead of silently no-op-ing.

## Verification performed

- **37/37 unit tests pass** (`pytest`, see `tests/`), covering every pure
  ingestion/enrichment/clustering function without needing a live DB or
  satellite credentials: FIRMS row normalization (VIIRS categorical +
  MODIS numeric confidence), exact/near/against-existing dedup (including
  the cross-sensor-must-not-collapse requirement), DBSCAN clustering and
  existing-cluster reconciliation, cloud-fraction gating and confidence
  decay math, haversine distance + facility proximity scoring + Overpass
  query construction, and MCD12Q1 land-cover classification.
- **Every application module import-checked successfully** (`python -c
  "import app...."` across all 23 app modules + 4 scripts) — catches
  import-time errors that unit tests alone wouldn't (e.g. circular
  imports, missing dependencies at the top of a module).
- **FastAPI app smoke-tested with `TestClient`**: `/health` returns
  `200 {"status": "ok"}`; all 6 declared routes
  (`/hotspots`, `/clusters`, `/clusters/{cluster_id}`, `/facilities`,
  `/health`, `/ingest/run`) are present in the OpenAPI schema.
- **Alembic migration structurally validated** (`alembic history` shows
  the single `0001` head correctly) without needing a live Postgres
  instance.
- **Everything syntax-compiles** (`python -m py_compile` across every
  `.py` file in the project).

Not verified (would require live external accounts/credentials or an
actual deployment, neither of which this session had or was asked to
set up): a live FIRMS/MOSDAC/CDSE API round-trip, a real Postgres+PostGIS
database, Celery/Redis task execution, and Docker Compose actually
starting.

## Known limitations / what to confirm before this is production-real

- **MOSDAC/INSAT-3DS**: needs a registered account; the download URL
  pattern and NetCDF variable names in `insat3ds.py` are best-effort and
  must be confirmed against a real downloaded product.
- **Himawari**: currently a no-op by design — needs either JAXA P-Tree
  registration for a real derived fire product, or should be dropped
  from the source list if the team decides NE-India cross-validation
  isn't worth that integration cost.
- **psycopg2-binary** has no prebuilt wheel for Python 3.13 on Windows
  yet (this dev machine's default Python) — it tries to build from
  source and fails without a local `pg_config`. Either run everything
  inside the Docker containers (which use `python:3.11-slim`, where the
  pinned version installs cleanly), or use Python 3.11/3.12 locally, or
  bump to a newer `psycopg2-binary` release once one ships a 3.13 wheel.
- **WorldPop raster download isn't scripted yet** — `bulk_load_gem.py`
  covers GEM+GGFR; a `bulk_load_population.py` following the same
  pattern as `bulk_load_landcover.py` still needs to be written once the
  team picks a specific WorldPop product/year.
- **`enrich_cluster`'s land-cover/population sampling silently skips**
  if the local raster files (`data/landcover/...`, `data/population/...`)
  don't exist yet — intentional (so the pipeline doesn't crash before
  Day 1's bulk loads are run), but means `fingerprints` will have NULL
  land-cover/population fields until those rasters are actually fetched.
- **Cloud-fraction gate calls GEE once per touched cluster per poll
  cycle** — fine for a hackathon-scale demo, but a persistent cluster
  seen every 30 minutes for weeks will re-query GEE every time; worth
  caching/only-refreshing-when-stale if GEE quota becomes a concern.
- Downstream task names (`tasks.m2_build_feature_vector`,
  `tasks.m3_spatial_attribution`, `tasks.m4_imagery_features`,
  `tasks.m5_rhythm_and_kalman`) are **contracts, not implementations** —
  M2–M5 need to register Celery tasks under exactly these names for the
  fan-out to actually do anything.

## How to actually run this (when ready — not done in this session)

```bash
cd member1-ingestion
python -m venv .venv && source .venv/Scripts/activate
pip install -r requirements.txt
cp .env.example .env   # fill in FIRMS_MAP_KEY at minimum
docker compose up --build
docker compose exec api alembic upgrade head
python -m scripts.bulk_load_osm
# python -m scripts.bulk_load_gem --gem <path> --ggfr <path>   # once files are downloaded
# then either let the `scheduler` container poll every 30 min, or:
curl -X POST http://localhost:8000/ingest/run
```

---

## Member 2 (ML Engineer) — Implementation Changelog

Location: `classifier/` (package `classifier`) — merged into the single project root; this file describes the work as originally done.
Nothing was deployed — no containers started, no database created, no
cloud resources touched, per instructions.

Built against the plan agreed in this conversation and cross-checked
against `SIH26162_team_task_division.md`, the architecture diagram, and
the research notes (particularly the spatial-leakage paper, which drove
several design decisions below).

---

## How this connects to Member 1

M2 does not duplicate M1's schema or config. It imports them:

```
member1-ingestion/app/     <- M1's package (DB models, config, session, Celery app)
member2-classifier/classifier/   <- M2's package, imports `app.*`
```

The package is named `classifier`, **not** `app`, because two packages
named `app` on one `sys.path` shadow each other (this caused a real bug
during the build — see the bugs section).

The runtime chain, using the task name M1 already dispatches:

```
M1 creates/updates a cluster
  -> tasks.enrich_cluster            (M1: land cover, facility, population)
  -> tasks.m2_build_feature_vector   (M2: assemble cluster_features)   <- already dispatched by M1
       -> tasks.m2_classify          (M2: predict + SHAP + persist)
```

`tests/test_integration_with_m1.py` asserts this contract mechanically:
that M1's dispatch string and M2's registered task name match, and that
both halves share one Celery app object rather than two pointed at
different brokers.

### Changes made to M1's repo (3 files, all additive)

1. `app/db/migrations/versions/0002_add_classification_tables.py` — new
   migration. It goes in M1's chain deliberately: one database, one
   `alembic_version` table, one history. A separate chain would produce
   conflicting states.
2. `app/db/migrations/env.py` — added a **guarded** `import
   classifier.db.models` so autogenerate sees the whole schema. Wrapped
   in `try/except ImportError` so M1's containers still run without M2
   installed. Verified: M1's Alembic works from M1's own venv, where
   `classifier` is absent.
3. `pyproject.toml` — new, so M2 can `pip install -e ../member1-ingestion`
   and share the models rather than copying them. M1's Docker build is
   unchanged (runtime deps stay in `requirements.txt`).

M1's own test suite still passes unchanged: **37/37**.

---

## What was built

### 1. The 28-feature contract (`classifier/features/schema.py`)
Single source of truth for which features exist, their order, and which
member owns each. Six groups:

| Group | Count | Owner | Features |
|---|---|---|---|
| CONTEXT | 7 | M1 | land-cover %, facility distance/prior weight, population, near_facility |
| THERMAL | 7 | M2 | FRP mean/max/std/z-score, brightness mean/max, confidence |
| TEMPORAL | 5 | M2 | persistence days, detection count/rate, night fraction, source diversity |
| SPATIAL | 2 | M2 | footprint extent, growth rate |
| IMAGERY | 4 | M4 | Dozier temp, NDVI, NDBI, smoke red/blue ratio |
| RHYTHM | 3 | M5 | shift sharpness, weekend suppression, Kalman time-to-critical |

`spatial_extent_km` and `spatial_growth_rate` were added because the
research notes call spatial behaviour a primary discriminator — a
wildfire expands across vegetation while a flare stays a fixed point —
and nothing in M1's output captured that.

**`threat_corridor_present` (M3) is deliberately excluded from the model
matrix** while still being stored in the table. A threat corridor is
computed *downstream of* classification (you only draw one once
something is already classed as a spreading fire), so feeding it back in
would be target leakage. This is asserted in the tests.

### 2. Feature computation (`features/thermal.py`, `features/assemble.py`)
Pure functions over detection rows (fully unit-tested), plus a DB-backed
assembler that reads M1's `hotspots`/`fingerprints`/`facilities`.

The FRP z-score baseline **excludes the current burst from its own
baseline** (`before_datetime`), for two reasons: a spike otherwise
inflates the mean it is being compared against and partly hides itself,
and the research notes require temporal features to use only information
available at prediction time.

`upsert_cluster_features` writes **only M2-owned columns**. M3/M4/M5's
tasks run concurrently on the same row, and a blind full-row write would
erase whichever finished first. A test asserts M2's ownership list
doesn't trespass on theirs.

### 3. Evidence-weighting engine (`features/evidence_weighting.py`)
Handles missing feature groups — cloud-blocked imagery, M5 features that
haven't landed yet.

- Missing groups stay **NaN, never imputed to zero**. XGBoost has native
  missing-value handling, so NaN genuinely means "unknown"; imputing 0.0
  would assert "NDVI is exactly zero here", which shifts the prediction.
- Confidence is discounted proportionally to the missing evidence weight,
  floored at 0.5 (the thermal detection itself is always real — we just
  know less about it).
- The reason is recorded in human-readable form, so the XAI panel can say
  *"No optical corroboration: cloud fraction 82% exceeded the usable
  threshold"* rather than silently showing a lower number. The cloud
  fraction comes straight off M1's existing cloud gate.

### 4. Leakage-safe splitting (`model/split.py`)
The most safety-critical module, driven directly by the research notes'
finding that random splits of FIRMS detections inflate scores because the
same physical fire lands in both train and test.

Clustering already fixes the crudest version, but isn't enough: two
clusters 800m apart in one refinery are effectively the same subject. So
the split groups by **spatial tile (~55km)** and keeps whole tiles on one
side. `verify_no_leakage` runs before a single tree is fitted and aborts
training if it fails. A `temporal_split` is also provided for
point-in-time validation.

`tests/test_split.py` includes the contrast test that justifies the whole
module: it demonstrates that a naive random split **does** leak (raises
`LeakageError`) while ours does not.

### 5. Weak-supervision labels (`labels/`)
Five rules — GGFR flare sites, coalfield regions, Punjab/Haryana stubble
season, FSI forest-fire alerts, and persistent/anomalous industrial heat
— each returning a confidence, arbitrated by `resolve_label`.

Contradictory rules at similar confidence return `None` and the cluster
is marked `is_ambiguous` and **excluded from training** — those are the
analyst-review queue, which is also how the MLOps loop later gets
genuinely-verified labels. `persist_labels` never overwrites a
human-verified label with a rule guess.

The `agri_burn_rule` rejects persistent sources on farmland, because a
permanent hot spot in Punjab cropland is a brick kiln, not stubble
burning — mislabelling those would teach the model that cropland implies
agricultural burning.

### 6. Training, registry, fusion, SHAP (`model/`)
- `train.py` — Optuna tuning optimising **macro-F1, not accuracy** (the
  classes are heavily imbalanced; accuracy would let a model that ignores
  rare classes look excellent).
- `registry.py` — versioned artefacts. Every model persists its
  `feature_columns`, re-validated at inference; feature-order drift is
  the classic silent ML bug and here it raises instead.
- `fusion.py` — 0.70 XGBoost + 0.30 EfficientNet, applied **only** when
  imagery was available; otherwise the tabular model carries full weight
  rather than being blended with a blank-patch verdict. `align_to_classes`
  defends against M4 emitting a different class set/order.
- `shap_explain.py` — TreeExplainer with a fallback to gain importances
  if shap is unavailable (an unexplained classification is still useful),
  plus plain-English reason strings for the event card.

### 7. Persistence and API
- `classifier/db/models.py` — `cluster_features`, `classifications`,
  `training_labels`, declared against **M1's `Base`** so they share one
  MetaData and FKs resolve (verified: all 8 tables on one MetaData).
- `api/routes_classify.py` — `GET /classify/{cluster_id}` and
  `GET /classify`, designed to mount onto M1's existing FastAPI app, with
  pre-sorted SHAP factors and a `had_image_corroboration` flag so the UI
  surfaces evidence quality rather than burying it.

---

## Bugs found and fixed during self-review

Five real defects, all found *after* the first green test run — which is
itself the lesson: the unit tests passed while the production path was
broken.

1. **Every classification would have crashed in production.**
   `tasks.m2_classify` passes the whole `cluster_features` ORM row into
   the classifier — including `facility_type` (a string), `updated_at`
   (a datetime) and `cluster_id` (an int). `apply_evidence_weighting`
   called `float()` on every value, so `float('refinery')` raised
   `ValueError` on the first real cluster. My unit tests never caught it
   because they passed clean synthetic dicts containing only the 28
   features. Fixed: the function now returns *exactly* the declared
   `FEATURE_COLUMNS`, coercing defensively (non-numeric → NaN, bool →
   1.0/0.0). Regression tests added for whole-DB-row input.

2. **Training crashed whenever fewer than all five classes were
   labelled.** Encoding labels by their index into the five-element
   `CLASSES` tuple produced `[0, 2, 4]` for a three-class training set,
   and XGBoost rejects non-contiguous classes outright ("Invalid classes
   inferred from unique values of `y`"). This is the *normal* early-project
   case, not an edge case. Fixed: encode contiguously over the classes
   actually present, in canonical order, and carry that list into the
   metadata so inference still maps columns back to the right names —
   absent classes then come back at 0.0 probability, so the API contract
   never changes shape. Verified working for 5-, 3- and 2-class sets.

3. **A stray `app/` directory shadowed M1's entire package.** My initial
   `mkdir` created `member2-classifier/app/...` before I settled on the
   `classifier` name. Python treated the leftover empty directories as an
   implicit namespace package, so `import app.db.models` resolved to the
   empty M2 directory and failed. The unit tests passed regardless
   because none of them imported `app.*`. Removed the empty directories
   (verified zero files first) and added `tests/test_integration_with_m1.py`
   so any recurrence fails loudly.

4. **`NameError` in the Optuna tuning path.** While removing a
   now-unused `n_classes` parameter I left a reference behind in
   `tune_hyperparameters`. Every existing training test used
   `tune=False`, so the entire tuning branch was untested and this would
   only have surfaced on the first real training run. Fixed, and added a
   test that actually exercises tuning.

5. **SHAP explainer cache could return the wrong explainer.** It was
   keyed on `id(model)`, and CPython recycles ids after garbage
   collection — so a new model allocated at a recycled address would get
   a previous model's explainer, producing wrong explanations with no
   error. Switched to a `WeakKeyDictionary`, which drops entries when the
   model is collected.

Also fixed, lower severity: FSI alert dates that parse to `NaT` silently
never matched (NaT comparisons are always False) — now treated as a
missing date; a cross-module import of a private `_group_is_present`;
a misleading comment in `tasks.py` promising DB-read behaviour that
doesn't exist; and a redundant explicit `num_class` that can conflict
with what XGBoost infers.

### One design correction (not a crash)
The first test run flagged a cluster at a GGFR flare site *inside* the
Jharia coalfield as ambiguous — `gas_flare` (0.90) vs `mining` (0.85),
within the 0.10 ambiguity margin. The code was behaving as designed; the
**confidence calibration** was wrong. A located match against a
catalogued facility 100m away is far stronger evidence than "somewhere
inside a 50km box", and coalfields routinely contain pithead power
plants and flare stacks. Lowered the regional mining prior to 0.70 so
specific evidence outranks regional priors. Without this, every
industrial site inside a coalfield would have been dropped from training
as "ambiguous". Four tests now pin this interaction, including one
asserting that a genuinely unclear case (spreading forest fire inside a
coalfield) *does* still go to review.

---

## Verification performed

- **133/133 M2 tests pass.** Coverage: the feature contract, thermal /
  spatial feature maths, evidence weighting (incl. whole-DB-row input),
  fusion and class alignment, all five label rules and their arbitration,
  FSI matching, the leakage guard, and a full **train → save → reload →
  predict** cycle against real XGBoost on synthetic data.
- **M1's 37/37 tests still pass** after the three additive changes.
- **All 23 M2 modules + 2 scripts import cleanly**, with `app` correctly
  resolving to M1's package.
- **Shared MetaData verified**: all 8 tables (M1's 5 + M2's 3) on one
  MetaData with working cross-package foreign keys.
- **Migration chain verified**: `0001 -> 0002 (head)`, resolving both
  with and without `classifier` installed.
- **Task-name contract verified mechanically** against M1's source.
- **API router verified** to mount and expose `/classify/{cluster_id}`.
- Everything syntax-compiles (`py_compile` across all files).

**Not verified** (needs live infrastructure this session didn't set up
and wasn't asked to): a real Postgres/PostGIS database, actual Celery
task execution over Redis, training on real FIRMS-derived data, and any
deployment.

---

## Known limitations / what to confirm before this is production-real

- **No model is trained yet.** `data/models/` is empty until
  `scripts/backfill_training_labels.py` then `scripts/train_model.py`
  run against a populated database. Until then `tasks.m2_classify`
  returns `{"status": "no_model"}` rather than crashing — deliberate, so
  the pipeline degrades rather than breaks.
- **All accuracy numbers are unknown.** The macro-F1 in the tests is on
  *synthetic, separable-by-construction* data and means nothing about
  real performance. Do not quote it anywhere.
- **The labels are weak supervision, not ground truth.** Every metric
  derived from them inherits their biases. FSI alerts are themselves
  derived from the same VIIRS/MODIS anomalies we ingest, and FSI filters
  out known mining/industrial areas — so an FSI match is strong evidence
  of "forest fire", but the absence of one is weak evidence of "not".
- **Optuna tunes against the same held-out split that gets reported**,
  which makes macro-F1 mildly optimistic. This is recorded in the model
  metadata notes rather than hidden. With a larger labelled set, switch
  to a separate validation fold.
- **Coalfield and agri-belt regions are hardcoded bounding boxes**, which
  is coarse. Real coalfield polygons would cut the ambiguity rate.
- **`facility_type` is not used as a model feature** (it's categorical
  and would need encoding); its signal reaches the model through
  `facility_prior_weight` and `near_facility` instead.
- **M4/M5 columns are NULL until their tasks land.** The system works in
  a degraded, thermal-plus-context-only mode until then, with confidence
  discounted accordingly — by design, so M2 was never blocked on them.
- Downstream contract for M4: call
  `tasks.m2_classify(cluster_id, image_probabilities={class: prob})`.
  Agree this shape with M4 before Day 7.

---

## How to run it (when the DB is up — not done in this session)

```bash
cd member2-classifier
python -m venv .venv && source .venv/Scripts/activate
pip install -r requirements.txt            # includes -e ../member1-ingestion

# schema (from member1-ingestion, applies both 0001 and 0002)
cd ../member1-ingestion && alembic upgrade head && cd ../member2-classifier

# labels, then training
python -m scripts.backfill_training_labels --fsi data/fsi_alerts.csv
python -m scripts.train_model --version v1

# worker with M2's tasks registered
celery -A app.orchestration.queue.celery_app worker -I classifier.tasks

# and in member1-ingestion/app/main.py:
#   from classifier.api.routes_classify import router as classify_router
#   app.include_router(classify_router)
```

---

## Member 3 (GIS / Geospatial Engineer) — Implementation Changelog

Location: `geospatial/` (package `geospatial`) — merged into the single
project root. Nothing was deployed — no containers, no database, no
cloud resources, per instructions.

Built against the plan agreed in this conversation and cross-checked
against `SIH26162_team_task_division.md`, the architecture diagram, and
the research notes.

---

## The overlap problem, and how it was resolved

M1 had **already implemented** two things the original task division
gave M3: nearest-facility attribution (`app/enrichment/osm_facilities.py`)
and MODIS land-cover sampling (`app/enrichment/landcover.py`), both
inside `tasks.enrich_cluster`. Rebuilding them would have been wasted
work, so M3 *upgrades* them instead and M1's simpler versions remain as
the inline fallback when M3's worker isn't running.

## The ordering problem, and how it was resolved

M1 dispatches all four downstream tasks in parallel, which breaks two
things:

1. M2 snapshots `fingerprints` into `cluster_features`. If M3 improved
   the attribution *after* that snapshot, M2 would train on the coarser
   values.
2. A threat corridor is only meaningful once the source is known to be a
   spreading fire — which is exactly why M2 excluded
   `threat_corridor_present` from the model's feature matrix (computing
   it downstream of the prediction means feeding it back would be target
   leakage).

So M3 runs in **two phases**:

```
M1 creates/updates a cluster
  ├─ tasks.enrich_cluster              (M1: coarse fallback enrichment)
  ├─ tasks.m2_build_feature_vector     (M2)
  └─ tasks.m3_spatial_attribution      PHASE A  <- M1 already dispatched this name
        │  probabilistic attribution + zonal land cover -> fingerprints
        └─ re-triggers tasks.m2_build_feature_vector   (idempotent upsert)
              └─ tasks.m2_classify     (M2)
                    └─ tasks.m3_threat_and_plume   PHASE B  <- added to M2's classify task
                         plume cone + threat corridor + exposure
```

### Changes to other members' repos (3 files, all additive)

1. `member1-ingestion/app/db/migrations/versions/0003_add_geospatial_tables.py`
   — new migration in the shared chain (`0001 → 0002 → 0003`).
2. `member1-ingestion/app/db/migrations/env.py` — added a **guarded**
   `import geospatial.db.models` alongside M2's, so autogenerate sees
   the whole schema. Still works from M1's own venv where `geospatial`
   is absent.
3. `member2-classifier/classifier/tasks.py` — the Phase-B dispatch
   (`send_task("tasks.m3_threat_and_plume", ...)`) after classification,
   plus a new `member2-classifier/pyproject.toml` so M3 can
   editable-install M2 and read `cluster_features`.

**M1's 37/37 and M2's 133/133 tests still pass unchanged.**

---

## What was built

### Geodesy foundations (`geometry.py`)
Bearings, destination points, great-circle distance, circular means. Pure
and heavily tested because every polygon M3 produces sits on them.
`downwind_bearing` is a named function with its own tests rather than an
inline `+ 180` specifically because the meteorological convention (wind
direction is where wind comes *from*) is a classic sign-error trap that
would put a hazard zone on precisely the wrong side of a town.
`circular_mean` exists because averaging 350° and 10° arithmetically
gives 180° — exactly backwards.

### Gaussian plume dispersion (`plume/gaussian.py`)
Implemented directly from published formulae: Pasquill-Gifford stability
classification, Briggs (1973) rural dispersion coefficients, Briggs final
plume rise for buoyant sources, with the buoyancy flux derived from FRP
via a 0.17 radiative fraction.

⚠️ The task-division doc names **"pyELDQM"** for this. I could not verify
that such a library exists, and building a safety-relevant calculation on
an unverifiable dependency is a bad trade when the physics is this
well-documented. So it is implemented directly, with the reference for
each constant named in the code.

The drawn footprint is the 10%-of-centreline concentration contour
(`2.146 × σy`), and a test asserts that constant really does correspond
to 10% rather than being a magic number.

**Scope honesty:** this is a steady-state, flat-terrain, single-source
screening model. It ignores terrain, building downwash, plume depletion,
chemistry and wind shear. It is a decision-support cone for a dashboard,
not a regulatory dispersion product, and the module docstring says so.

### Probabilistic facility attribution (`attribution/facility_match.py`)
Upgrades M1's nearest-only match with three things it couldn't express:
containment (`ST_Contains` — a hotspot inside a refinery isn't "0 m from"
it, it *is* it), facility-type priors (a flare stack and a warehouse
equidistant is not a 50/50 call), and a normalised distribution over
candidates so responders see the runners-up. Mirrors M2's ambiguity
handling: a near-tie is surfaced, not hidden behind a single answer.

### Land-cover zonal statistics (`attribution/landcover_zonal.py`)
ESA WorldCover at 10 m instead of MODIS at 500 m, with the sampling
buffer **sized from M2's `spatial_extent_km`** rather than a fixed
window — a 30 km fire front and a single flare stack shouldn't get
identical context. Falls back to M1's MODIS sampler, and records which
source was used. Both class schemes are kept and a test asserts they
disagree on the same code (10 = forest in WorldCover, grassland in IGBP),
guarding against feeding the wrong raster to the wrong summariser.

### Threat trajectory and corridor (`threat/`)
Time-bucketed centroid tracking (so a dense satellite overpass burst
counts as one look, not ten), a confidence score from path straightness
that widens the corridor when the direction is poorly defined, and a
blend of observed movement with downwind direction weighted by that
confidence.

`conservative_spread_rate` takes the **larger** of centroid travel rate
and M2's footprint growth rate — these are genuinely different quantities
(a fire burning outward in all directions has high growth and near-zero
travel), and for a number feeding an evacuation decision the safe error
is to say the fire arrives sooner than it does.

Corridors are gated to spreading classes only: projecting an advance
corridor for a gas flare would be actively misleading on an operations
map. A reclassified cluster has its stale corridor deleted.

### Exposure, chemicals, routing, API, schema
Zonal population sums with an explicit `raster_is_density` flag (mixing
count and density rasters is an order-of-magnitude error waiting to
happen), emission profiles by facility type, opt-in OSMnx routing, three
GeoJSON endpoints, and four new tables in migration `0003`.

---

## Bugs found and fixed in self-review

1. **The accurate PostGIS distance was being thrown away.** The candidate
   query computes `ST_Distance` against each facility's full geometry —
   which for a polygon is distance to its *boundary*, zero at the fence
   line. `attribute()` discarded that and recomputed haversine distance
   to the polygon's **centroid** instead. For a hotspot at the edge of a
   large refinery this reported **1178 m instead of 0 m**, under-scoring
   the correct facility by roughly 10×.

   This was the worst bug found, because it propagates across all three
   members: the wrong distance lands in `fingerprints.facility_distance_m`,
   which M2 snapshots into `cluster_features` as a **model feature**, and
   which M2's label rules threshold at `< 500 m` to assign
   `industrial_fire` and `gas_flare` labels. It would have corrupted
   attribution, features and training labels simultaneously. Fixed to use
   a supplied distance when present, with the haversine path retained for
   offline/pure use. Three regression tests added.

2. **Network calls were being made inside open database transactions.**
   Both tasks held a Postgres transaction across an Earth Engine
   `reduceRegion` call (Phase A) and an Open-Meteo request (Phase B) —
   multi-second network round-trips pinning a connection and its locks
   the whole time. That is how a worker pool starves under load, and the
   team plans a 100-concurrent-connection load test. Both tasks
   restructured into read → network → write, with two short transactions
   and no I/O in between.

3. **Historical wind was silently unavailable.** `fetch_wind` always
   used M1's `open_meteo_url`, which is the *forecast* endpoint and only
   covers roughly the last few days. Every cluster in the 90-day
   retrospective replay would have fallen back to assumed wind while
   still producing a confident-looking plume. Added `choose_endpoint`,
   which switches to Open-Meteo's archive API beyond a configurable
   cutoff, with tests on both sides of the boundary.

Also cleaned up: an unused import in the attribution task, and an empty
leftover `scripts/` directory (the same class of namespace-shadowing
hazard that cost real debugging time in M2 — verified harmless here, but
removed rather than left lying around).

---

## Verification performed

- **219/219 M3 tests pass**, covering geodesy conventions, dispersion
  physics behaviour, trajectory/corridor geometry, attribution scoring,
  wind parsing and endpoint selection, chemical profiles, exposure maths,
  point-in-polygon, and polygon validity.
- **M1's 37/37 and M2's 133/133 still pass** after the three additive
  changes.
- **All 16 M3 modules import cleanly**, with `app` and `classifier`
  resolving to the real M1/M2 packages.
- **Shared MetaData verified**: all 12 tables (M1's 5, M2's 3, M3's 4) on
  one MetaData with cross-package foreign keys resolving, and all
  geometry columns on SRID 4326 (mixing SRIDs makes `ST_Intersects`
  silently return nothing).
- **Migration chain verified**: `0001 → 0002 → 0003 (head)`, resolving
  from M1's own venv where M2/M3 are not installed.
- **Task contracts verified mechanically** against M1's and M2's real
  source: Phase-A name matches M1's dispatch, Phase-B name matches M2's,
  all three share one Celery app object, and no route path collides
  across the three routers.
- **Polygon validity verified** on the production storage path (shapely
  `Polygon(...)` → PostGIS) across the full range of wind directions,
  stability classes, half-angles and suggested lengths — valid, simple,
  non-zero-area in every case, including calm wind and minimum steps.
- **End-to-end smoke test** of the composed pipeline on a synthetic
  advancing wildfire: trajectory direction, wind blending, class gating,
  corridor geometry, rear-facility exclusion and plume physics all behave
  correctly (1.5 km plume rise for an 88 MW fire in 4 m/s wind, which is
  in the realistic range for wildfire smoke).

**Not verified** (needs live infrastructure this session didn't set up
and wasn't asked to): a real Postgres/PostGIS database, Celery execution
over Redis, live Open-Meteo/Earth Engine calls, and any deployment.

---

## Known limitations

- **Plume geometry has never been validated against a real dispersion
  model or observed plume.** The tests check internal consistency and
  physical behaviour (stable air ⇒ narrower, bigger fire ⇒ higher rise),
  not accuracy. Do not present the footprint as a validated hazard
  prediction.
- **The case-study validation (Baghjan, Chembur, Morbi) has not been
  run.** It is the Day-10 deliverable and needs a populated database.
  Until then there is no evidence the attribution picks the right
  facility on real incidents.
- **Population exposure and 10 m land cover are unavailable without
  optional dependencies** (`rasterio`, `earthengine-api`). Both degrade
  cleanly — exposure returns `None` (distinguishable from zero, which
  matters: "we don't know" and "nobody lives there" must not look the
  same to a responder) and land cover falls back to MODIS.
- **`tasks.m3_threat_and_plume` runs twice per cluster.** M1 dispatches
  M2's feature build and M3's Phase A also re-triggers it, so the chain
  to classification fires twice. All writes are upserts so the result is
  correct, but it is wasted work — and under two concurrent workers the
  `session.get(...) → session.add(...)` pattern could race into a
  duplicate-key error. The fix is PostgreSQL `INSERT ... ON CONFLICT DO
  UPDATE` (as M1 already uses for `hotspots`); it needs a real database
  to test, so it is flagged rather than guessed at. The same pattern
  exists in M1's and M2's persistence code.
- **M1's and M2's `scripts` packages collide** when both are on the path
  (M1's wins). Harmless today because each member runs scripts from their
  own directory, but worth knowing before anyone imports across them.
- **Evacuation routing is disabled by default** (`routing_enabled=False`)
  and untested against a real road network — building an OSMnx graph is
  slow and network-bound, which is wrong inside a task that fires every
  30 minutes. Run it per-incident, not in the live pipeline.
- **Chemical profiles are indicative, for responder awareness only** —
  general process chemistry per industry, not measured emission factors
  for a specific plant. Nothing in them is a quantitative estimate.
- **Corridor projection assumes a constant spread rate and wind** over
  the projection window. Real fire behaviour changes with terrain, fuel
  and diurnal wind shifts.

---

## How to run it (when the DB is up — not done in this session)

```bash
cd member3-geospatial
python -m venv .venv && source .venv/Scripts/activate
pip install -r requirements.txt      # pulls in M1 and M2 as editable installs

cd ../member1-ingestion && alembic upgrade head && cd ../member3-geospatial

celery -A app.orchestration.queue.celery_app worker -I classifier.tasks,geospatial.tasks

# and in member1-ingestion/app/main.py:
#   from geospatial.api.routes_geo import router as geo_router
#   app.include_router(geo_router)
```

---

## Member 4 (CV / Imagery Engineer) — Implementation Changelog

Location: `imagery/` (package `imagery`) — merged into the single project root. Nothing was deployed —
no containers, no database, no cloud resources, per instructions.

Built against the plan agreed in this conversation and cross-checked
against `SIH26162_team_task_division.md`, the architecture diagram, and
the research notes.

---

## The gap M4 had to fill

The plan flagged it and the code confirmed it: **nobody had built the
Sentinel-2 fetcher.** It was on M1's Day-5 plan but `app/ingestion/`
contains FIRMS, INSAT-3DS, Sentinel-3 and Himawari only. M2's evidence
engine has been treating every imagery feature as unavailable ever since.
M4 owns it now (`imagery/optical/sentinel2.py`).

A happier discovery: **the second brightness band was already there.**
M1 stores `bright_ti4` in `hotspots.brightness`, but its normaliser keeps
the entire original FIRMS row in `hotspots.raw_payload` — so `bright_ti5`
(and MODIS `bright_t31`) is recoverable without touching M1's busiest
table. Dozier needed no schema change to `hotspots`.

## Two wiring changes, both made and both tested

**1. M1 was silently disabling the Dozier retrieval on cloudy clusters.**
M1 only dispatched M4 when the cloud gate was open:

```python
if optical_available:
    celery_app.send_task("tasks.m4_imagery_features", args=[cluster_id, lon, lat])
```

But M4's work splits across two data sources with different dependencies.
The Sentinel-2 half genuinely needs a clear sky; the Dozier sub-pixel
temperature uses VIIRS I4/I5 brightness temperatures straight from FIRMS
and needs no optical imagery at all. Gating the whole task on cloud cover
threw away sub-pixel fire temperature for exactly the clusters with the
least other evidence — most of India through the monsoon, and
`dozier_temp` is one of M2's 28 features.

M1 now dispatches unconditionally and passes the flag through; M4
branches internally. Its signature defaults `optical_available=True`, so
a stale three-argument dispatch still works.

**2. M3's Phase B was being re-run three times per cluster.** M1 triggers
M2's classify, M3's Phase A re-triggers it, and now M4's fusion callback
triggers it a third time — and each classification dispatched
`tasks.m3_threat_and_plume`, which makes an Open-Meteo request every
time. M2's classify now only re-dispatches M3 when the predicted class
actually changed.

### Files changed outside M4 (4, all additive)

1. `member1-ingestion/app/db/migrations/versions/0004_add_imagery_tables.py` — new migration (chain is now `0001 → 0002 → 0003 → 0004`).
2. `member1-ingestion/app/db/migrations/env.py` — guarded `import imagery.db.models` alongside M2's and M3's.
3. `member1-ingestion/app/orchestration/tasks.py` — the dispatch change above.
4. `member2-classifier/classifier/tasks.py` — the re-dispatch guard, plus a new `member3-geospatial/pyproject.toml` so M4 can editable-install M3.

**M1's 37, M2's 133 and M3's 219 tests all still pass.**

---

## What was built

### Planck radiation (`thermal/planck.py`)
Spectral radiance and its exact inverse, in SI, with CODATA constants.
Kept separate and tested to round-trip to 1e-9 because every temperature
M4 reports depends on it.

### Dozier sub-pixel retrieval (`thermal/dozier.py`)
A 375 m VIIRS pixel is vastly larger than the flaming front inside it, so
the reported brightness temperature is an area-weighted mix. Dozier
inverts two bands for two unknowns — fire temperature and fractional area
— via `scipy.optimize.fsolve`, solved in log-fraction space because the
fraction spans several orders of magnitude and an unconstrained linear
solve wanders negative.

Physical gates before the solver, not after: VIIRS I4 saturation (~367 K,
above which the retrieval is numerically fine and physically meaningless),
no-thermal-excess, and plausibility bounds of 400–1800 K.

### Background estimation (`thermal/background.py`)
An honest workaround for a real data limitation. Textbook Dozier takes the
background from neighbouring non-fire pixels in the granule; **FIRMS
distributes only the fire pixels**. So the background comes from a low
percentile of the cluster's own thermal-band values, falling back to
day/night climatology, and the value *and its provenance* are stored with
every retrieval.

### Sentinel-2 fetch (`optical/sentinel2.py`)
The gap. GEE `S2_SR_HARMONIZED`, six bands, patch sized from M2's
`spatial_extent_km` (the same trick M3 uses for its land-cover buffer),
least-cloudy scene in a ±7-day window, saved as a compressed `.npz` plus
a percentile-stretched RGB thumbnail for M6's event card. Reuses M1's GEE
credentials rather than adding a second auth path. Reflectance is divided
by the 10000 scale factor — omitting that would make every index and
threshold wrong by four orders of magnitude.

### Indices and smoke (`optical/indices.py`, `optical/smoke.py`)
NDVI, NDBI, NBR, red/blue ratio and a four-condition smoke mask. The SWIR
condition is what separates smoke from cloud — both are bright in the
visible, but water/ice cloud stays bright in SWIR while smoke does not.

Then the cross-member payoff: the observed plume bearing measured from
pixels is compared against **M3's modelled `downwind_bearing_deg`**. Two
independent sources agreeing is real corroboration — precisely the "wind
is supporting evidence, not a classifier" point from the research notes.
A missing comparison reports *"not comparable"* rather than
*"disagreement"*.

### EfficientNet-B0 (`model/`)
torchvision ImageNet weights with two adaptations: the stem convolution is
widened from 3 to 6 channels (extra bands seeded from the mean of the
pretrained RGB kernels and the layer rescaled by 3/6, so the pretrained
signal survives and downstream activations stay in range), and a 5-class
head replaces the 1000-class one.

Training reuses **M2's `spatial_group_split` and `verify_no_leakage`**
rather than writing a second splitter — a random patch split is exactly
the spatial-leakage failure the research notes document. Labels come from
M2's `training_labels`, so both models share ground truth and their
probabilities are genuinely fusable. The registry pins `band_order` and
validates it at inference, mirroring M2's feature-order discipline.

A `min_image_confidence` gate withholds weak verdicts from fusion: M2
weights the image at 30%, so a near-uniform guess would still move the
answer. Sending nothing and letting M2 run tabular-only is better than
diluting a confident tabular verdict with noise.

---

## A scientific limitation found in self-review

A test failed with the solver returning **1859 K for a true 950 K fire**.
Investigating rather than adjusting the test showed the cause: the
estimated background was **0.32 K** too high.

Measuring the sensitivity properly:

| Fire | ±1 K background error → retrieved temperature |
|---|---|
| Small/hot — 950 K, p=3e-4 (**a gas flare**) | 648 K … rejected |
| Larger/cooler — 700 K, p=5e-3 | 658 K … 764 K (±10%) |

The Dozier inversion is badly ill-conditioned for small hot fires — and
that is *exactly* the gas-flare case this problem statement cares about.
Combined with a background that is estimated rather than measured,
reporting a bare temperature would imply a precision the method does not
have.

So `solve_with_uncertainty` now re-solves at the edges of the background's
own uncertainty, stores the bracket (`temperature_low_k`,
`temperature_high_k`) and a `well_constrained` flag, and the API surfaces
all three. A poorly-constrained retrieval says so in its `reason` string.
This turned a silent accuracy problem into an explicit, reported one.

## Bugs found and fixed in self-review

1. **A numpy truthiness crash on every complete patch.**
   `summarise_patch` used `if None not in (blue, green, red)`, which
   compares element-wise against arrays and raises *"truth value of an
   array with more than one element is ambiguous"*. Any patch with all
   visible bands present — i.e. every successful fetch — would have
   crashed. Fixed with explicit identity tests. I grepped the other three
   members for the same pattern: M1 has one occurrence but on plain
   floats, which is safe.

2. **An optical failure would have discarded the thermal result.** Dozier
   is computed before the optical half but written after it, so an
   unguarded exception anywhere in imagery (band mismatch, disk error,
   malformed scene) would throw away a perfectly good temperature. The
   optical half is now wrapped; Dozier is the more reliable product and
   must not be lost to an imagery failure.

3. **A dynamically-attached attribute.** `background_source` was being
   `setattr`-ed onto the result rather than declared, so the retrieval and
   the assumption it rests on could drift apart. Now a real dataclass
   field.

Also corrected: two test assertions that were wrong rather than the code —
an arbitrary "10x" threshold where the physics gives 5x (rewritten to
compare fire contribution against background contribution, which is the
meaningful claim), and an end-to-end test that used the ill-conditioned
regime to assert well-conditioned accuracy.

---

## Verification performed

- **145/145 M4 tests pass.** Coverage: Planck round-trip across four
  bands and eight temperatures, Dozier forward-simulation recovery across
  five fire regimes, the ill-conditioning contrast, background estimation,
  spectral indices, smoke masking (including cloud and vegetation
  rejection), smoke bearing conventions, the M3 cross-check, EfficientNet
  stem adaptation against real torch, registry round-trip, and the
  band-order contract.
- **M1's 37, M2's 133 and M3's 219 still pass** after the four additive
  changes. **530 tests across the four members.**
- **All 16 M4 modules import cleanly** with `app`, `classifier` and
  `geospatial` resolving to the real packages.
- **Shared MetaData verified**: 15 tables (M1's 5, M2's 3, M3's 4, M4's 3)
  on one MetaData, foreign keys resolving, patch bbox on SRID 4326.
- **Migration chain verified**: `0001 → 0002 → 0003 → 0004 (head)`, and
  **ORM/migration column parity checked** for all three M4 tables (12/12,
  11/11, 14/14) — now a permanent test.
- **Real torch used throughout the model tests** (torch 2.14.0+cpu,
  torchvision 0.29.0+cpu): the stem adaptation provably preserves the
  pretrained RGB kernels and seeds extra channels from their mean.
- **End-to-end smoke test** of the composed pipeline: INSAT-3DS correctly
  excluded from Dozier as a single-band source, background estimated from
  the cluster percentile, a 764 K / 335 m² retrieval correctly flagged as
  poorly constrained (32% spread), smoke bearing 48° recovered from a
  synthetic north-east plume, and the M3 cross-check agreeing at 50°,
  disagreeing at 220°, and abstaining with no plume.

**Not verified** (needs live infrastructure this session didn't set up and
wasn't asked to): a real Postgres/PostGIS database, Celery over Redis,
live Earth Engine calls, actual Sentinel-2 imagery, and any deployment.

---

## Known limitations

- **No image model is trained.** `data/image_models/` is empty until
  labelled patches exist. `classify_patch` returns `None` and M2 runs
  tabular-only — deliberate degradation, not a failure. Every number in
  the model tests comes from an *untrained* network on random input and
  says nothing about real accuracy.
- **The Dozier retrieval has never been validated against ground truth or
  an independent product.** The forward-simulation tests prove the solver
  inverts the physics correctly; they do not prove the physics matches a
  real fire, and the background is estimated rather than measured. Treat
  `well_constrained=False` as "do not quote this number".
- **The smoke indices are documented heuristics, not validated
  retrievals.** The physical reasoning (blue scattering, SWIR
  transparency) is sound; the specific thresholds are tuned guesses until
  checked against the case studies.
- **Sentinel-2 fetch is untested against live GEE.** The code path is
  exercised only through its failure branch; the `sampleRectangle` size
  limit in particular may need a different extraction approach for large
  patches.
- **Emissivity is assumed to be 1.0** in the Planck inversion. Real
  surfaces sit around 0.96–0.99, which biases retrieved temperatures
  slightly. Standard for a screening-level product, worth stating.
- **Only the hottest detection in a cluster is inverted.** A multi-front
  fire has more than one fire temperature; this reports the dominant one.
- **`tasks.m4_imagery_features` may still run more than once per cycle**
  (M1 dispatches it, and a re-clustering can re-dispatch). Writes are
  upserts so the result is correct, but `sentinel2_patches` accumulates a
  row per fetch by design — that is the time series, not a duplicate bug.
  The concurrent-worker `session.get → session.add` race noted in M3's
  changelog applies here too; the fix is `ON CONFLICT DO UPDATE` and needs
  a real database to test.
- **torch pins in `requirements.txt` (2.4.1/0.19.1) differ from what was
  installed and tested here** (2.14.0+cpu / 0.29.0+cpu, from the CPU index
  URL). Align the pins before anyone builds a container.

---

## How to run it (when the DB is up — not done in this session)

```bash
cd member4-imagery
python -m venv .venv && source .venv/Scripts/activate
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

cd ../member1-ingestion && alembic upgrade head && cd ../member4-imagery

celery -A app.orchestration.queue.celery_app worker \
       -I classifier.tasks,geospatial.tasks,imagery.tasks

# and in member1-ingestion/app/main.py:
#   from imagery.api.routes_imagery import router as imagery_router
#   app.include_router(imagery_router)
```

---

## Member 5 (Novelty Modules) — Implementation Changelog

Location: `temporal/` (package `temporal`) — merged into the single project root. Nothing was deployed —
no containers, no database, no cloud resources, per instructions.

Built against the plan agreed in this conversation and cross-checked
against `SIH26162_team_task_division.md`, the architecture diagram, the
phased deliverables list, and the research notes.

---

## The gap M5 filled

I grepped all four existing members before starting: **PTSI / the
persistent-source catalog did not exist anywhere.** Only a stray
`persistent_source_min_days` setting in M2's config. Yet:

- the architecture puts "Persistent-source catalog (PTSI registry of
  normal heat)" in Layer 4;
- Phase-1 deliverable **#2** is the "Persistent vs. Transient Anomaly
  Segregator";
- the project notes call *normal vs abnormal industrial behaviour* the
  team's differentiator over a basic FIRMS + OSM + map project.

Without it the brief's own example output — *"persistent for 27 days,
currently 4.2× its normal FRP"* — is not expressible, because nothing
knew what "its normal" was. `ptsi/` is that, and an end-to-end run on a
synthetic 45-day flare with a spike now produces:

> Established persistent source, active across 45 days. Currently 4.1x
> its normal output, +23.6 sigma from its baseline — potentially abnormal.

---

## The technical trap, and why the plan called it out first

The task division asks for "hour/day histograms, FFT periodicity" to
infer a shift schedule. **A naive hour-of-day histogram does not work**,
and it fails by producing confident nonsense rather than an obvious error.

Polar-orbiting satellites cross a given point at fixed local solar times.
A histogram of detection hours therefore shows spikes *at the overpass
times* no matter what the facility below does, and an FFT over it will
report a clean 12- or 24-hour "shift pattern" for a flare that burns
continuously. It measures the satellite's schedule, not the plant's.

What survives the bias, and what `rhythm/overpass.py` is built on:

- **Day-of-week comparisons** — the overpass time is the same every day,
  so Monday-13:30 against Sunday-13:30 is fair. This is why
  `weekend_suppression` is sound.
- **Detection rate per overpass slot** — of all the S-NPP daytime passes
  over this cluster, what fraction saw fire? A flare seen on ~100% of
  passes is continuous; one seen on 60% of day passes and 5% of night
  passes has a daytime-operation signature.

Slots are keyed on `(platform, daynight)` — both already on M1's
`hotspots` — rather than hardcoded equator-crossing times, which vary by
satellite and drift over a mission. `slot_consistency()` checks the
premise by confirming each slot really does sit at one local solar hour.

**Sub-daily periodicity is not recoverable** and the package says so in
code (`sub_daily_periodicity_is_not_recoverable()`): the Nyquist limit on
a daily series is a 2-day period. Detecting a two-shift pattern would
need a geostationary sensor — which is exactly what INSAT-3DS is, and why
M1 wired it in, though its fire product is still stubbed pending MOSDAC
access.

---

## What was built

| Module | Purpose |
|---|---|
| `rhythm/overpass.py` | Bias-corrected overpass slots, detection rates, local-solar-hour consistency check |
| `rhythm/fingerprint.py` | `shift_sharpness`, `weekend_suppression`, day/night contrast, schedule anomaly |
| `rhythm/periodicity.py` | Daily-resampled periodogram, restricted to periods the sampling can support |
| `ptsi/baseline.py` | Per-source "normal", with strict point-in-time exclusion |
| `ptsi/index.py` | Persistence index (longevity / reliability / stability) + persistent-intermittent-transient verdict |
| `forecast/kalman.py` | 2-D `[FRP, dFRP/dt]` filter, FRP-scaled process noise, NIS self-check |
| `forecast/escalation.py` | Monte-Carlo time-to-critical with 50%/90% intervals and a reach-probability |
| `forecast/freshness.py` | FRESH / MODERATE / STALE against each source's own cadence |
| `validation/calibration.py` | Walk-forward harness — the Day-10 deliverable |

Two design details worth flagging:

**Weekend = Sunday, not Saturday+Sunday.** The weekly off in Indian
industry is Sunday, with Saturday commonly a working day. Counting both
dilutes a real signal with a working day; a test asserts the India
setting separates a Sunday-off pattern more strongly than the Western one.

**PTSI and the Kalman filter catch different things, deliberately.** On
the synthetic flare above, PTSI flags the spike instantly (+23.6σ) while
the filter barely moves (9.7 MW against an observed 42), because one
outlier is not a trend and a well-behaved filter should resist it. The
instantaneous anomaly and the sustained trend are different questions and
the system answers both rather than conflating them.

---

## The forecast tuning — a two-objective problem found the hard way

This was the substantial work, and the first approach was wrong.

**Attempt 1** tuned the process noise `q` purely on interval coverage and
landed on `q=0.02`, reporting it as "best calibrated". Pooled coverage
looked fine at 87.5%.

**A test I wrote caught that the pooled number was hiding a regime.**
Broken down by source behaviour, the *growing* case covered only 74.7% —
overconfident in exactly the situation that matters. Diagnosing it showed
the filter lagged rising sources by **+159 MW on average** (predicted 375,
actual 535). Root cause: process noise was a fixed absolute value while
measurement noise scaled with FRP, so at high FRP the measurement term
dominated completely, the filter stopped correcting its rate, and the
interval — computed from the lagging prediction — was also too narrow.

**Fix:** make process noise scale with FRP² so the filter is
scale-invariant. The three regimes immediately came into agreement.

**Attempt 2** then re-tuned on coverage and chose `q_rel=5e-5`, reaching
90.4% coverage. **A second test caught that this was worse, not better:**
the filter now reported a rate of **−0.05 MW/h for a genuinely rising
fire**. Coverage had been satisfied by *wide intervals* while the rate
estimate was destroyed — and the rate is the product, because
time-to-critical is computed from it.

| q_rel | coverage 90% | rate estimate (true 1.5 MW/h) |
|---|---|---|
| 1e-7 | 0.709 | **1.04** — usable |
| 1e-6 | 0.775 | 0.46 |
| 5e-6 | 0.836 | 0.20 |
| 2e-5 | 0.882 | −0.72 — a rising fire reported as shrinking |
| 5e-5 | 0.904 | −2.76 |

**Final design:** keep `q` low enough that the rate stays meaningful
(`q_rel=1e-7`) and correct the *interval* separately with an empirical
variance multiplier (`CALIBRATED_STD_MULTIPLIER=2.5`), measured by the
same walk-forward harness. Final measured coverage: **93.4% on a stated
90% interval**, with rate estimates that still track (true 1.5 → 1.04,
true 3.0 → 2.10).

Also disabled: **adaptive process noise**, now off by default. It
inflated `q` from recent innovations, which made sense when `q` was
absolute, but `q` now scales with FRP² and already responds to magnitude.
Stacking the two turned a rate estimate of 1.04 into −1.64.

### A safety-relevant bias, stated plainly

The filter **systematically underestimates growth** (true 1.5 → ~1.04,
true 3.0 → ~2.10). On its own that would push every time-to-critical
*later* than reality — the dangerous direction for evacuation timing.

It is compensated rather than hidden: the calibrated variance multiplier
is applied to the covariance the Monte Carlo samples from, which widens
the rate distribution and pulls the early percentiles back in. The number
a responder should plan against is `ttc_p90_low` — the soonest plausible
arrival — not the median. This is documented in
`escalation.py::_sample_states` where it takes effect.

---

## Bugs found and fixed in self-review

1. **A falling source was reported as escalating.** With the widened
   covariance, even a clearly decaying fire (−3 MW/h) has a thin positive
   tail — about 0.8% of sampled trajectories drew a positive rate — and
   any non-zero count set `escalating=True`. That would have produced
   false alerts on exactly the events an operator most wants filtered
   out, with a self-contradicting reason string reading "0% of sampled
   trajectories reach...". Added `MIN_ESCALATION_PROBABILITY=0.10`; the
   probability itself is still reported in full, only the boolean is
   gated.

2. **`RhythmFingerprint.ok` was tied to the wrong metric.** It required
   `shift_sharpness`, which needs at least two overpass slots. A source
   covered by a single platform in daylight only — a common case —
   computed a perfectly good `weekend_suppression` of 1.0 and was still
   marked unusable, discarding one of the two rhythm features M2
   consumes. `ok` now means "any usable feature".

3. **A division-by-zero path discarded a real signal.** A source active
   *only* at weekends gave `weekday_mean = 0` and returned `None`.
   Agricultural burning genuinely does cluster at weekends in places, so
   that is maximal negative suppression, not undefined; it now returns
   −1.0.

4. **Broken escalation wiring in the task.** Mid-implementation I left a
   placeholder line and a `getattr(ptsi, "_baseline_mean", ...)` for
   attributes that do not exist, so the escalation threshold would always
   have been `None` and no cluster would ever have been flagged
   escalating. Rewritten to thread the `Baseline` through properly — and
   this also revealed that the `baseline_frp_*` columns I had declared on
   `ptsi_registry` were never being written. Both fixed.

Two test-fixture errors were also corrected rather than papered over: a
negative-rate trajectory that clamped at the FRP floor (so the "known
rate" was no longer in the data — the fixture now asserts it does not
clamp), and an assertion on M2's message text that broke on a line wrap
(now asserts the rendered string instead of the source).

---

## Verification performed

- **135/135 M5 tests pass.** Coverage: Planck-style forward simulation
  for the filter, NIS consistency, gap handling, covariance symmetry over
  200 updates, overpass-slot construction and consistency, all three
  rhythm features, periodicity limits, PTSI components and end-to-end
  verdicts, escalation gating and intervals, freshness against
  per-source cadence, and the walk-forward calibration claim.
- **All five suites pass: 37 + 133 + 219 + 145 + 135 = 669 tests.**
- **All 15 M5 modules import cleanly** with `app`, `classifier`,
  `geospatial` and `imagery` resolving to the real packages.
- **Shared MetaData verified**: 18 tables (M1's 5, M2's 3, M3's 4, M4's
  3, M5's 3) on one MetaData with foreign keys resolving.
- **Migration chain verified**: `0001 → 0002 → 0003 → 0004 → 0005 (head)`.
- **ORM/migration column parity** checked for all three M5 tables and
  pinned as a permanent test.
- **End-to-end compute chain** run on a synthetic 45-day flare with a
  spike, producing the brief's own target phrasing.

**Not verified** (needs live infrastructure this session did not set up
and was not asked to): a real Postgres/PostGIS database, Celery over
Redis, and any deployment.

---

## Known limitations

- **Every number here comes from synthetic data.** The calibration claim
  (93.4% coverage) is measured on simulated trajectories with assumed
  30% multiplicative noise and irregular 6–24 h sampling. Re-run
  `validation/calibration.py` against the real 90-day backfill before
  quoting any figure — that is the actual Day-10 deliverable, and it has
  not been done.
- **The rate estimate is conservative.** It underestimates growth by
  roughly 30%, compensated in the interval but not in the point estimate.
  Plan against `ttc_p90_low`, not the median.
- **The 50% interval over-covers (≈73%)** while the 90% interval is
  calibrated. That is an honest consequence of correcting a heavy-tailed,
  biased error distribution with a single variance multiplier: inflating
  enough to fix the tails over-inflates the centre. Present the 90%
  figure, not the 50% one.
- **`MIN_ESCALATION_PROBABILITY = 0.10` is a judgement call.** A steady
  source with wide uncertainty can still cross it (≈19% in testing).
  Raising it reduces false alerts but drops genuinely uncertain real
  escalations; the probability is always exposed so the alert console can
  rank rather than just filter.
- **Expected-passes assumes ~1 pass per slot per day.** True for
  wide-swath VIIRS/MODIS at Indian latitudes, less so where swath gaps
  open up. Rates are clamped to 1.0 to bound the error.
- **PTSI weights (0.40 / 0.35 / 0.25) are reasoned, not fitted.** They
  have not been validated against labelled persistent/transient examples.
- **The Kalman estimate can diverge from the latest observation** (9.7 MW
  against an observed 42 in the worked example). That is correct filter
  behaviour — one outlier is not a trend — but a dashboard must label
  `frp_estimate` as a filtered estimate, not "current FRP", or it will
  read as a bug.
- **The task division's "support M3 on plume dispersion (pyELDQM)" is
  already discharged.** M3 is built and implemented the Gaussian plume
  directly, because pyELDQM could not be verified to exist. Do not
  re-open it.

---

## How to run it (when the DB is up — not done in this session)

```bash
cd member5-temporal
python -m venv .venv && source .venv/Scripts/activate
pip install -r requirements.txt

cd ../member1-ingestion && alembic upgrade head && cd ../member5-temporal

celery -A app.orchestration.queue.celery_app worker \
       -I classifier.tasks,geospatial.tasks,imagery.tasks,temporal.tasks

# and in member1-ingestion/app/main.py:
#   from temporal.api.routes_temporal import router as temporal_router
#   app.include_router(temporal_router)
```

---

## Member 6 (Frontend & Dashboard) — Implementation Changelog

Location: `gateway/`, `static/` — merged into the single project root. Running locally on **http://localhost:8000**.
Nothing was deployed.

---

## 1. Summary of what changed

Your `index.html` was **not rebuilt**. It was taken as the foundation,
copied to `member6-dashboard/static/index.html`, and extended in place.
Every original line of the Cesium implementation is byte-identical; the
Member-6 portal was built *around* it.

Alongside the frontend, Member 6 also owns **architecture Layer 3 — the
API gateway**. M1 built the FastAPI app but only ever mounted its own
three routers, and the `WS /ws/live-updates` channel in the plan was
never implemented. Both are now done, so a single process serves the
dashboard and all five members' APIs.

Files added:

| File | Purpose |
|---|---|
| `static/index.html` | Your file, extended with the Member-6 portal |
| `gateway/routes_dashboard.py` | Aggregation endpoints (`/dashboard/*`) |
| `gateway/live.py` | WebSocket connection manager + broadcast |
| `run_local.py` | One-command local launcher |
| `tests/test_gateway.py` | 21 tests, incl. globe-integrity guards |

Files modified outside M6 — **one file**, `member1-ingestion/app/main.py`:
mounts M2–M5 + M6 routers (each guarded), adds the WebSocket endpoint,
serves the dashboard statically, adds a 503 handler, and reports mounted
modules from `/health`.

**Run it:**
```bash
cd member6-dashboard
python -m venv .venv && source .venv/Scripts/activate
pip install -r requirements.txt
python run_local.py          # → http://localhost:8000
```

---

## 2. Member 6 requirements implemented

### 3D Earth interaction
- **Cursor-following geographic tooltip** showing live latitude, longitude,
  state and nearest district. Coordinates come from
  `camera.pickEllipsoid` → `Cartographic.fromCartesian` → `toDegrees` —
  the *same projection your click handler already used*. Nothing is
  hard-coded or faked.
- Updates smoothly (coalesced to one update per animation frame),
  follows the cursor, flips before hitting a viewport edge, hides when
  the pointer leaves the canvas or moves over a panel, and hides when the
  window loses focus.

### Fire detection interaction
- **Hover** → compact card with Fire ID, Sensor, Acquisition Time, Date,
  Status, Latitude, Longitude, State, District, Range, Beat, Area, FRP.
- **Click** → full record organised into the sections the brief
  specifies: **Fire Detection Details**, **Location** (State, District,
  Circle, Division, Range, Block, Beat), **Geometry** (Area, Perimeter,
  Latitude, Longitude), plus **Thermal Observation** and the incident
  description.

### Layers — 11, all functional
Administrative Boundary · NRT Detections · MODIS · VIIRS · Large Forest
Fires · Fire Prone Map · FFDR · FCM · FTM · Industrial Zones · Industrial
Corridors.

MODIS/VIIRS/Large act as filters on the point set; the five thematic
layers draw graded per-state overlays on the globe. A test asserts every
declared layer has both a toggle element and a working case.

### Search
States, districts and fire detections in one box, with keyboard
navigation (↑/↓/Enter/Esc), a clear button, and an explicit no-match
state. Selecting a district or state flies the **existing** camera and
scopes the filters so the map and the data agree.

### Filters — all change what renders
Date from/to · Sensor · Detection type · Status · State · District ·
Min risk · Min fire size. Plus **Reset filters** and **Zoom to results**.
Statistics and layer counts recompute on every change.

### Controls — every button does something
Zoom In · Zoom Out · Home · Reset · Layers · Filters · Alerts · Measure ·
Export CSV · Fullscreen · Help. A test extracts every `tb-*` button id
from the HTML and asserts each appears in `wirePortalControls` — there
are no decorative buttons.

Also: legend, live statistics strip, alert feed, help modal with
keyboard shortcuts, connection status pill, and toast feedback.

### Responsiveness
Breakpoints at 1180 / 900 / 768 / 520 px. Panels become bottom sheets on
phones, the toolbar collapses to icons, and cursor tooltips are
suppressed on touch (where there is no hover). The globe stays usable at
every size.

---

## 3. Existing functionality preserved

**The globe is untouched.** A test (`test_the_protected_globe_implementation_is_intact`)
asserts all ten load-bearing tokens are still present: Cesium 1.115 CDN,
`new Cesium.Viewer('cesiumContainer'`, `imageryProvider: false`, the
NaturalEarth base layer, both ESRI tile URLs, `labelsLayer.alpha = 0.6`,
`EllipsoidTerrainProvider`, `SkyAtmosphere`, `enableLighting`.

Two further tests assert **all 27 original functions** and **all original
data structures** survive: `initGlobe`, `onCameraChanged`, `altToState`,
`onStateTransition`, `enforceIndiaLock`, `snapToIndia`,
`nudgeCameraToIndia`, `beginDescent`, `setDescentProgress`,
`triggerAutoZoom`, `buildFireLayer`, `firePtColor`,
`buildIndustrialLayer`, `buildCorridorLayer`, `onGlobeClick`,
`toggleLayer`, `openDetail`, `closeDetail`, `renderNearbyHotspots`,
`renderHotspotDetail`, `hotspotCard`, `typeLabel`, `updateHUD`, `bc`,
`updateZoomProgress`, `setStatus`, `showFlash`, `flashState` — plus
`HOTSPOTS`, `CORRIDORS`, `ALT`, `INDIA`, `STATES_ORDER`.

Preserved behaviour: cinematic descent, space auto-rotation, the
seven-stage altitude state machine, India camera lock and snap-back,
breadcrumb, zoom-progress rail, altitude HUD, scanlines/vignette/flash
effects, industrial ellipses, corridor glow polylines, and the original
detail panel and hotspot cards.

### Minimal necessary edits to existing code
Four, each to connect rather than replace:
1. `toggleLayer` — added `case`s for the new layers; the original three
   are unchanged.
2. `onStateTransition` — added `setPortalChrome()` and
   `refreshThematicVisibility()` calls.
3. `onGlobeClick` — tries an exact `scene.pick` first, then falls back to
   your original proximity search unchanged.
4. The fire-pulse loop — now indexes `FIRE_POINTS` instead of `HOTSPOTS`,
   because the collection is rebuilt when filters change. **This was a
   latent bug in the original**: the loop used `HOTSPOTS[i % length]`, so
   any change to the collection's contents would have animated the wrong
   marker sizes.

---

## 4. New features added

Hover tooltips (geographic + fire) · full detail panel with the specified
sections · 11-layer panel with counts · unified search · 8 filters with
reset · 11-button toolbar · legend · live stats strip · alert feed ·
help modal · keyboard shortcuts (`/ L F A H R M ? + − Esc`) · toast
notifications · connection status pill · CSV export (all filtered, or
single detection) · copy-coordinates · two-point distance measurement ·
skip-intro button · empty/error/loading states throughout.

Backend: 4 aggregation endpoints, the WebSocket channel, router
composition for all five members, and a 503 degradation handler.

---

## 5. Assumptions made because backend data was unavailable

The database is not running, so these are stated plainly rather than
presented as real:

- **Administrative hierarchy** (Circle, Division, Range, Block, Beat) is
  derived deterministically from the district via a hash, so the same
  detection always shows the same values. Real values need the FSI forest
  administrative dataset. The detail panel carries a visible note saying
  so.
- **Sensor, acquisition date/time and status** are likewise derived
  deterministically. Thermal values (brightness, FRP, temperature) and
  all site names/coordinates are your original real reference figures,
  untouched.
- **State/district lookup uses bounding boxes**, not survey polygons.
  Adequate for an operator readout; a point near a state border may
  resolve to its neighbour.
- **Area** is derived as `FRP × 0.42 + jitter` so the size filter is
  meaningful; perimeter assumes a circular fire.
- **Thematic layers** (Fire Prone, FFDR, FCM, FTM) render graded
  per-state overlays rather than real rasters. Real deployment swaps them
  for FSI WMS services — the toggle wiring does not change.
- **Risk score** is composed in the gateway from FRP, brightness, class
  and escalation, because no member produces one.

`API.online` drives all of this: when the gateway answers,
`/dashboard/detections` replaces the sample set and the status pill
switches from "Sample data" to "Live API". Mock data lives entirely in
the `API` / `normaliseApiDetection` layer, separate from the UI.

---

## 6. Bugs found and fixed in self-review

1. **All five thematic/admin layer toggles were silently dead.**
   `_m6Layer` was passed inside the `viewer.entities.add({...})` options
   object, but Cesium's `Entity` constructor merges only properties it
   recognises — the key was dropped, so `e._m6Layer` was `undefined` and
   `layers[undefined]` never matched. Fixed by tagging after
   construction. Regression test added.

2. **Duplicate listeners and duplicate dropdown options on live load.**
   `buildFilterControls()` ran again when the API returned data, stacking
   a second copy of all 7 listeners (one slider drag would re-filter
   twice) and appending a second full set of state `<option>`s. Split
   into one-time `wireFilterControls()` and repeatable
   `populateFilterOptions()`. Regression test added.

3. **Database-down returned an opaque 500.** Indistinguishable from a
   real bug. Now a 503 with an actionable hint. Extended to cover the
   missing-driver case as well — a test caught that a fresh checkout
   raises `ModuleNotFoundError: psycopg2` (no Python 3.13 wheel, as M1's
   changelog documents) rather than `OperationalError`.

4. **`/health` reported no modules.** It introspected `app.routes` for
   tags, but this FastAPI version wraps included routers in an object
   without a flat `.tags`, so the list was always empty — the same quirk
   that bit M2's test earlier. Now tracked explicitly at mount time.

5. **Unhandled promise rejection on boot.** `initPortal()` is async and
   was called bare inside a `try/catch`, which cannot catch a rejected
   promise — a failure would have left a half-built UI with only a
   console message. Now has a `.catch` that degrades to offline mode.

---

## 7. Verification performed

- **21/21 M6 tests pass**, including the globe-integrity guards, the
  "every button has a handler" check, the "every layer has a working
  toggle" check, and confirmation that coordinates derive from
  `pickEllipsoid` rather than constants.
- **All six suites pass: 37 + 133 + 219 + 145 + 135 + 21 = 690 tests.**
- **JavaScript validated with `node --check`** on the extracted inline
  script — no syntax errors.
- **Server actually run on localhost:8000.** Verified live: `/health`
  returns all six modules mounted; `/` serves the 130 KB dashboard;
  `/docs` renders; all 21 routes appear in the OpenAPI schema; DB-backed
  endpoints return a clean 503 with a hint.
- **Static mount ordering verified** — a `StaticFiles` mount at `/`
  swallows unmatched paths, so it is registered after every router, and a
  test asserts `/health` and `/openapi.json` are not shadowed.

---

## 8. Remaining issues that cannot be solved from the provided files

- **No live data.** PostgreSQL/PostGIS is not running and `psycopg2` has
  no Python 3.13 wheel on Windows. The dashboard therefore runs on sample
  data. To go live: install `psycopg2-binary` on Python 3.11/3.12 (or use
  the Docker image), `alembic upgrade head`, then run the ingestion
  pipeline. No frontend change is needed — the API layer switches
  automatically.
- **Browser interaction is not automatically tested.** Hover, click,
  drag-zoom and the WebSocket were verified by code inspection, syntax
  checking and HTTP probing, not by driving a real browser. I could not
  visually confirm the rendered globe. Open
  `http://localhost:8000/` and check the console.
- **The WebSocket registry is per-process.** Correct for the single
  worker the demo runs; with multiple Uvicorn workers each holds its own
  registry and a client sees only its own worker's events. The fix is
  Redis pub/sub fan-out (the broker already exists for Celery) — noted in
  `gateway/live.py` rather than built, because it cannot be tested
  without the full stack.
- **Nothing calls `notify_cluster_update()` yet.** The broadcast helper
  is ready, but M1's pipeline would need one line after a cluster is
  persisted to push live updates. The dashboard already handles the
  message type.
- **Real district polygons, FSI thematic rasters and the forest
  administrative dataset** are external data the project does not have.

---

## Unification and post-hoc fixes

Two things happened after the six members' code above was first written, in this order.

### 1. GEE and MOSDAC alternatives (real data, not sample data)

The user had no Google Earth Engine service account and no approved MOSDAC account (2-3 day approval pending). Both were replaced with real, verified-working alternatives rather than sample data:

- **A security finding, fixed first:** `.env.example` (meant to be a safe-to-share template) contained real, live FIRMS and CDSE credentials. Verified live, then moved into `.env` (the file the app actually reads — `.env.example` was never being loaded), `.env.example` scrubbed back to placeholders, `.gitignore` added.
- **Land cover** (`geospatial/attribution/landcover_worldcover_cog.py`) now reads ESA WorldCover 10m directly from the public, keyless `esa-worldcover` AWS S3 bucket via GDAL `/vsicurl/` windowed reads — no GCP account needed. Verified live for Delhi and Dhanbad. The old GEE path (`landcover_zonal.py::zonal_landcover_gee`) is kept as a dormant fallback tier, not deleted.
- **Cloud-fraction gate** (`app/ingestion/cloud_gate.py`) and **Sentinel-2 patch fetch** (`imagery/optical/sentinel2.py`) both now try the new `app/ingestion/cdse_client.py` first — a real client for CDSE's Sentinel Hub Catalog and Process APIs, using the CDSE credentials that were already real and working. Verified live end-to-end: real cloud cover (79.03%, Dhanbad), and a real 6-band Sentinel-2 scene (2026-09-08, Dhanbad) with real reflectance values. GEE is kept as a fallback tier in both, not required.
- **INSAT-3DS/MOSDAC** now has a real temporary backup: `app/ingestion/geostationary_supplementary.py` picks MOSDAC if configured, else EUMETSAT's Meteosat-9 IODC active-fire product (`app/ingestion/eumetsat_iodc.py`) if its (free, near-instant, self-service) credentials are configured, else neither — FIRMS/Sentinel-3 continue regardless, since INSAT-3DS was always supplementary. Filling in MOSDAC credentials later switches back automatically with no code change (tested).

Full detail: see the "GEE/MOSDAC credential alternatives" memory entry, or re-derive from `app/ingestion/cdse_client.py`, `app/ingestion/eumetsat_iodc.py`, and `geospatial/attribution/landcover_worldcover_cog.py`'s docstrings, which carry the verification notes.

### 2. Merged into one project (this change)

The six `memberN-*` folders (each its own installable package, its own `requirements.txt`/`pyproject.toml`/`pytest.ini`, cross-referencing each other via `-e ../memberN` editable installs) were merged into a single project at the repository root:

- `app/`, `classifier/`, `geospatial/`, `imagery/`, `temporal/`, `gateway/`, `static/`, `scripts/` are now top-level directories, all importable from one root with no editable installs.
- One `requirements.txt`, one `pyproject.toml` (`sih162-firesight`), one `pytest.ini` (`pythonpath = .`), one `tests/` directory (all 40 test files moved with zero filename collisions).
- `alembic.ini`, `.env`/`.env.example`, `.gitignore`, `Dockerfile.api`, `Dockerfile.worker`, `docker-compose.yml`, and `run_local.py` all moved to the root; `Dockerfile.api`/`Dockerfile.worker` were updated to `COPY` every package (they previously only copied `app/`, which was correct back when M1's container didn't yet host the whole gateway).
- Fixed the one functionally-real path assumption the move broke: `app/main.py::_mount_dashboard` computed the static directory as `parents[2] / "member6-dashboard" / "static"`; with everything now flat under one root it is `parents[1] / "static"`. `run_local.py`'s multi-package `sys.path` shim was simplified to just adding the project root.
- The six per-member `CHANGES_MEMBER{1..6}.md` files were merged into this one file (sections above), each with its stale `Location:` line corrected to the current path.
- Full test suite re-verified passing from the new unified root after the move (see the memory entry for the exact count at merge time).

### 3. Dashboard replaced and wired to live data (this change)

The Agni Pehchan frontend (`sihfire/index.html` — Globe.gl 3D globe handing
over to a Leaflet 2D map, landing/login screen, control panel, detail panel)
was merged in as the dashboard and connected to the gateway. Everything below
was found by running the merged system against live data, not by reading it.

**The starting position.** The incoming frontend made zero network calls. Its
`genFires()` fabricated ~300 detections with `Math.random()`, and its
`classify()` assigned each a class from hardcoded geofence tables
(`IND`/`FZ`/`UZ`/`AZ`). The outgoing dashboard had its own bundled `HOTSPOTS`
array it fell back to whenever the backend was unreachable. Both are gone. The
dashboard now holds no dataset and no classifier: `test_gateway.py`
asserts `Math.random` does not appear in the page at all, and that none of the
generators come back.

**Wiring.** `static/index.html` reads `/dashboard/detections`,
`/dashboard/event/{id}`, `/dashboard/summary` and `/dashboard/states`, and
subscribes to `/ws/live-updates`. The detail panel shows the model's own SHAP
reasoning, M3's plume (with its wind flagged when assumed), M4's Dozier
temperature and M5's PTSI/escalation state. Anything the pipeline did not
produce reads "not available" rather than a plausible number.

**Bugs found and fixed while verifying:**

1. **The live-push channel had never worked.** `app/main.py`'s WebSocket
   handler took `websocket` with no type annotation. FastAPI resolves handler
   parameters by type, so it treated the parameter as a *query* parameter,
   failed validation, and rejected every handshake with 403 before the body
   ran. Annotated as `WebSocket`; `tests/test_live_updates.py` now opens a
   real socket and asserts the annotation, and was confirmed to fail without
   the fix.

2. **Acquisition times were labelled UTC while being rendered in IST.**
   `hotspots.acq_datetime` is TIMESTAMPTZ, and psycopg returns it already
   converted into the session TimeZone (`Asia/Calcutta` on a default Indian
   install). The dashboard then formatted that object as `"%H:%M UTC"`, so a
   FIRMS acquisition at 06:52 UTC was published as "12:22 UTC". Added
   `_as_utc()` and applied it to every timestamp the endpoint emits.

3. **One population sample read 4.8 GB.**
   `app/enrichment/population.py` did `src.read(1)[row, col]` — decompressing
   the entire 35075 x 34497 WorldPop band to index one cell. Measured at
   90-150 s per call, and it ran inside an open DB transaction, so six
   pipeline workers sat "idle in transaction" while doing it. Replaced with a
   single-pixel windowed read plus a thread-local dataset cache: same values,
   ~15-110 ms. This was the pipeline's dominant cost.

4. **`/dashboard/detections` issued ~8 queries per cluster.** The per-cluster
   readers were being called in a loop, plus a per-cluster query for the
   latest sensor — about 16 000 round-trips for a 2000-detection page, which
   measured at six minutes and left the dashboard stuck on its loading
   screen. Added batch loaders (one query per section, `DISTINCT ON` for the
   sensor) and verified they return byte-identical results to the per-cluster
   readers across 60 clusters and four sections. 2000-limit page: 5.9 s cold,
   1.8 s warm.

5. **`POST /ingest/run` 500'd without a broker, after committing its writes.**
   It always ended in `.delay()`, which with no Redis retries and then raises
   — by which point the ingestion had already committed. One observed call
   left 116 clusters stored and unclassified. It now probes the broker and
   falls back to the in-process pipeline, reporting which path it took.

6. **Overpass could not answer the facility query.** `bulk_load_osm` sent one
   whole-India query for seven tag sets; the public instances reject it
   outright (406). Rewritten to walk 2-degree tiles across three mirrors with
   backoff, committing per tile, with `--resume` and
   `--priority-from-clusters` (tiles containing detections first, because
   facility distance can only change a verdict where a cluster exists).

7. **Sentinel-3 FRP: the credentials cannot download.** Fixed a real
   redirect bug first — `requests` strips `Authorization` across hosts and
   CDSE redirects `catalogue.` to `download.` — but the underlying cause is
   that the configured client is a *Sentinel Hub* client (`sh-` prefix).
   Its token is accepted by the OData catalogue, which is why the
   cloud-cover gate works, and rejected by the download service with
   DAT-ZIP-609 "Token audience not allowed". Now logged as that, in those
   words, instead of a stack trace.

8. **Himawari could not be reached** because `boto3` was in
   `requirements.txt` but not installed. Installed; the bucket now lists
   scenes. The module still yields no records, by its own admission — it has
   no fire-detection algorithm wired in. Pre-existing, unchanged.

9. **Three view controls had handlers but no UI.** `initViewCtrls` wired a
   basemap switcher, an atmosphere toggle and a reset-view button; none of
   those elements existed. Added the Map View panel they expected. The
   basemap switcher then rendered a watermark grid, because CartoDB's
   `light_all` now requires an API key and serves a 200 with an
   "API KEY REQUIRED" tile to unauthenticated callers — switched to ESRI
   World Street Map, which is keyless and already the satellite provider.

10. **The Sensor Data day rows were decorative.** The six checkboxes
    collapsed into a two-value sensor set, so unticking a day changed
    nothing while any box in its group stayed ticked. They now filter on
    `(instrument, acquisition day)`, and the days are labelled from the data
    actually loaded rather than from the browser clock — FIRMS NRT lags, so
    "today" was frequently a row that could never match anything.

11. **The first paint ignored the filters it advertised.** The map plotted
    every loaded detection while the sidebar said "High + Nominal", so the
    count jumped the first time any control was touched. Bootstrap now draws
    through `applyFilt()`.

**Local-run enablement.** TimescaleDB has no current Windows build and the
`hotspots` hypertable is a partitioning optimisation, not a semantic
requirement, so migration `0001` probes `pg_available_extensions` and creates
a plain table when it is absent. Five Geometry columns were given
`spatial_index=False` where the migration also creates the index explicitly —
GeoAlchemy2 was emitting a second `CREATE INDEX` under the same generated name
and the migration failed on a clean database. `app/orchestration/local_pipeline`
drives the real task bodies in dependency order without Celery; ordering
matters, because M4 and M5 write their columns into an *existing*
`cluster_features` row and return early if there is none, so running M2's
feature build after them silently discarded the Dozier temperature and the
rhythm columns.

**New reference data.** `geospatial/admin_boundaries.py` resolves each
detection to its state and district by point-in-polygon against published
district boundaries (`scripts/bulk_load_boundaries.py`). The endpoint had
shipped `"state": None, "district": None` with a comment saying they were
resolved client-side; no such resolution existed. `/dashboard/detections` also
gained `india_only` (default on), because the FIRMS bbox is a rectangle that
also covers parts of Pakistan, Nepal, China, Bangladesh and Myanmar — 222 of
725 clusters in one cycle fell outside India.

**Caching.** `app/geo_cache.py` is an on-disk SQLite cache for the three
per-location network lookups (cloud fraction, land cover, wind). It never
fabricates: a miss returns a sentinel and the caller does the real lookup, and
a *fallback* wind is deliberately never cached, since it is an assumption
rather than an observation. `tests/conftest.py` disables it for the suite —
without that, a test patching a fetcher was served a value an earlier test had
recorded and never called its own mock.

**Verification.** Live FIRMS pull over India, clustered, enriched and
classified end to end with the real XGBoost v2 model (macro-F1 0.89): 841
clusters, 2 669 detections, all classified, zero stage failures. Land cover,
cloud fraction and wind were each really fetched for ~700 clusters (the cache
row counts are the receipt). Full suite: 735 passing.
