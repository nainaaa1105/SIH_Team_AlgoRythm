# SIH162 FireSight — Thermal Anomaly Classification Platform

Smart India Hackathon PS162 (NTRO): classifies NASA FIRMS thermal
anomalies over India into industrial fire, gas flare, wildfire,
agricultural burning or mining, and distinguishes normal from abnormal
industrial behaviour over time. One unified project — ingestion,
classification, geospatial analysis, imagery, temporal forecasting and
the dashboard/API gateway all live here as top-level packages, not as
six separate installable projects.

Nothing in this repository is deployed anywhere. Everything runs locally
only, by design.

## What works, and what doesn't

Stated up front so nobody has to discover it by running the thing.

| Capability | Status |
|---|---|
| NASA FIRMS ingestion (VIIRS S-NPP / NOAA-20 / NOAA-21, MODIS) | **Working** — the primary feed |
| DBSCAN clustering, dedup, cloud gate | **Working** |
| Facility attribution, land cover, population, state/district | **Working** |
| Dozier sub-pixel fire temperature (dual-band retrieval) | **Working** — converges on multi-detection clusters |
| Sentinel-2 patches, spectral indices, smoke bearing | **Working** |
| PTSI persistence, rhythm fingerprint, Kalman escalation forecast | **Working** |
| XGBoost classification + SHAP explanations | **Working** — model v5, macro-F1 0.663 (v2 scored 0.893 but was trained/evaluated against a looser weak-label rule set since found to mislabel real industrial fires and agri burns as mining — see `classifier/labels/rules.py`; v5 reflects the corrected labels) |
| Gaussian plume, threat corridor, population exposure | **Working** |
| Dashboard, live WebSocket push, incident report export | **Working** |
| EfficientNet-B0 image classifier | **No trained weights.** `data/image_models/` does not exist, so `image_predicted_class` is always null and late fusion runs tabular-only. The evidence-weighting engine records the gap and discounts confidence. The Sentinel-2 *indices* still reach the model. |
| MOSDAC / INSAT-3DS | **Returns 0 rows.** The product path in `app/ingestion/insat3ds.py` is best-effort scaffolding, not a verified endpoint — MOSDAC has no public REST API and needs an approved account. Credentials being present also *blocks* the EUMETSAT fallback, since presence is what selects the source. |
| Sentinel-3 SLSTR FRP | **Returns 0 rows.** The configured CDSE credentials are a Sentinel Hub client (`sh-` prefix): the token is accepted by the OData catalogue — which is why the cloud gate works — and rejected by the download service with `DAT-ZIP-609 "Token audience not allowed"`. Bulk download needs a CDSE OAuth client, not a Sentinel Hub one. |
| Himawari-8 | **Returns 0 rows.** The AWS bucket is reachable and lists scenes; no fire-detection algorithm is wired in. |

All three inactive sources are supplementary by design — cross-validation
and geostationary refresh rate. Nothing in the classification pipeline
depends on them.

## Layout

```
app/            ingestion (FIRMS/CDSE/EUMETSAT/Himawari/INSAT), dedup,
                DBSCAN clustering, cloud gate, enrichment, DB schema +
                Alembic migrations, the FastAPI gateway (app/main.py),
                Celery orchestration AND the broker-free local runner
                (app/orchestration/local_pipeline.py)
                app/geo_cache.py — on-disk SQLite cache for the three
                per-location network lookups
classifier/     28-feature contract, XGBoost + (dormant) EfficientNet
                late fusion, SHAP explainability, evidence weighting,
                label rules
geospatial/     facility attribution, ESA WorldCover land cover, Gaussian
                plume dispersion, threat corridors, evacuation routing,
                admin_boundaries.py — state/district point-in-polygon
imagery/        Dozier sub-pixel fire temperature, Planck radiance,
                Sentinel-2 spectral indices / smoke direction,
                EfficientNet-B0 classifier (no weights shipped)
temporal/       PTSI (persistent thermal source index), rhythm
                fingerprinting (satellite-bias corrected), Kalman
                forecast + escalation
gateway/        dashboard aggregation endpoints + WebSocket live-push
static/         the dashboard itself (Globe.gl 3D globe -> Leaflet 2D map,
                operator UI, live-wired to the gateway)
scripts/        bulk loads and jobs — see the table below
tests/          41 test files, flat, one pytest run for everything
archive/        the superseded Cesium dashboard, deliberately OUTSIDE
                static/ so it is never served (it carries a sample-data
                fallback)
data/           models/ (trained XGBoost), boundaries/, cache/,
                population/, patches/, thumbnails/ — all gitignored
datasets/       historical FIRMS CSVs + WorldPop rasters (gitignored)
```

`member6-dashboard/` is an empty leftover from the pre-merge structure and
can be deleted.

Every package's Celery tasks, DB models (declared against one shared
SQLAlchemy `Base`) and feature-column ownership are cross-referenced —
see `CHANGES.md` for the full per-stage data flow and the reasoning
behind every design decision, bug fix, and known limitation.

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate      # or .venv\Scripts\activate on cmd
pip install -r requirements.txt
cp .env.example .env               # fill in real credentials — see below
```

### Credentials — what's real, what's optional

| Service | Required for | Status |
|---|---|---|
| `FIRMS_MAP_KEY` | Primary hotspot feed (VIIRS/MODIS) | Free, instant, self-service at firms.modaps.eosdis.nasa.gov/api/map_key |
| `CDSE_CLIENT_ID`/`SECRET` | Land cover, cloud gate, Sentinel-2 imagery | Free, self-service at dataspace.copernicus.eu — this is the primary path, not a fallback. A Sentinel Hub (`sh-`) client works for catalogue and Process API but **not** for bulk product download. |
| `MOSDAC_USERNAME`/`PASSWORD` | INSAT-3DS (supplementary only) | Manual approval, 2-3 days. Setting these selects MOSDAC over the EUMETSAT backup, so leave blank unless the account is approved *and* the product path in `insat3ds.py` has been confirmed. |
| `EUMETSAT_CONSUMER_KEY`/`SECRET` | Temporary INSAT-3DS backup while MOSDAC is pending | Free, near-instant, self-service at api.eumetsat.int — optional |
| `GEE_SERVICE_ACCOUNT` | Nothing required — dormant fallback only | Not needed; land cover/cloud gate/Sentinel-2 all work without it |

## Run locally

### With Docker

```bash
docker compose up --build            # Postgres/PostGIS + Redis + api + worker + scheduler
docker compose exec api alembic upgrade head
python -m scripts.bulk_load_osm
python -m scripts.bulk_load_gem --gem path/to/tracker.xlsx --ggfr path/to/ggfr.csv
```

### Without Docker (Windows/macOS/Linux, no admin rights needed)

The fan-out to M2-M5 is a Celery topology, and Celery needs Redis. With
no broker running, every `.delay()` raises and clusters land in the
database but are never classified. `app/orchestration/local_pipeline`
drives the same task bodies in-process instead, so the whole platform
runs from one Python process against a plain PostgreSQL install.

**1. PostgreSQL + PostGIS.** Any 16.x server works. The portable
EnterpriseDB binaries plus the OSGeo PostGIS bundle need no installer and
no administrator rights — unzip both into one directory, then:

```bash
initdb -D <datadir> -U sih_user --pwfile=<file containing the password>
pg_ctl -D <datadir> -l <datadir>/server.log -o "-p 5432" start
psql -U sih_user -d postgres   -c "CREATE DATABASE sih_thermal"
psql -U sih_user -d sih_thermal -c "CREATE EXTENSION postgis"
```

A portable install is **not** a system service — start it again after
every reboot, before starting the app.

TimescaleDB is optional. It has no current Windows build, and the
`hotspots` hypertable is a time-partitioning optimisation rather than a
semantic requirement — migration `0001` probes `pg_available_extensions`
and creates a plain table when it is absent.

**2. Driver.** `psycopg2-binary` has no Python 3.13 wheel. `psycopg`
(v3) does, and SQLAlchemy speaks it under a different scheme, so on 3.13
set:

```
DATABASE_URL=postgresql+psycopg://sih_user:sih_pass@localhost:5432/sih_thermal
```

**3. Schema and reference data.**

```bash
python -m alembic upgrade head
python -m scripts.bulk_load_boundaries                             # state/district polygons
python -m scripts.bulk_load_osm --resume --priority-from-clusters  # industrial facilities
```

`bulk_load_osm` walks India in 2-degree tiles across several Overpass
mirrors: the public instances reject a single whole-India query for these
tag sets outright (HTTP 406). It commits per tile, `--resume` skips tiles
already loaded, and `--priority-from-clusters` fetches the tiles that
actually contain detections first — facility distance can only change a
verdict where a cluster exists, so this makes a multi-hour walk useful
within the first few minutes.

**4. Ingest and classify.**

```bash
python -m scripts.run_live_cycle          # FIRMS -> clusters -> XGBoost + SHAP
python run_local.py                       # serve the dashboard + API
```

- Dashboard: http://localhost:8000/
- API docs: http://localhost:8000/docs
- Health (shows which routers mounted): http://localhost:8000/health

### scripts/

| Script | What it does |
|---|---|
| `run_live_cycle` | One full cycle in-process: ingest, enrich, classify, forecast, model plumes. `--no-ingest` to only finish stored clusters, `--reclassify` to redo every cluster, `--limit`, `--workers`. |
| `bulk_load_osm` | Tiled Overpass walk for industrial facilities. `--resume`, `--priority-from-clusters`, `--workers`. |
| `bulk_load_boundaries` | Fetches the district/state GeoJSON the resolver reads. |
| `bulk_load_gem` | Global Energy Monitor + GGFR flare registries. |
| `bulk_load_landcover` | Optional local MODIS MCD12Q1 raster (third-tier land-cover fallback). |
| `backfill_90day` | Historical FIRMS windows — the route to filling the sparse temporal features. |
| `backfill_training_labels` | Rebuilds training labels from the rule set. |
| `train_model` / `ingest_and_train` | Retrains the XGBoost classifier and writes a new version to `data/models/`. |

Re-run `run_live_cycle` whenever you want a fresh satellite pass; add
`--reclassify` after loading more facilities, since facility distance
feeds three of the model's twenty-eight features.

## API surface

```
GET  /hotspots  /clusters  /clusters/{id}  /facilities        M1 ingestion
GET  /classify/{id}                                           M2 classification
GET  /plume/{id}  /threat/{id}  /attribution/{id}             M3 geospatial
GET  /imagery/{id}  /imagery/{id}/patches                     M4 imagery
GET  /ptsi/{id}  /forecast/{id}  /rhythm/{id}  /escalating    M5 temporal
GET  /dashboard/detections  /dashboard/event/{id}
     /dashboard/alerts  /dashboard/summary  /dashboard/states  M6 gateway
WS   /ws/live-updates                                         live push
POST /ingest/run                                              manual cycle
GET  /health                                                  liveness + mounted routers
```

`/dashboard/detections` takes `india_only` (default on): the FIRMS bbox is
a rectangle that also covers parts of Pakistan, Nepal, China, Bangladesh
and Myanmar, and roughly a third of the clusters in a typical cycle fall
outside India.

`POST /ingest/run` detects whether a Celery broker is reachable. With one,
it dispatches as the deployment does; without one it runs the cycle
in-process — which is also the only path on which the WebSocket push
reaches open dashboards, since the broadcast goes to the API process's own
connection manager.

## The model

XGBoost, five classes (`industrial_fire`, `gas_flare`, `wildfire`,
`agricultural_burning`, `mining`), 28 features, macro-F1 **0.663** on a
spatial-tile split (119,782 train / 30,178 test, model v5). An earlier
version (v2) scored 0.893 on a smaller split, but its weak-supervision
labels for mining and agricultural_burning were later found to be
leaky — see `classifier/labels/rules.py`'s `mining_rule`/`agri_burn_rule`
docstrings — so that number reflected an easier, partly-wrong ground
truth rather than genuinely higher accuracy. v5 is trained against the
corrected label rules. Feature order is
persisted alongside the model and re-validated at inference, so a
train/predict mismatch fails loudly instead of silently scrambling the
vector.

Feature groups: context (land cover, facility proximity, population),
thermal (FRP/brightness statistics), temporal (persistence, cadence,
day/night), spatial (footprint extent and growth), imagery (Dozier
temperature, NDVI, NDBI, smoke ratio), rhythm (shift sharpness, weekend
suppression, Kalman time-to-critical).

Several groups are legitimately sparse on a 3-day ingestion window —
`weekend_suppression` needs a week of history, `frp_zscore` needs a prior
baseline, Dozier converges only on well-constrained clusters. The
evidence-weighting engine discounts confidence accordingly rather than
imputing zeros, which is why live confidences sit in the 30-60% band.
`scripts/backfill_90day` is the route to filling them.

### About FIRMS_DAY_RANGE

FIRMS NRT publishes on a lag. A `FIRMS_DAY_RANGE=1` query over India
routinely returns zero rows simply because the current day's granules
have not been processed yet; `3` is the smallest window that reliably
returns data. The archive endpoint caps a single query at 5 days
(confirmed against the live API's own error text — `scripts/backfill_90day.py`
loops in 5-day windows, not the 10 some FIRMS docs suggest).

### No sample data

The dashboard has no bundled dataset and no client-side classifier. Every
detection, class, explanation, plume and population figure on it comes
from the live pipeline. When the gateway is unreachable or the database
is empty it says so on a banner and shows nothing, rather than
substituting stand-in records. `tests/test_gateway.py` enforces this — it
fails if `Math.random`, a `genFires()` generator or a bundled `HOTSPOTS`
array reappears in `static/index.html`.

## Tests

```bash
pytest
```

735 tests across 41 files, one run. Covers schema/migration parity, Celery
task contracts and shared `MetaData`, column-ownership non-collision,
route non-collision, physics/statistics correctness (Planck/Dozier
round-trip, Kalman forward simulation, walk-forward calibration
coverage), the WebSocket handshake and broadcast, and the dashboard
integrity checks. See `CHANGES.md` for what each file is proving and why
it matters.

`tests/conftest.py` disables the geo cache for the whole suite — without
that, a test patching a network fetcher can be served a value an earlier
test recorded and never call its own mock.

## Further reading

- `CHANGES.md` — the full implementation log: how each part was built,
  every bug found in self-review and how it was fixed, known
  limitations, and the later unification, credential-alternatives and
  dashboard-replacement work.
- `PROJECT_OVERVIEW.md` — the plain-language architecture walkthrough:
  the problem, the layers, what the 28 features are, and why the design
  decisions were made.
