# AGNI PEHCHAN — Thermal Anomaly Classification Platform

**Smart India Hackathon 2026 — Problem Statement PS162 (NTRO)**

Agni Pehchan is an AI-driven, end-to-end intelligence platform that ingests thermal anomalies (hotspots) over India from NASA FIRMS and satellite feeds, classifies them into **Industrial Fire**, **Gas Flare**, **Wildfire**, **Agricultural Burning**, or **Mining Anomaly**, and continuously evaluates normal vs. abnormal thermal behaviour over time.

The platform provides live spatial clustering, 28-feature engineering, XGBoost classification with SHAP explainability, Gaussian plume dispersion modeling, threat corridor analysis, temporal escalation forecasting, and an interactive 3D/2D visual operator dashboard.


LINK : https://sihteamalgorythm-production.up.railway.app/
---

## Table of Contents

- [System Overview](#system-overview)
- [System Architecture](#system-architecture)
- [Capability & Source Status](#capability--source-status)
- [Repository Layout](#repository-layout)
- [Environment Configuration](#environment-configuration)
- [Local Setup & Execution](#local-setup--execution)
  - [Option A: Running with Docker Compose](#option-a-running-with-docker-compose)
  - [Option B: Running Locally without Docker](#option-b-running-locally-without-docker)
  - [Bulk Data Loaders & Ops Scripts](#bulk-data-loaders--ops-scripts)
- [Deployment Guide (Railway & Cloud)](#deployment-guide-railway--cloud)
  - [Railway Architecture Overview](#railway-architecture-overview)
  - [Step 1: Provision Database & Caching Services](#step-1-provision-database--caching-services)
  - [Step 2: Deploy API Web Service (`Dockerfile.api`)](#step-2-deploy-api-web-service-dockerfileapi)
  - [Step 3: Deploy Worker & Scheduler Services (`Dockerfile.worker`)](#step-3-deploy-worker--scheduler-services-dockerfileworker)
  - [Step 4: Configure Railway Environment Variables](#step-4-configure-railway-environment-variables)
  - [Step 5: Run Database Migrations & Initial Reference Data Load](#step-5-run-database-migrations--initial-reference-data-load)
  - [Step 6: Standalone / Lightweight Deployment Mode (Alternative)](#step-6-standalone--lightweight-deployment-mode-alternative)
- [API Reference](#api-reference)
- [Machine Learning Model & Feature Pipeline](#machine-learning-model--feature-pipeline)
- [Testing & Quality Assurance](#testing--quality-assurance)
- [Further Reading](#further-reading)

---

## System Overview

NASA FIRMS (Fire Information for Resource Management System) detects thermal anomalies globally via satellites like VIIRS (S-NPP, NOAA-20, NOAA-21) and MODIS. However, satellite sensors only report *"something is hot"* without context. 

Agni Pehchansolves this problem by answering three critical operational questions automatically:
1. **Event Classification:** Is this hotspot an industrial fire, gas flare, wildfire, agricultural stubble burn, or mining activity?
2. **Behavioral Anomaly Detection:** Is this thermal footprint normal for this exact facility/location, or is it an abnormal surge/escalation?
3. **Downstream Risk Assessment:** Which way is the smoke plume traveling, which populations are in the threat corridor, and is the thermal output escalating toward a critical threshold?

---

## System Architecture

```mermaid
flowchart TB
    subgraph EXT["🛰️ External Data Sources"]
        FIRMS["NASA FIRMS\n(VIIRS + MODIS hotspots)"]
        CDSE["Copernicus Data Space\n(Sentinel-2 imagery & cloud fraction)"]
        WORLDCOVER["ESA WorldCover\n(10m land cover raster)"]
        OSM["OpenStreetMap / GEM\n(Industrial facilities & flare registries)"]
        METEO["Open-Meteo\n(Wind direction & speed)"]
    end

    subgraph INGEST["📥 Ingestion & Preprocessing"]
        POLL["Scheduled Poller / Trigger"]
        DEDUP["DBSCAN Spatial Clustering (~500m)\n& Cross-source Dedup"]
        CLOUDGATE["Cloud-Fraction Gate\n(Optical usability check)"]
    end

    subgraph FEATURE["🧮 28-Feature Pipeline"]
        CONTEXT["Contextual: Land cover %, facility dist, pop density"]
        THERMAL["Thermal: FRP, brightness temp, confidence stats"]
        TEMPORAL_F["Temporal: Persistence, detection rate, day/night mix"]
        SPATIAL_F["Spatial: Footprint extent & growth rate"]
        IMAGERY_F["Imagery: Sub-pixel temperature, NDVI/NDBI, smoke ratio"]
        RHYTHM_F["Rhythm: Shift sharpness, weekend suppression, Kalman forecast"]
    end

    subgraph INTEL["🧠 Intelligence & Modeling Layer"]
        EVIDENCE["Evidence-Weighting Engine\n(Handles sparse/missing inputs)"]
        XGB["XGBoost Classifier (Model v5)"]
        SHAP["SHAP Explainability Engine"]
        PTSI["PTSI (Persistent Thermal Source Index)"]
        KALMAN["Kalman Filter Escalation Model"]
        PLUME["Gaussian Plume & Population Exposure"]
    end

    subgraph STORE["🗄️ Persistence"]
        PG["PostgreSQL + PostGIS"]
        TS["TimescaleDB (FRP Time-series)"]
    end

    subgraph API["🌐 API Gateway & Web Application"]
        GATEWAY["FastAPI Gateway (app.main)"]
        WS["WebSocket Push Channel (/ws/live-updates)"]
        DASH["Globe.gl 3D & Leaflet 2D Dashboard"]
    end

    FIRMS --> POLL --> DEDUP --> CLOUDGATE
    CDSE & WORLDCOVER & OSM & METEO --> CONTEXT
    CLOUDGATE --> FEATURE
    FEATURE --> EVIDENCE --> XGB --> SHAP & FUSION
    FEATURE --> PTSI --> KALMAN
    XGB --> PLUME
    SHAP & KALMAN & PLUME --> PG & TS
    PG & TS --> GATEWAY --> WS & DASH
```

---

## Capability & Source Status

| Capability / Source | Status | Details |
|---|---|---|
| **NASA FIRMS Ingestion** | **Working** | Primary hotspot feed (VIIRS S-NPP / NOAA-20 / NOAA-21, MODIS). |
| **DBSCAN Clustering & Cloud Gate** | **Working** | ~500m spatial clustering, deduplication, optical cloud-cover check. |
| **Facility Attribution & Land Cover** | **Working** | OpenStreetMap/GEM infrastructure matching, ESA WorldCover 10m lookup. |
| **Dozier Sub-Pixel Temperature** | **Working** | Dual-band thermal retrieval converging on multi-pixel detection clusters. |
| **Sentinel-2 Indices & Smoke Vector** | **Working** | Fetches spectral patches, calculates NDVI/NDBI, estimates smoke direction. |
| **PTSI & Kalman Escalation Forecast** | **Working** | Persistence index, satellite-bias corrected rhythm, escalation timeline. |
| **XGBoost Classifier + SHAP** | **Working** | Model v5 (macro-F1 0.663 on tile split with corrected supervision rules). |
| **Gaussian Plume & Population Exposure**| **Working** | Wind-driven atmospheric dispersion modeling and threat corridor overlay. |
| **Dashboard & WebSocket Push** | **Working** | Globe.gl 3D to Leaflet 2D view, operator UI, real-time push. |
| **EfficientNet-B0 Image Classifier** | *Working* | Neural network code is present; shipped in `data/image_models/`. Evidence engine discounts confidence; tabular features handle classification. |
| **MOSDAC / INSAT-3DS** | *Integrated* | Ready in `app/ingestion/insat3ds.py`. |
| **Sentinel-3 SLSTR FRP** | *Integrated* | Catalogue searches work via CDSE; bulk download requires CDSE OAuth client credentials rather than Sentinel Hub keys. |
| **Himawari-8/9** | *Scaffolding* | NOAA S3 bucket access verified; raw scenes reachable. |

---

## Repository Layout

```
SIH26/
├── app/                  # Core application logic
│   ├── api/              # FastAPI routes for M1 ingestion, clusters, facilities, auth
│   ├── auth/             # Authentication & user sessions
│   ├── ingestion/        # Satellites poller (FIRMS, CDSE, EUMETSAT, INSAT-3DS)
│   ├── orchestration/    # Celery tasks, scheduler, and broker-free local pipeline
│   ├── geo_cache.py      # SQLite disk cache for network lookups
│   └── main.py           # Unified FastAPI gateway entry point
├── classifier/           # 28-feature contract, XGBoost model, SHAP, evidence weighting
├── geospatial/           # Facility attribution, ESA WorldCover, Gaussian plume, admin boundaries
├── imagery/              # Dozier dual-band retrieval, Planck radiance, Sentinel-2 spectral indices
├── temporal/             # PTSI index, rhythm fingerprinting, Kalman escalation forecasting
├── gateway/              # Dashboard API aggregation endpoints & WebSocket manager
├── static/               # Web Dashboard frontend (Globe.gl 3D Earth, Leaflet 2D, operator UI)
├── scripts/              # Bulk data loaders, training scripts, live cycle executor
├── tests/                # 41 test files (735 tests covering all pipeline stages)
├── data/                 # Models, boundaries, cache, population rasters (gitignored)
├── datasets/             # Historical FIRMS data & WorldPop rasters (gitignored)
├── Dockerfile.api        # Production Dockerfile for FastAPI Gateway API service
├── Dockerfile.worker     # Production Dockerfile for Celery Worker & Scheduler services
├── docker-compose.yml    # Full local multi-container orchestration stack
├── alembic.ini           # Database migration configuration
├── requirements.txt      # Python dependencies
├── run_local.py          # Standalone local launcher (no Docker / broker needed)
├── PROJECT_OVERVIEW.md   # Plain-language architecture walkthrough
└── CHANGES.md            # Comprehensive changelog and implementation rationale
```

---

## Environment Configuration

Copy `.env.example` to `.env` and fill in the required parameters:

```bash
cp .env.example .env
```

### Essential Environment Variables

| Variable | Description | Default / Required |
|---|---|---|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql+psycopg2://sih_user:sih_pass@localhost:5432/sih_thermal` |
| `REDIS_URL` | Redis instance URL | `redis://localhost:6379/0` |
| `AUTH_SECRET_KEY` | JWT Secret Key for dashboard authentication | Production secret key |
| `CORS_ORIGINS` | Comma-separated CORS allowed origins | `*` (set specific domain in production) |
| `FIRMS_MAP_KEY` | NASA FIRMS API key | **Required** (free from NASA FIRMS website) |
| `CDSE_CLIENT_ID` | Copernicus Data Space Client ID | **Required** for optical/cloud check |
| `CDSE_CLIENT_SECRET` | Copernicus Data Space Client Secret | **Required** for optical/cloud check |
| `PORT` | Web service port (injected by Railway/Render) | `8000` |

---

## Local Setup & Execution

### Prerequisites

- Python 3.11+
- PostgreSQL 16+ with PostGIS extension enabled (or Docker)
- Redis 7+ (optional if running in broker-free local mode)

### Standard Setup Steps

1. **Clone the repository and set up a virtual environment:**
   ```bash
   python -m venv .venv
   
   # On Linux/macOS:
   source .venv/bin/activate
   
   # On Windows (cmd):
   .venv\Scripts\activate.bat
   
   # On Windows (PowerShell):
   .venv\Scripts\Activate.ps1
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your FIRMS_MAP_KEY and CDSE credentials
   ```

---

### Option A: Running with Docker Compose

Docker Compose builds PostgreSQL (with PostGIS & TimescaleDB), Redis, the FastAPI gateway, Celery worker, and the scheduler.

```bash
# Build and launch all services
docker compose up --build -d

# Run database migrations
docker compose exec api alembic upgrade head

# Load initial reference data
docker compose exec api python -m scripts.bulk_load_boundaries
docker compose exec api python -m scripts.bulk_load_osm --resume --priority-from-clusters
```

Access services at:
- **Dashboard UI:** `http://localhost:8000/`
- **Interactive API Docs:** `http://localhost:8000/docs`
- **System Liveness Check:** `http://localhost:8000/health`

---

### Option B: Running Locally without Docker

The platform contains an in-process local pipeline (`app/orchestration/local_pipeline.py`) that allows the entire application to run inside a single Python process against a standalone PostgreSQL database without needing Redis or Celery.

1. **Start PostgreSQL with PostGIS:**
   Create database and enable PostGIS:
   ```sql
   CREATE DATABASE sih_thermal;
   \c sih_thermal
   CREATE EXTENSION postgis;
   ```

2. **Apply Database Migrations:**
   ```bash
   python -m alembic upgrade head
   ```

3. **Load Boundary Polygons & Industrial Facility Data:**
   ```bash
   python -m scripts.bulk_load_boundaries
   python -m scripts.bulk_load_osm --resume --priority-from-clusters
   ```

4. **Execute Ingestion & Classification Cycle:**
   ```bash
   python -m scripts.run_live_cycle
   ```

5. **Start FastAPI Gateway & Dashboard:**
   ```bash
   python run_local.py
   ```

---

### Bulk Data Loaders & Ops Scripts

| Script Command | Purpose |
|---|---|
| `python -m scripts.run_live_cycle` | Runs one complete ingestion, clustering, classification, forecasting, and plume modeling cycle in-process. |
| `python -m scripts.bulk_load_boundaries` | Downloads and populates administrative boundaries (states & districts). |
| `python -m scripts.bulk_load_osm` | Fetches industrial infrastructure (refineries, power plants, chemical works) via OpenStreetMap Overpass API tiles. |
| `python -m scripts.bulk_load_gem` | Imports Global Energy Monitor (GEM) and GGFR flare location registries. |
| `python -m scripts.backfill_90day` | Backfills historical satellite hotspots to populate temporal baseline features. |
| `python -m scripts.train_model` | Retrains the XGBoost classifier model against updated label definitions. |

---

## Deployment Guide (Railway & Cloud)

This project is fully containerized and configured for one-click or automated deployment on platforms such as **Railway** (railway.app).

---

### Railway Architecture Overview

On Railway, a full production stack consists of:
1. **PostgreSQL Service with PostGIS Extension**
2. **Redis Service** (Message broker for Celery)
3. **API Web Service** (FastAPI Gateway running `Dockerfile.api`)
4. **Worker Service** (Celery Background Worker running `Dockerfile.worker`)
5. **Scheduler Service** (Celery Beat Scheduler running `Dockerfile.worker`)

```
                 +-------------------------------------------------------+
                 |                    RAILWAY PROJECT                    |
                 |                                                       |
                 |  +-------------------+        +--------------------+  |
                 |  | PostgreSQL Service|        |   Redis Service    |  |
                 |  | (PostGIS enabled) |        |  (Celery Broker)   |  |
                 |  +---------+---------+        +---------+----------+  |
                 |            |                            |             |
                 |            |                            |             |
+--------------+ |  +---------v---------+        +---------v----------+  |
| Public Users |--->|  API Web Service  |        |   Worker Service   |  |
|  / Browser   | |  |  (Dockerfile.api) |        | (Dockerfile.worker)|  |
+--------------+ |  +-------------------+        +--------------------+  |
                 |                                         ^             |
                 |                                         |             |
                 |                               +---------+----------+  |
                 |                               | Scheduler Service  |  |
                 |                               | (Dockerfile.worker)|  |
                 |                               +--------------------+  |
                 +-------------------------------------------------------+
```

---

### Step 1: Provision Database & Caching Services

1. Log into your **Railway Dashboard** and create a **New Project**.
2. **Add PostgreSQL:**
   - Click **+ New** -> **Database** -> **Add PostgreSQL**.
   - After creation, open the database service settings and ensure PostGIS is supported (Railway PostgreSQL includes PostGIS by default).
3. **Add Redis:**
   - Click **+ New** -> **Database** -> **Add Redis**.

---

### Step 2: Deploy API Web Service (`Dockerfile.api`)

1. Click **+ New** -> **GitHub Repo** and select your repository.
2. Go to **Settings** -> **Build & Deploy**:
   - **Build Pack / Dockerfile Path:** `Dockerfile.api`
   - **Custom Build Command:** Leave empty (uses Dockerfile).
   - **Start Command:**
     ```bash
     sh -c "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"
     ```
3. Under **Networking**, click **Generate Domain** to get a public HTTPS URL (e.g. `https://firesight-production.up.railway.app`).

---

### Step 3: Deploy Worker & Scheduler Services (`Dockerfile.worker`)

#### Deploy Celery Worker:
1. Click **+ New** -> **GitHub Repo** (same repo).
2. Rename service to `worker`.
3. Go to **Settings**:
   - **Dockerfile Path:** `Dockerfile.worker`
   - **Start Command:**
     ```bash
     celery -A app.orchestration.queue.celery_app worker --loglevel=info
     ```

#### Deploy Celery Scheduler:
1. Click **+ New** -> **GitHub Repo** (same repo).
2. Rename service to `scheduler`.
3. Go to **Settings**:
   - **Dockerfile Path:** `Dockerfile.worker`
   - **Start Command:**
     ```bash
     python -m app.orchestration.scheduler
     ```

---

### Step 4: Configure Railway Environment Variables

Set the following environment variables across services in Railway (or use **Shared Variables**):

```env
# Database & Broker (Railway provides reference variables like ${Postgres.DATABASE_URL})
DATABASE_URL=postgresql+psycopg2://${POSTGRESUSER}:${POSTGRESPASSWORD}@${POSTGRESHOST}:${POSTPORT}/${POSTGRESDATABASE}
REDIS_URL=${REDIS_URL}

# API Credentials
FIRMS_MAP_KEY=your_nasa_firms_key
CDSE_CLIENT_ID=your_cdse_client_id
CDSE_CLIENT_SECRET=your_cdse_client_secret

# Security & CORS
AUTH_SECRET_KEY=your_generated_secure_random_hex
CORS_ORIGINS=https://firesight-production.up.railway.app
FRONTEND_BASE_URL=https://firesight-production.up.railway.app
```

---

### Step 5: Run Database Migrations & Initial Reference Data Load

Once the database and services are up on Railway, populate initial reference data (boundaries and facilities):

1. **Option A: Via Railway CLI**
   ```bash
   railway run python -m scripts.bulk_load_boundaries
   railway run python -m scripts.bulk_load_osm --resume --priority-from-clusters
   ```

2. **Option B: Via One-Off Task Execution**
   You can trigger a single cycle manually via the API endpoint:
   ```bash
   curl -X POST https://firesight-production.up.railway.app/ingest/run
   ```

---

### Step 6: Standalone / Lightweight Deployment Mode (Alternative)

If you wish to keep resource utilization low on Railway (e.g., using the free tier or a single container):

- You only need **PostgreSQL Service** + **API Web Service**.
- Leave `REDIS_URL` empty or omit worker services.
- The API's `POST /ingest/run` endpoint detects the absence of Celery/Redis and automatically executes cycles **in-process** via `local_pipeline.py`.
- You can trigger periodic ingestion using Railway Cron or an external HTTP trigger hitting `POST /ingest/run` every 30 minutes.

---

## API Reference

The FastAPI gateway exposes REST endpoints and a WebSocket channel:

```
# Core Hotspot & Cluster Management (M1)
GET  /hotspots                # Filtered raw satellite hotspot detections
GET  /clusters                # DBSCAN clustered thermal events
GET  /clusters/{id}           # Detailed cluster metadata
GET  /facilities              # Industrial facility registry
POST /ingest/run              # Manual ingestion & classification trigger
GET  /health                  # Liveness probe & mounted module manifest

# Authentication (M1)
POST /auth/signup             # Operator registration
POST /auth/login              # Authentication & JWT token issuance
GET  /auth/me                 # Authenticated operator profile

# Intelligence & Analytics (M2-M5)
GET  /classify/{id}           # M2: XGBoost 5-class verdict & SHAP feature contributions
GET  /plume/{id}              # M3: Atmospheric Gaussian dispersion plume geometry
GET  /threat/{id}             # M3: Population exposure & threat corridor bounding
GET  /attribution/{id}        # M3: Nearest facility attribution & land cover breakdown
GET  /imagery/{id}            # M4: Dozier sub-pixel fire temperature & Sentinel-2 indices
GET  /ptsi/{id}               # M5: Persistent Thermal Source Index (PTSI)
GET  /forecast/{id}           # M5: Kalman filter escalation forecast & critical ETA
GET  /rhythm/{id}             # M5: Diurnal cadence & weekend suppression pattern
GET  /escalating              # M5: List of clusters exhibiting active escalation

# Gateway & Dashboard Integration (M6)
GET  /dashboard/detections    # High-performance spatial cluster feed for 3D/2D map
GET  /dashboard/event/{id}    # Comprehensive event record (combines M1-M5 data)
GET  /dashboard/summary       # Platform summary stats & thermal metrics
GET  /dashboard/states        # State/district thermal aggregation summaries
WS   /ws/live-updates         # WebSocket push channel for live dashboard updates
```

---

## Machine Learning Model & Feature Pipeline

### XGBoost Classifier (Model v5)

- **Input Dimension:** 28 tabular features
- **Output Classes:** 5 classes (`industrial_fire`, `gas_flare`, `wildfire`, `agricultural_burning`, `mining`)
- **Macro-F1 Score:** `0.663` (evaluated on a strict spatial-tile cross-validation split with updated supervision rules)

### Feature Breakdown

1. **Contextual (3):** ESA WorldCover land-cover dominant class %, distance to nearest mapped industrial facility, local population density.
2. **Thermal (6):** Fire Radiative Power (FRP) max/mean/std, brightness temperature statistics, detection confidence.
3. **Temporal (5):** Multi-day persistence index, historical detection frequency, day/night satellite pass ratio.
4. **Spatial (4):** Cluster convex hull area, point density, spatial expansion rate.
5. **Imagery (5):** Dozier sub-pixel fire temperature, NDVI (vegetation index), NDBI (built-up index), smoke optical ratio.
6. **Rhythm & Dynamics (5):** Shift sharpness (work-hour cadence), weekend suppression ratio, Kalman trend velocity, time-to-critical threshold.

### Explainability & Evidence Weighting

- **SHAP Engine:** Calculates exact feature contributions for every single prediction, enabling operators to see *why* a hotspot was flagged as a gas flare vs. an industrial fire.
- **Evidence Weighting Engine:** Automatically down-weights confidence when certain satellite passes or inputs are missing (e.g. cloud cover blocking optical imagery), preventing false certainty.

---

## Testing & Quality Assurance

The codebase includes 735 automated tests across 41 test modules covering schema migrations, physics calculations, model inference, route non-collision, and WebSocket broadcasting.

To run the complete test suite:

```bash
pytest
```

---

## Further Reading

- [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) — Detailed plain-language walkthrough of the problem statement, feature design, and decision rationale.
- [CHANGES.md](CHANGES.md) — Comprehensive technical implementation log, bug fixes, refactoring history, and design tradeoffs.
