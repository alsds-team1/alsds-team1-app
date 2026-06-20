# Spatial Intelligence Platform (SIP)

### Randm: Jirapa, Zexuan, and Yixuan

#### Web direction: https://alsds-team1-app-e3fed9azh2fpb4ej.eastus-01.azurewebsites.net/

---

## v2.0.1 — Azure SQL Backend

#### Added
- **`migrate_to_azure_sql.py`** — Migration script that reads local SQLite and writes all tables to Azure SQL
- **`/db_structure` Endpoint** — Flask route returning table names and row counts to verify migration success
- **Azure SQL Tables** — Full migration of all Huff engine dependencies:
  - POI table
  - Distance matrix table
  - Visit count table
  - Model parameter table
  - Precomputed / optimized lookup tables

#### Changed
- **`huff_engine.py`** — Replaced local SQLite connection with Azure SQL via `db` module:



**Format:**
```python
from db import get_connection
conn = get_connection()
```

- **Database backend** — Switched from `urban_ai_v2.db` (SQLite file) to `alsds_teamN_db` on shared Azure SQL server `cpl-sql-prod-shared`
- **Function signature preserved** — `run_huff_model()` remains unchanged:

```python
run_huff_model(
    candidate_lat,
    candidate_lon,
    business_category,
    floor_area,
    db_connection=None
)
```

## Verification

After deployment, confirm migration at:

```
https://alsds-team1-app-e3fed9azh2fpb4ej.eastus-01.azurewebsites.net/
```

## Deployment Workflow

```
dev branch → testing → merge into main → deployment
```

1. Push changes to GitHub
2. Wait for GitHub Actions deployment to complete
3. Test the deployed Azure Web App

---



## v2.0.0 — Azure Cloud Deployment

#### Added
- **Azure Web App Deployment** — Full end-to-end deployment of the ALSDS application via GitHub Actions
- **`/health` Endpoint Verification** — Confirms live app is running and database is reachable
- **End-to-End System Validation** — Confirms homepage, map, model inference, results display, and chatbot all function on Azure

#### Changed
- **`huff_engine.py`** — Replaced baseline model with V3 implementation matching required function signature:

```python
run_huff_model(candidate_lat, candidate_lon, business_category, floor_area, db_connection=None)
```

- **Database Connection** — Switched from absolute local paths to relative path:

```python
sqlite3.connect("Data/your_team.db")
```

- **Return structure** now includes:

```
predicted_visits
market_share
competitors
runtime_ms
notes
```

## Deployment Steps

| Step | Action |
|---|---|
| 1 | Replace `huff_engine.py` with V3 implementation |
| 2 | Place database at `/Data/your_team.db` (relative path) |
| 3 | Push updated code to GitHub |
| 4 | Wait for GitHub Actions to complete deployment |
| 5 | Open Azure Web App URL and verify system |

---




## v1.0.2 — Migration v2 & Inference Engine v3


#### Added
- **`migration_v2.py`** — One-time setup script that builds and optimizes the full database
- **CBG Master Enrichment** — Combines demographic data with projected centroids (X and Y coordinates in EPSG:26919)
- **Competitor Bottleneck Solver** — Pre-calculates the sum of competitor utilities for every category and every neighborhood
- **`Competitor_Summary` Table** — Stores all pre-computed utility sums for instant retrieval at inference time
- **SQL Indexes** — Applied to `geoid` and `top_category` columns for sub-millisecond query performance
- **`huff_engine_v3.py`** — Fully refactored inference engine with zero file reads and optimized math
- **Performance Benchmarking Suite** — Measures total execution time across all three engine versions

#### Changed
- **`huff_engine_v2.py` → `huff_engine_v3.py`** — Removed all `pd.read_csv()` and `json.load()` calls; all data sourced exclusively from SQLite
- **Competitor Gravity Calculation** — Replaced per-CBG loop with a single database fetch of the pre-computed sum
- **`urban_ai.db` → `urban_ai_v2.db`** — Optimized schema with indexed tables and denormalized competitor sums

#### Security
- All user inputs (Latitude, Longitude, Category) use parameterized SQL queries (`?` placeholders) to prevent SQL injection

## Deliverables

| File | Description |
|---|---|
| `migration_v2.py` | Builds and optimizes the SQLite database |
| `huff_engine_v3.py` | Refactored inference engine (zero file reads) |
| `urban_ai_v2.db` | Optimized local SQLite database file |
| `performance_report.pdf` / `.md` | Execution time comparison (V1 vs V2 vs V3) + schema efficiency explanation |

---



## v1.0.1 - Database Migration & Engine Refactor

#### Added
- **SQLite3 Database (`urban_ai.db`)** — Local database replacing raw CSV lookups for all model data
- **CBG Master Table** — Merges demographic data from `worcester_cbgs.csv` with spatial coordinates (`INTPTLAT10`, `INTPTLON10`) from GeoJSON into a single table
- **Projected Coordinate Pre-computation** — Stores pre-calculated X and Y coordinates (EPSG:26919) per CBG instead of raw degrees
- **Pre-computed Competitor Utility Table** — Stores the Utility Sum for every category in every CBG; at runtime only the new site's utility is calculated and added to the existing sum
- **`migration_script.py`** — Migrates all CSV/GeoJSON source data into SQLite
- **`huff_engine_v2.py`** — Refactored engine querying the database instead of reading files
- **Performance Benchmarking** — Timed comparison between Old File Method vs. New Database Method

#### Changed
- **`huff_engine.py` → `huff_engine_v2.py`** — Removed all `pd.read_csv()` and `json.load()` calls; replaced with `sqlite3` parameterized queries (`?` placeholders) to prevent SQL injection

#### Architecture
- Applied normalization vs. denormalization strategy — category parameters kept in separate tables; frequently joined data merged for faster retrieval


## Deliverables

| File | Description |
|---|---|
| `migration_script.py` | Moves CSV/GeoJSON data into SQLite |
| `urban_ai.db` | Local SQLite database file |
| `huff_engine_v2.py` | Refactored engine querying the database |
| `screenshot.png` | Terminal showing model run success and performance time |

---


## v1.0.0 — Initial Release

#### Added
- **Parameter Lookup** — Resolves `Alpha` and `Beta` values from `calibrated_parameters_filtered.csv` using a user-supplied `top_category` or NAICS code
- **Distance Mapping** — Projection-based distance function computing distances from candidate site to all ~150 CBGs
- **Competitive Context Engine** — Per-CBG competitor utility calculation (`Area^alpha / distance^beta`) sourced from `worcester_cbg_poi_distance.csv`
- **Probability Share Calculation** — Huff model formula: `P_new = U_new / (U_new + ΣU_existing)`
- **Visit Estimation** — Multiplies `P_new` by historical visit totals from `worcester_cbg_poi_visits.csv` to output Total Predicted Visits
- **`predict_site.py`** — Clean, commented Python script encapsulating the full pipeline


## Validation Case

| Parameter | Value |
|---|---|
| Store Type | New Liquor Store |
| Coordinates | `42.27, -71.80` |
| Floor Area | `2,500 sq meters` |
| Output | Total Predicted Visits |


## Data Dependencies

| File | Purpose |
|---|---|
| `calibrated_parameters_filtered.csv` | Alpha & Beta model parameters |
| `worcester_cbg_poi_distance.csv` | Competitor distances per CBG |
| `worcester_cbg_poi_visits.csv` | Historical visit counts per CBG |

