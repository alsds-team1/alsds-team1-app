# Spatial Intelligence Platform (SIP)

### Randm: Jirapa, Zexuan, and Yixuan

#### Web direction: https://alsds-team1-app-e3fed9azh2fpb4ej.eastus-01.azurewebsites.net/

---

## v2.6.0 — Responsive Layout

### Changed
- Widened the page container (`.als-wrap`) from `1120px` to `1440px` and centered the
  tool area at the same width, so the homepage and tool fill large monitors instead of
  sitting in a narrow column with empty side margins.
- Mobile/tablet breakpoints (`900px`, `860px`) left intact — phones and narrow windows
  still stack correctly.

---

## v2.5.1 — Chatbot Behavior & Chart Fixes

### Fixed
- **Chart crash** — removed a reference to an undefined `drawMeanLinePlugin` that threw
  `drawMeanLinePlugin is not defined` and blanked the bar chart. Replaced with a properly
  defined candidate reference-line plugin (`drawCandidateLinePlugin`).
- **Pie chart** — no longer mixes incompatible scales or shows duplicated competitors.
- Removed a duplicate market-share-pie render call.

### Changed
- **Guided chatbot** — no longer asks the user to re-confirm inputs it already has
  ("Should I proceed?" / "use the same details?"). When only the location changes, it
  now re-runs immediately using the known category and floor area, and only asks when a
  required input is genuinely missing.

---

## v2.5.0 — Real Competitor Intelligence

### Added
- **Real competitor stores** — competitors now come from the `pois` table (deduped per
  store via `placekey`), each with real coordinates, store size (m²), total historical
  visits/day, and a relative Huff pull (`size^α / dist^β`, normalized so the strongest = 1.0).
- **Competitor bar chart** — plots nearby stores by real visits/day, with a dashed
  "your store (predicted)" reference line for comparison.
- **Visit-share pie** — share of daily visits among your store and the nearby competitors
  (kept distinct from the true Worcester market share, which stays in the scorecard).
- **Map markers** — real competitor stores are now plotted on the map (they finally carry
  lat/lon, which the marker code always expected).

### Changed
- Competitor selection switched from "nearest 12 overall" to **within a 3-mile radius** of
  the candidate pin (`COMPETITOR_RADIUS_MILES = 3.0`), so the list genuinely changes as the
  location moves — and correctly shows none in areas with no nearby competitors.
- Clarified headings so labels match the data: bar chart = "Top competitors by visits/day",
  neighborhood chart = "capture probability", results table = "Nearby competitor stores".

---

## v2.4.0 — Results Scorecard & Charts

### Added
- **Model Result scorecard** — Predicted visits (highlighted headline / "key result"),
  Market share, and Worcester demand, each with units (`visits/day`, `%`, `m²`).
- **Animated charts** — the competitor bar chart updates in place on re-run instead of
  being destroyed and rebuilt, so bars animate to new values.
- **Units on table headers** — Distance (mi), Size (m²), Visits/day.

### Changed
- Result figures are framed as rounded estimates rather than exact counts.

---

## v2.3.0 — Brand & Homepage Redesign

### Changed
- Renamed the product from **"ALSDS — Location Decision Support"** to
  **"Spatial Intelligence Platform (SIP)"** across the brand line, intro copy, and page
  title; enlarged the brand to a 26px display-font title.
- Reworded the "How it works" steps to the real flow:
  choose the business type → enter coordinates + floor area → read the trade area.
- Redesigned the homepage hero (gravity / cartographic theme, teal–blue–amber palette,
  animated "pull-field" graphic).

### Removed
- The "Ready to test a location?" closing bar and its now-unused styles.

---

## v2.2.0 — Model Transparency & Honest Calibration

### Added
- **`notes` field** on every engine result — calibration state, straight-line-distance
  caveat, and any dropped neighborhoods.
- **`no_demand_data` flag** and **`total_demand`** — categories with no demand are reported
  as a data gap, not as a bad location (prevents misreading 0 visits / 0 share).
- **Expanded NAICS resolver** — 213-entry fallback mapping granular codes to canonical
  categories, with three tiers: 0 = calibrated, 1 = rough (uncalibrated but has demand),
  2 = unsupported.

### Changed
- Engine drops CBGs with missing centroid coordinates so they don't distort market share.
- The assistant now opens every result with a one-line confidence label
  (**Calibrated** / **Rough estimate** / **No data**) and flags a "modest fit" when the
  calibration correlation is below ~0.4.
- Shortened neighborhood labels to `Tract X · BG Y`.

### Fixed
- Granular NAICS codes that map to a calibrated category are now correctly tiered as
  calibrated (detection is by category **name**, not by the exact code).

---

## v2.1.0 — AI Controller Chatbot

### Changed
- Replaced the fixed JavaScript state-machine chatbot with an **"AI as controller"**
  design. A stateless `/api/chat` endpoint runs an Azure OpenAI tool-calling loop; the
  model collects the inputs in any order and calls the real Huff engine as a tool
  (`run_huff_model`).
- `static/chat.js` reduced to a thin client that posts the conversation to `/api/chat`
  and renders the returned result.

### Added
- Core guardrail — the assistant never invents numeric results. Predicted visits, market
  share, and distances come **only** from the engine tool output.

---


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

