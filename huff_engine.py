"""
Azure SQL Huff Engine

This version removes the local SQLite dependency and queries Azure SQL through db.py.
It exposes run_huff_model(), which is the function app.py already expects.
"""

import difflib
import json
import time
from typing import Any, Dict, Optional

import pandas as pd
from pyproj import Transformer

from db import get_connection


def _load_geoid_to_name_mapping(geojson_path: str = "Data/worcester_cbgs_map.geojson") -> Dict[str, Dict[str, str]]:
    """Load GeoJSON and create a mapping from GEOID10 to a small props dict.

    The returned mapping value is a dict containing keys we may use for naming:
    - 'name' : NAMELSAD10 if present
    - 'tract': TRACTCE10 if present
    - 'blk'  : BLKGRPCE10 if present
    """
    try:
        with open(geojson_path, 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)

        mapping: Dict[str, Dict[str, str]] = {}
        for feature in geojson_data.get("features", []):
            props = feature.get("properties", {})
            geoid = props.get("GEOID10")
            if not geoid:
                continue

            entry: Dict[str, str] = {}
            if props.get("NAMELSAD10"):
                entry["name"] = str(props.get("NAMELSAD10"))
            if props.get("TRACTCE10"):
                entry["tract"] = str(props.get("TRACTCE10"))
            if props.get("BLKGRPCE10"):
                entry["blk"] = str(props.get("BLKGRPCE10"))

            mapping[str(geoid)] = entry

        return mapping
    except Exception as e:
        # If GeoJSON loading fails, return empty dict
        # (will fall back to geoid-based naming)
        print(f"Warning: Could not load GeoJSON mapping: {e}")
        return {}


def _find_category_parameters(conn, user_category: str):
    """Find category parameters by exact category, NAICS code, substring, or fuzzy match."""
    params = pd.read_sql(
        "SELECT top_category, naics_code, alpha, beta, correlation FROM category_parameters",
        conn,
    )
    params["top_category"] = params["top_category"].astype(str)
    params["naics_code"] = params["naics_code"].astype(str)

    query = str(user_category).strip()

    exact = params[params["top_category"].str.lower() == query.lower()]
    if len(exact) > 0:
        return exact.iloc[0], False

    naics = params[params["naics_code"] == query]
    if len(naics) > 0:
        return naics.iloc[0], False

    contains = params[params["top_category"].str.contains(query, case=False, na=False)]
    if len(contains) > 0:
        return contains.iloc[0], False

    matches = difflib.get_close_matches(query, params["top_category"].tolist(), n=1, cutoff=0.55)
    if matches:
        return params[params["top_category"] == matches[0]].iloc[0], False

    fallback = pd.Series(
        {
            "top_category": query,
            "naics_code": "Unknown",
            "alpha": 1.0,
            "beta": 1.0,
            "correlation": None,
        }
    )
    return fallback, True


def predict_site(lat: float, lon: float, category_query: str, store_area_sq_m: float) -> Dict[str, Any]:
    """Run the Huff model against Azure SQL and return detailed results."""
    start = time.perf_counter()

    with get_connection() as conn:
        params, used_fallback = _find_category_parameters(conn, category_query)
        matched_category = str(params["top_category"])
        naics_code = str(params["naics_code"]) if not pd.isna(params["naics_code"]) else "Unknown"
        alpha = float(params["alpha"])
        beta = float(params["beta"])
        correlation = None if pd.isna(params["correlation"]) else float(params["correlation"])

        transformer = Transformer.from_crs("EPSG:4326", "EPSG:26919", always_xy=True)
        new_x, new_y = transformer.transform(float(lon), float(lat))

        # Azure SQL / pyodbc uses ? placeholders. This is parameterized and safe.
        # Also join the POI table to get a representative POI name for each CBG.
        # We pick the single POI with the largest historical visit_count for that CBG
        # (if any) restricted to the matched top_category. This provides a
        # human-friendly `location_name` for display instead of the raw CBG name.
        cbg_data = pd.read_sql(
            """
            SELECT
                c.geoid,
                c.total_population,
                c.median_household_income,
                c.x_26919,
                c.y_26919,
                COALESCE(s.total_u_existing,      0) AS total_u_existing,
                COALESCE(d.total_category_visits, 0) AS total_category_visits,
                tp.location_name AS top_location_name,
                tp.placekey AS top_placekey,
                tp.visit_count AS top_poi_visit_count
            FROM cbg_master AS c
            LEFT JOIN Competitor_Summary AS s
                ON c.geoid = s.geoid AND s.top_category = ?
            LEFT JOIN category_demand AS d
                ON c.geoid = d.geoid AND d.top_category = ?
            LEFT JOIN (
                -- Choose the single POI in each CBG with the most historical visits
                SELECT geoid, location_name, placekey, visit_count FROM (
                    SELECT v.geoid, v.placekey, v.visit_count, p.location_name,
                        ROW_NUMBER() OVER (PARTITION BY v.geoid ORDER BY v.visit_count DESC) AS rn
                    FROM cbg_poi_visits v
                    JOIN pois p ON v.placekey = p.placekey
                    WHERE p.top_category = ?
                ) t WHERE rn = 1
            ) AS tp
                ON c.geoid = tp.geoid
            """,
            conn,
            params=[matched_category, matched_category, matched_category],
        )

    # CBGs without a projected centroid can't be scored. Dropping them keeps NaN
    # distances out of the totals; otherwise they inflate total demand while
    # contributing zero predicted visits, which understates market share.
    total_cbgs = len(cbg_data)
    cbg_data = cbg_data.dropna(subset=["x_26919", "y_26919"]).copy()
    dropped_cbgs = total_cbgs - len(cbg_data)

    cbg_data["new_dist_m"] = (
        ((cbg_data["x_26919"] - new_x) ** 2 + (cbg_data["y_26919"] - new_y) ** 2) ** 0.5
    ).clip(lower=0.1)

    # Core logic correction: Separate probability calculations
    cbg_data["u_new"] = (float(store_area_sq_m) ** alpha) / (cbg_data["new_dist_m"] ** beta)
    
    # 1. Calculate capture probability and predicted visits for [YOUR NEW STORE]
    # Using .fillna(0) to prevent division by zero if both utilities are 0
    cbg_data["p_new"] = (cbg_data["u_new"] / (cbg_data["u_new"] + cbg_data["total_u_existing"])).fillna(0)
    cbg_data["predicted_visits"] = cbg_data["p_new"] * cbg_data["total_category_visits"]

    # 2. Calculate retention probability and remaining visits for [EXISTING COMPETITORS]
    # This represents the market share competitors manage to keep after your store opens
    cbg_data["p_existing"] = (cbg_data["total_u_existing"] / (cbg_data["u_new"] + cbg_data["total_u_existing"])).fillna(0)
    cbg_data["existing_competitor_visits"] = cbg_data["p_existing"] * cbg_data["total_category_visits"]

    total_predicted_visits = float(cbg_data["predicted_visits"].sum())
    total_demand = float(cbg_data["total_category_visits"].sum())
    total_existing_u = float(cbg_data["total_u_existing"].sum())
    market_share = float(total_predicted_visits / total_demand) if total_demand else 0.0
    runtime_seconds = time.perf_counter() - start

    # If there is no demand for this category anywhere in Worcester, the model has
    # nothing to allocate: predicted visits and market share are 0 by construction,
    # NOT because the location is bad. Flag this so callers don't misread the zeros.
    no_demand_data = total_demand <= 0

    # Preserve the full DataFrame for the downstream wrapper function
    details = cbg_data.copy()

    return {
        "matched_category": matched_category,
        "naics_code": naics_code,
        "alpha": alpha,
        "beta": beta,
        "correlation": correlation,
        "used_fallback": bool(used_fallback),
        "dropped_cbgs": int(dropped_cbgs),
        "no_demand_data": bool(no_demand_data),
        "total_demand": total_demand,
        "total_existing_u": total_existing_u,
        "total_predicted_visits": total_predicted_visits,
        "market_share": market_share,
        "runtime_seconds": runtime_seconds,
        "details": details, # Passing the full DataFrame to the wrapper
    }

def _build_notes(result: Dict[str, Any]) -> str:
    """Human-readable transparency note for the UI / chatbot.

    Surfaces the calibration state (especially the uncalibrated fallback) and the
    distance assumption, so a user is never shown confident numbers without context.
    """
    parts = []

    if result.get("no_demand_data"):
        parts.append(
            f"No demand data exists for '{result['matched_category']}' in Worcester, so the "
            f"model cannot estimate visits or market share for this category — the zero "
            f"values are a data gap, not a prediction. Try a category the dataset covers."
        )

    if result.get("used_fallback"):
        parts.append(
            f"No calibrated parameters were found for '{result['matched_category']}', "
            f"so default values (alpha=1.0, beta=1.0) were used — treat these results as rough."
        )
    elif not result.get("no_demand_data"):
        naics = result.get("naics_code")
        naics_str = f", NAICS {naics}" if naics and naics != "Unknown" else ""
        parts.append(
            f"Category matched: {result['matched_category']}{naics_str} "
            f"(alpha={result['alpha']:.3g}, beta={result['beta']:.3g})."
        )

    corr = result.get("correlation")
    if corr is not None:
        parts.append(f"Calibration correlation: {corr:.3g}.")

    parts.append("Distances are straight-line from neighborhood (CBG) centroids.")

    dropped = result.get("dropped_cbgs", 0)
    if dropped:
        parts.append(f"{dropped} CBG(s) without centroid coordinates were excluded.")

    return " ".join(parts)


def run_huff_model(
    candidate_lat: float,
    candidate_lon: float,
    business_category: str,
    floor_area: float,
    db_connection: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Flask/API-friendly wrapper expected by app.py.

    db_connection is accepted for compatibility with the provided starter app, but this
    engine uses get_connection() from db.py so Azure SQL is always used.
    """
    
    result = predict_site(candidate_lat, candidate_lon, business_category, floor_area)
    details = result["details"]
    
    # Load GeoJSON mapping for better CBG names
    geoid_to_name = _load_geoid_to_name_mapping() 
    
    total_market_demand = float(result["total_demand"])
    total_pred = float(result["total_predicted_visits"])

    # =========================================================
    # List 1: Top Customer Source Areas (Top CBGs)
    # Sorted descending by the predicted visits your store captures from them
    # =========================================================
    top_cbg_rows = details.sort_values("predicted_visits", ascending=False).head(10)
    top_cbgs_list = []
    
    for _, row in top_cbg_rows.iterrows():
        geoid_str = str(row['geoid'])
        # The name here should be the census tract/neighborhood name, NOT a store name
        cbg_name = geoid_to_name.get(geoid_str, f"Tract {geoid_str}")
        
        pred_vis = float(row.get('predicted_visits', 0.0))
        # This share indicates what percentage of YOUR store's total visits come from this specific CBG
        contribution_share = (pred_vis / total_pred) if total_pred else 0.0

        top_cbgs_list.append({
            "geoid": geoid_str,
            "name": cbg_name,
            "distance_miles": round(row["new_dist_m"] / 1609.34, 2),
            "demand_size": int(row["total_category_visits"]),
            "capture_probability": round(row["p_new"], 4), # Your store's capture probability for this area
            "predicted_visits": round(pred_vis, 2),
            "contribution_share": round(contribution_share, 4)
        })

    # =========================================================
    # List 2: Top Competitor Entities (Top Competitors)
    # Sorted descending by their historical visit counts to identify primary rivals
    # =========================================================
    # Filter out rows without actual competitor store data, then take the strongest ones
    comp_rows = details.dropna(subset=["top_location_name"]).sort_values("top_poi_visit_count", ascending=False).head(10)
    competitors_list = []
    
    for _, row in comp_rows.iterrows():
        hist_visits = float(row['top_poi_visit_count'])
        p_existing = float(row['p_existing']) # The probability that the competitor retains its customers
        
        # Core Calculation: Historical Visits * Retention Probability = Estimated Remaining Visits
        est_retained_visits = hist_visits * p_existing
        
        # True Competitor Market Share = Their remaining visits / Total market demand
        comp_market_share = (est_retained_visits / total_market_demand) if total_market_demand else 0.0

        competitors_list.append({
            "name": str(row['top_location_name']),
            "placekey": str(row['top_placekey']),
            "distance_miles": round(row["new_dist_m"] / 1609.34, 2),
            "historical_visits": int(hist_visits),                      # Original competitor foot traffic
            "predicted_visits": round(est_retained_visits, 2), # Traffic they keep after your store opens
            "attraction": round(p_existing, 4),                     # Their customer retention probability
            "market_share": round(comp_market_share, 6),                 # Their adjusted market share
            "my_store_attraction": round(row["p_new"], 4),
            "my_store_captured_visits": round(float(row.get('predicted_visits', 0.0)), 2)
        })

    # Clean up the large DataFrame to free memory
    # Compute a simple scalar representation of the candidate location's
    # attraction: the mean capture probability across all CBGs (p_new).
    # This is provided to the frontend as a single reference value for charts.
    try:
        candidate_attraction = float(details["p_new"].mean())
    except Exception:
        candidate_attraction = 0.0

    del result["details"]

    return {
        "matched_category": result["matched_category"],
        "naics_code": result["naics_code"],
        "alpha": result["alpha"],
        "beta": result["beta"],
        "correlation": result["correlation"],
        "used_fallback": result["used_fallback"],
        "no_demand_data": result["no_demand_data"],
        "total_demand": round(total_market_demand, 2),
        "predicted_visits": round(total_pred, 2),
        "market_share": round(result["market_share"], 6),
        "runtime_ms": round(result["runtime_seconds"] * 1000, 2),
        "notes": _build_notes(result),    # Assuming this function is defined elsewhere
        "competitors": competitors_list,  # Accurate list of rival stores
        "top_cbgs": top_cbgs_list,        # Accurate list of customer source neighborhoods
        "candidate_attraction": round(candidate_attraction, 6),
    }

if __name__ == "__main__":
    output = run_huff_model(42.27, -71.80, "Liquor Stores", 2500.0)
    print(output)
