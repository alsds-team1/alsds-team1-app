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
        cbg_data = pd.read_sql(
            """
            SELECT
                c.geoid,
                c.total_population,
                c.median_household_income,
                c.x_26919,
                c.y_26919,
                COALESCE(s.total_u_existing,      0) AS total_u_existing,
                COALESCE(d.total_category_visits, 0) AS total_category_visits
            FROM cbg_master AS c
            LEFT JOIN Competitor_Summary AS s
                ON c.geoid = s.geoid AND s.top_category = ?
            LEFT JOIN category_demand AS d
                ON c.geoid = d.geoid AND d.top_category = ?
            """,
            conn,
            params=[matched_category, matched_category],
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

    cbg_data["u_new"] = (float(store_area_sq_m) ** alpha) / (cbg_data["new_dist_m"] ** beta)
    cbg_data["p_new"] = cbg_data["u_new"] / (cbg_data["u_new"] + cbg_data["total_u_existing"])
    cbg_data["predicted_visits"] = cbg_data["p_new"] * cbg_data["total_category_visits"]

    total_predicted_visits = float(cbg_data["predicted_visits"].sum())
    total_demand = float(cbg_data["total_category_visits"].sum())
    total_existing_u = float(cbg_data["total_u_existing"].sum())
    market_share = float(total_predicted_visits / total_demand) if total_demand else 0.0
    runtime_seconds = time.perf_counter() - start

    # If there is no demand for this category anywhere in Worcester, the model has
    # nothing to allocate: predicted visits and market share are 0 by construction,
    # NOT because the location is bad. Flag this so callers don't misread the zeros.
    no_demand_data = total_demand <= 0

    details = cbg_data.sort_values("predicted_visits", ascending=False)

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
        "details": details,
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

    top_cbg_rows = details[
        ["geoid", "total_category_visits", "total_u_existing", "new_dist_m", "p_new", "predicted_visits"]
    ].head(10)

    # Convert to frontend-expected format for the competitors table
    # Load GeoJSON mapping for better CBG names
    geoid_to_name = _load_geoid_to_name_mapping()

    competitors_sample = []
    for idx, row in top_cbg_rows.iterrows():
        geoid = str(row['geoid'])

        cbg_name = None
        meta = geoid_to_name.get(geoid, {})

        # Prefer explicit tract and blk from GeoJSON if available
        tract_val = meta.get('tract')
        blk_val = meta.get('blk')
        if tract_val and blk_val:
            # Ensure block group is two digits with leading zero if needed
            try:
                blk_int = int(blk_val)
                blk_str = f"{blk_int:02d}"
            except Exception:
                blk_str = str(blk_val).zfill(2)

            # Tract in GeoJSON may be like '731204' or '7312' depending on source; normalize to last 4-5 digits
            tract = tract_val
            # If tract looks longer than 4-5, keep as-is; otherwise use it directly
            cbg_name = f"Tract {tract}, Block Group {blk_str}"

        elif geoid in geoid_to_name and meta.get('name'):
            # Fallback to NAMELSAD10 text
            cbg_name = meta.get('name')

        else:
            # Final fallback: parse from geoid string
            if len(geoid) >= 12:
                tract = geoid[5:10]
                block = geoid[10:12]
                # Zero-pad block group to 2 digits
                try:
                    block_int = int(block)
                    block_str = f"{block_int:02d}"
                except Exception:
                    block_str = block

                cbg_name = f"Tract {tract}, Block Group {block_str}"
            else:
                cbg_name = f"CBG {geoid}"

        competitors_sample.append({
            "name": cbg_name,
            "distance_miles": round(row["new_dist_m"] / 1609.34, 2),
            "size": int(row["total_category_visits"]),
            "attraction": round(row["p_new"], 4),
            # Also keep raw data for backend use
            "geoid": row["geoid"],
            "predicted_visits": round(row["predicted_visits"], 2),
        })

    return {
        "matched_category": result["matched_category"],
        "naics_code": result["naics_code"],
        "alpha": result["alpha"],
        "beta": result["beta"],
        "correlation": result["correlation"],
        "used_fallback": result["used_fallback"],
        "no_demand_data": result["no_demand_data"],
        "total_demand": round(result["total_demand"], 2),
        "predicted_visits": round(result["total_predicted_visits"], 2),
        "market_share": round(result["market_share"], 6),
        "runtime_ms": round(result["runtime_seconds"] * 1000, 2),
        "notes": _build_notes(result),
        "competitors": competitors_sample,
        "top_cbgs": competitors_sample,
    }


if __name__ == "__main__":
    output = run_huff_model(42.27, -71.80, "Liquor Stores", 2500.0)
    print(output)
