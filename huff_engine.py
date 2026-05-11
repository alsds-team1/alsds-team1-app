"""
Module 5 - Huff Engine v3 (database-optimized)

This is the LIVE inference engine. It is intentionally thin:
    - It does NOT read CSVs or GeoJSON. Ever.
    - It reads from team1.db using parameterized SQL ('?' placeholders).
    - All "heavy lifting" (coordinate projection of CBGs, summing competitor
      utilities) was amortized into migration_v3.py at build time.
    - At inference, we project the ONE candidate site's coordinates, fetch the
      pre-computed Competitor_Summary, add the candidate's utility, and divide.

Run:
    python huff_engine_v3.py
"""

import difflib
import sqlite3
import time
from pathlib import Path

import pandas as pd
from pyproj import Transformer

DB_PATH = Path(__file__).resolve().parent / "Data" / "team1.db"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def prompt_with_default(prompt_text, default, cast_func=str):
    """CLI helper: read input with a fallback default if the user just hits Enter."""
    raw = input(f"{prompt_text} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return cast_func(raw)
    except ValueError:
        print(f"Invalid input '{raw}'. Using default {default}.")
        return default


def get_category_parameters(conn, user_category):
    """Resolve a user's category string to a calibrated (alpha, beta) row.

    Resolution order (most specific first):
        1. Exact match on top_category
        2. Exact match on naics_code
        3. Substring match on top_category
        4. Fuzzy match (difflib, cutoff 0.55)
        5. Neutral fallback alpha=1.0, beta=1.0  (engine flags this as a fallback)
    """
    params = pd.read_sql_query(
        "SELECT top_category, naics_code, alpha, beta, correlation "
        "FROM category_parameters",
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

    matches = difflib.get_close_matches(
        query, params["top_category"].tolist(), n=1, cutoff=0.55
    )
    if matches:
        return params[params["top_category"] == matches[0]].iloc[0], False

    fallback = pd.Series({
        "top_category": query,
        "naics_code": "Unknown",
        "alpha": 1.0,
        "beta": 1.0,
        "correlation": None,
    })
    return fallback, True


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def run_huff_model(
    candidate_lat,
    candidate_lon,
    business_category,
    floor_area,
    db_connection=None,
):
    """
    Predicts expected category visits for a candidate site using a SQLite backend.
    Matches the function signature and return structure of the ALSDS baseline.
    """
    if not DB_PATH.exists():
        raise FileNotFoundError(f"{DB_PATH.name} not found. Run migration_v3.py first.")

    start_time = time.perf_counter()

    # 1. Establish database connection
    # Use provided connection if available, otherwise create a new one
    conn = db_connection if db_connection else sqlite3.connect(DB_PATH)
    
    try:
        # 2. Retrieve calibrated category parameters
        # Maps the business_category input to a specific NAICS/Top Category
        params, used_fallback = get_category_parameters(conn, business_category)
        matched_category = str(params["top_category"])
        alpha = float(params["alpha"])
        beta = float(params["beta"])
        correlation = params.get("correlation", None)

        # 3. Project candidate coordinates to EPSG:26919 (UTM Zone 19N)
        # This allows for accurate distance calculations in meters
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:26919", always_xy=True)
        new_x, new_y = transformer.transform(float(candidate_lon), float(candidate_lat))

        # 4. Single parameterized SQL query
        # Performs LEFT JOINs to fetch pre-computed competitor utility and historical demand.
        # COALESCE handles cases with zero historical visits to ensure a complete geographic report.
        cbg_data = pd.read_sql_query(
            """
            SELECT
                c.geoid,
                c.x_26919,
                c.y_26919,
                COALESCE(s.total_u_existing, 0) AS total_u_existing,
                COALESCE(d.total_category_visits, 0) AS total_category_visits
            FROM cbg_master AS c
            LEFT JOIN Competitor_Summary AS s
                ON c.geoid = s.geoid AND s.top_category = ?
            LEFT JOIN category_demand AS d
                ON c.geoid = d.geoid AND d.top_category = ?
            """,
            conn,
            params=(matched_category, matched_category),
        )
    finally:
        # Only close the connection if we opened it locally
        if not db_connection:
            conn.close()

    # 5. Vectorized Huff Math (Using Pandas for high performance)
    # Calculate Euclidean distance between each CBG centroid and the candidate
    cbg_data["new_dist_m"] = (
        ((cbg_data["x_26919"] - new_x) ** 2 + (cbg_data["y_26919"] - new_y) ** 2) ** 0.5
    ).clip(lower=100) # Prevents division instability for very short distances

    # Calculate candidate utility: U = Area^alpha / Distance^beta
    cbg_data["u_new"] = (float(floor_area) ** alpha) / (cbg_data["new_dist_m"] ** beta)

    # Calculate Huff Probability: P = U_new / (U_new + Sum_U_competitors)
    cbg_data["p_new"] = cbg_data["u_new"] / (cbg_data["u_new"] + cbg_data["total_u_existing"])

    # Calculate expected visits per CBG
    cbg_data["predicted_visits"] = cbg_data["p_new"] * cbg_data["total_category_visits"]

    # 6. Final Aggregation
    total_predicted_visits = float(cbg_data["predicted_visits"].sum())
    total_market_visits = float(cbg_data["total_category_visits"].sum())
    
    # Calculate market share proxy
    market_share = (total_predicted_visits / total_market_visits) if total_market_visits > 0 else 0.0
    
    runtime_ms = round((time.perf_counter() - start_time) * 1000, 2)

    # 7. Return structured dictionary compatible with the Dashboard/Chatbot
    return {
        "predicted_visits": round(total_predicted_visits, 2),
        "market_share": round(market_share, 6),
        "competitors": [], # Competitor list can be populated via a separate query if needed
        "runtime_ms": runtime_ms,
        "notes": (
            f"Optimized SQLite version. Matched Category: {matched_category}. "
            f"Fallback applied: {used_fallback}. Correlation: {correlation}."
        ),
        "inputs": {
            "candidate_lat": candidate_lat,
            "candidate_lon": candidate_lon,
            "business_category": business_category,
            "floor_area": floor_area,
        }
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
# def main():
#     print("=" * 60)
#     print("Urban AI Assistant - Huff Engine v3 (SQLite-backed)")
#     print("=" * 60)
#     lat = prompt_with_default("Enter new store latitude", 42.27, float)
#     lon = prompt_with_default("Enter new store longitude", -71.80, float)
#     category = prompt_with_default(
#         "Enter top_category or NAICS code", "Beer, Wine, and Liquor Stores", str
#     )
#     area = prompt_with_default("Enter store size in square meters", 2500.0, float)

#     result = run_huff_model(lat, lon, category, area)

#     print(result)


# if __name__ == "__main__":
#     main()
