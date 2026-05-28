"""
Azure SQL Huff Engine

This version removes the local SQLite dependency and queries Azure SQL through db.py.
It exposes run_huff_model(), which is the function app.py already expects.
"""

import difflib
import time
from typing import Any, Dict, Optional

import pandas as pd
from pyproj import Transformer

from db import get_connection


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

    cbg_data["new_dist_m"] = (
        ((cbg_data["x_26919"] - new_x) ** 2 + (cbg_data["y_26919"] - new_y) ** 2) ** 0.5
    ).clip(lower=0.1)

    cbg_data["u_new"] = (float(store_area_sq_m) ** alpha) / (cbg_data["new_dist_m"] ** beta)
    cbg_data["p_new"] = cbg_data["u_new"] / (cbg_data["u_new"] + cbg_data["total_u_existing"])
    cbg_data["predicted_visits"] = cbg_data["p_new"] * cbg_data["total_category_visits"]

    total_predicted_visits = float(cbg_data["predicted_visits"].sum())
    total_demand = float(cbg_data["total_category_visits"].sum())
    market_share = float(total_predicted_visits / total_demand) if total_demand else 0.0
    runtime_seconds = time.perf_counter() - start

    details = cbg_data.sort_values("predicted_visits", ascending=False)

    return {
        "matched_category": matched_category,
        "alpha": alpha,
        "beta": beta,
        "correlation": correlation,
        "used_fallback": bool(used_fallback),
        "total_predicted_visits": total_predicted_visits,
        "market_share": market_share,
        "runtime_seconds": runtime_seconds,
        "details": details,
    }


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
        ["cbg", "total_category_visits", "total_u_existing", "new_dist_m", "p_new", "predicted_visits"]
    ].head(10)

    # Give the LLM/frontend a small sample instead of the full DataFrame.
    competitors_sample = top_cbg_rows.to_dict(orient="records")

    return {
        "matched_category": result["matched_category"],
        "alpha": result["alpha"],
        "beta": result["beta"],
        "correlation": result["correlation"],
        "used_fallback": result["used_fallback"],
        "predicted_visits": round(result["total_predicted_visits"], 2),
        "market_share": round(result["market_share"], 6),
        "runtime_ms": round(result["runtime_seconds"] * 1000, 2),
        "competitors": competitors_sample,
        "top_cbgs": competitors_sample,
    }


if __name__ == "__main__":
    output = run_huff_model(42.27, -71.80, "Liquor Stores", 2500.0)
    print(output)
