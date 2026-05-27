"""
Azure SQL migration script for the Urban AI / Huff Engine project.

What it does:
1. Reads the Worcester CSV files and GeoJSON file.
2. Converts the GeoJSON into a real SQL table: cbg_geojson.
3. Builds normalized tables and precomputed summary tables.
4. Uploads everything to Azure SQL using db.py / SQL_CONNECTION_STRING.

Run locally after setting SQL_CONNECTION_STRING:
    python migrate_to_azure_sql.py
"""

import json
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from pyproj import Transformer

from db import get_connection

BASE_DIR = Path(__file__).resolve().parent


def find_file(*names: str) -> Path:
    for name in names:
        path = BASE_DIR / name
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find any of these files: {names}")


CBGS_CSV = find_file("worcester_cbgs.csv", "worcester_cbgs(1).csv")
POIS_CSV = find_file("worcester_pois.csv", "worcester_pois(1).csv")
DISTANCE_CSV = find_file("worcester_cbg_poi_distance.csv", "worcester_cbg_poi_distance(1).csv")
VISITS_CSV = find_file("worcester_cbg_poi_visits.csv", "worcester_cbg_poi_visits(1).csv")
PARAMS_CSV = find_file("calibrated_parameters_filtered.csv", "calibrated_parameters_filtered(1).csv")
CBGS_GEOJSON = find_file("worcester_cbgs_map.geojson", "worcester_cbgs_map(1).geojson")


def load_cbg_geojson(path: Path) -> pd.DataFrame:
    """Store every GeoJSON feature as a row, including the full geometry JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for feature in data.get("features", []):
        props = feature.get("properties", {}) or {}
        geometry = feature.get("geometry", {}) or {}
        geoid = props.get("GEOID10")
        if geoid is None:
            continue
        rows.append(
            {
                "cbg": str(geoid),
                "statefp10": str(props.get("STATEFP10", "")),
                "countyfp10": str(props.get("COUNTYFP10", "")),
                "tractce10": str(props.get("TRACTCE10", "")),
                "blkgrpce10": str(props.get("BLKGRPCE10", "")),
                "namelsad10": str(props.get("NAMELSAD10", "")),
                "aland10": props.get("ALAND10"),
                "awater10": props.get("AWATER10"),
                "latitude": float(props.get("INTPTLAT10")) if props.get("INTPTLAT10") is not None else None,
                "longitude": float(props.get("INTPTLON10")) if props.get("INTPTLON10") is not None else None,
                "geometry_type": str(geometry.get("type", "")),
                "properties_json": json.dumps(props),
                "geometry_json": json.dumps(geometry),
                "feature_json": json.dumps(feature),
            }
        )
    return pd.DataFrame(rows)


def sanitize_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def insert_dataframe(conn, table_name: str, df: pd.DataFrame, columns: List[str]) -> None:
    placeholders = ", ".join(["?"] * len(columns))
    col_sql = ", ".join(f"[{c}]" for c in columns)
    sql = f"INSERT INTO {table_name} ({col_sql}) VALUES ({placeholders})"
    rows = [tuple(sanitize_value(v) for v in row) for row in df[columns].itertuples(index=False, name=None)]
    cursor = conn.cursor()
    cursor.fast_executemany = True
    if rows:
        cursor.executemany(sql, rows)
    conn.commit()


def execute_statements(conn, statements: Iterable[str]) -> None:
    cursor = conn.cursor()
    for statement in statements:
        cursor.execute(statement)
    conn.commit()


def build_tables() -> Tuple[pd.DataFrame, ...]:
    print("Loading CSV and GeoJSON source files...")
    cbgs = pd.read_csv(CBGS_CSV, dtype={"cbg": str})
    pois = pd.read_csv(POIS_CSV, dtype={"placekey": str, "poi_cbg": str, "naics_code": str})
    distances = pd.read_csv(DISTANCE_CSV, dtype={"placekey": str, "GEOID10": str})
    visits = pd.read_csv(VISITS_CSV, dtype={"visitor_home_cbg": str, "placekey": str})
    params = pd.read_csv(PARAMS_CSV, dtype={"NAICS code": str})
    cbg_geojson = load_cbg_geojson(CBGS_GEOJSON)

    cbg_master = cbgs.merge(cbg_geojson[["cbg", "latitude", "longitude"]], on="cbg", how="left")

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:26919", always_xy=True)
    x_coords, y_coords = transformer.transform(
        cbg_master["longitude"].astype(float).to_numpy(),
        cbg_master["latitude"].astype(float).to_numpy(),
    )
    cbg_master["x_26919"] = x_coords
    cbg_master["y_26919"] = y_coords

    category_parameters = params.rename(columns={"NAICS code": "naics_code"}).copy()
    category_parameters["naics_code"] = category_parameters["naics_code"].astype(str)

    pois_clean = pois[
        [
            "placekey",
            "location_name",
            "brands",
            "top_category",
            "sub_category",
            "category_tags",
            "naics_code",
            "latitude",
            "longitude",
            "poi_cbg",
            "street_address",
            "city",
            "region",
            "postal_code",
            "open_hours",
            "wkt_area_sq_meters",
            "polygon_wkt",
        ]
    ].copy()
    pois_clean["wkt_area_sq_meters"] = pois_clean["wkt_area_sq_meters"].fillna(1).clip(lower=1)

    distances_clean = distances.rename(columns={"GEOID10": "cbg"}).copy()
    distances_clean["distance_m"] = distances_clean["distance_m"].fillna(0.1).clip(lower=0.1)

    visits_clean = visits.copy()

    print("Precomputing competitor utility and demand summaries...")
    comp_base = (
        distances_clean.merge(
            pois_clean[["placekey", "top_category", "wkt_area_sq_meters"]], on="placekey", how="left"
        ).merge(category_parameters[["top_category", "alpha", "beta"]], on="top_category", how="inner")
    )
    comp_base["u_existing"] = np.power(comp_base["wkt_area_sq_meters"], comp_base["alpha"]) / np.power(
        comp_base["distance_m"], comp_base["beta"]
    )
    competitor_utility = (
        comp_base.groupby(["cbg", "top_category"], as_index=False)["u_existing"]
        .sum()
        .rename(columns={"u_existing": "total_u_existing"})
    )

    category_demand = (
        visits_clean.merge(pois_clean[["placekey", "top_category"]], on="placekey", how="left")
        .groupby(["visitor_home_cbg", "top_category"], as_index=False)["visit_count"]
        .sum()
        .rename(columns={"visitor_home_cbg": "cbg", "visit_count": "total_category_visits"})
    )

    migration_summary = pd.DataFrame(
        {
            "table_name": [
                "cbg_master",
                "cbg_geojson",
                "pois",
                "cbg_poi_distance",
                "cbg_poi_visits",
                "category_parameters",
                "competitor_utility",
                "category_demand",
            ],
            "row_count": [
                len(cbg_master),
                len(cbg_geojson),
                len(pois_clean),
                len(distances_clean),
                len(visits_clean),
                len(category_parameters),
                len(competitor_utility),
                len(category_demand),
            ],
        }
    )

    return (
        cbg_master,
        cbg_geojson,
        pois_clean,
        distances_clean,
        visits_clean,
        category_parameters,
        competitor_utility,
        category_demand,
        migration_summary,
    )


def migrate() -> None:
    (
        cbg_master,
        cbg_geojson,
        pois_clean,
        distances_clean,
        visits_clean,
        category_parameters,
        competitor_utility,
        category_demand,
        migration_summary,
    ) = build_tables()

    print("Connecting to Azure SQL...")
    with get_connection() as conn:
        print("Dropping old tables if they exist...")
        execute_statements(
            conn,
            [
                "DROP TABLE IF EXISTS migration_summary",
                "DROP TABLE IF EXISTS category_demand",
                "DROP TABLE IF EXISTS competitor_utility",
                "DROP TABLE IF EXISTS category_parameters",
                "DROP TABLE IF EXISTS cbg_poi_visits",
                "DROP TABLE IF EXISTS cbg_poi_distance",
                "DROP TABLE IF EXISTS pois",
                "DROP TABLE IF EXISTS cbg_geojson",
                "DROP TABLE IF EXISTS cbg_master",
            ],
        )

        print("Creating Azure SQL tables...")
        execute_statements(
            conn,
            [
                """
                CREATE TABLE cbg_master (
                    cbg NVARCHAR(20) NOT NULL PRIMARY KEY,
                    total_population INT NULL,
                    median_household_income FLOAT NULL,
                    median_age FLOAT NULL,
                    white_population FLOAT NULL,
                    black_population FLOAT NULL,
                    asian_population FLOAT NULL,
                    hispanic_population FLOAT NULL,
                    uni_degree FLOAT NULL,
                    income_q NVARCHAR(20) NULL,
                    education_q NVARCHAR(20) NULL,
                    age_q NVARCHAR(20) NULL,
                    latitude FLOAT NULL,
                    longitude FLOAT NULL,
                    x_26919 FLOAT NULL,
                    y_26919 FLOAT NULL
                )
                """,
                """
                CREATE TABLE cbg_geojson (
                    cbg NVARCHAR(20) NOT NULL PRIMARY KEY,
                    statefp10 NVARCHAR(10) NULL,
                    countyfp10 NVARCHAR(10) NULL,
                    tractce10 NVARCHAR(20) NULL,
                    blkgrpce10 NVARCHAR(10) NULL,
                    namelsad10 NVARCHAR(100) NULL,
                    aland10 FLOAT NULL,
                    awater10 FLOAT NULL,
                    latitude FLOAT NULL,
                    longitude FLOAT NULL,
                    geometry_type NVARCHAR(50) NULL,
                    properties_json NVARCHAR(MAX) NULL,
                    geometry_json NVARCHAR(MAX) NULL,
                    feature_json NVARCHAR(MAX) NULL
                )
                """,
                """
                CREATE TABLE pois (
                    placekey NVARCHAR(100) NOT NULL PRIMARY KEY,
                    location_name NVARCHAR(255) NULL,
                    brands NVARCHAR(MAX) NULL,
                    top_category NVARCHAR(255) NULL,
                    sub_category NVARCHAR(255) NULL,
                    category_tags NVARCHAR(MAX) NULL,
                    naics_code NVARCHAR(50) NULL,
                    latitude FLOAT NULL,
                    longitude FLOAT NULL,
                    poi_cbg NVARCHAR(20) NULL,
                    street_address NVARCHAR(255) NULL,
                    city NVARCHAR(100) NULL,
                    region NVARCHAR(20) NULL,
                    postal_code NVARCHAR(20) NULL,
                    open_hours NVARCHAR(MAX) NULL,
                    wkt_area_sq_meters FLOAT NULL,
                    polygon_wkt NVARCHAR(MAX) NULL
                )
                """,
                """
                CREATE TABLE cbg_poi_distance (
                    placekey NVARCHAR(100) NOT NULL,
                    cbg NVARCHAR(20) NOT NULL,
                    distance_m FLOAT NULL
                )
                """,
                """
                CREATE TABLE cbg_poi_visits (
                    visitor_home_cbg NVARCHAR(20) NOT NULL,
                    placekey NVARCHAR(100) NOT NULL,
                    visit_count FLOAT NULL
                )
                """,
                """
                CREATE TABLE category_parameters (
                    top_category NVARCHAR(255) NOT NULL,
                    naics_code NVARCHAR(50) NULL,
                    alpha FLOAT NULL,
                    beta FLOAT NULL,
                    correlation FLOAT NULL
                )
                """,
                """
                CREATE TABLE competitor_utility (
                    cbg NVARCHAR(20) NOT NULL,
                    top_category NVARCHAR(255) NOT NULL,
                    total_u_existing FLOAT NULL
                )
                """,
                """
                CREATE TABLE category_demand (
                    cbg NVARCHAR(20) NOT NULL,
                    top_category NVARCHAR(255) NOT NULL,
                    total_category_visits FLOAT NULL
                )
                """,
                """
                CREATE TABLE migration_summary (
                    table_name NVARCHAR(100) NOT NULL,
                    row_count INT NOT NULL
                )
                """,
            ],
        )

        print("Inserting rows...")
        insert_dataframe(conn, "cbg_master", cbg_master, list(cbg_master.columns))
        insert_dataframe(conn, "cbg_geojson", cbg_geojson, list(cbg_geojson.columns))
        insert_dataframe(conn, "pois", pois_clean, list(pois_clean.columns))
        insert_dataframe(conn, "cbg_poi_distance", distances_clean, list(distances_clean.columns))
        insert_dataframe(conn, "cbg_poi_visits", visits_clean, list(visits_clean.columns))
        insert_dataframe(conn, "category_parameters", category_parameters, list(category_parameters.columns))
        insert_dataframe(conn, "competitor_utility", competitor_utility, list(competitor_utility.columns))
        insert_dataframe(conn, "category_demand", category_demand, list(category_demand.columns))
        insert_dataframe(conn, "migration_summary", migration_summary, list(migration_summary.columns))

        print("Creating indexes...")
        execute_statements(
            conn,
            [
                "CREATE INDEX idx_pois_category ON pois(top_category)",
                "CREATE INDEX idx_distance_cbg_placekey ON cbg_poi_distance(cbg, placekey)",
                "CREATE INDEX idx_visits_home_placekey ON cbg_poi_visits(visitor_home_cbg, placekey)",
                "CREATE INDEX idx_params_category ON category_parameters(top_category)",
                "CREATE INDEX idx_params_naics ON category_parameters(naics_code)",
                "CREATE INDEX idx_utility_cbg_category ON competitor_utility(cbg, top_category)",
                "CREATE INDEX idx_demand_cbg_category ON category_demand(cbg, top_category)",
            ],
        )

    print("\nSUCCESS: Azure SQL migration completed.")
    print(migration_summary.to_string(index=False))


if __name__ == "__main__":
    migrate()
