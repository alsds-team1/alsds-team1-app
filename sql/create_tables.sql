-- SQLite Schema Export

-- Table: cbg_master
-- CREATE TABLE for cbg_master (SQL Server compatible)
CREATE TABLE IF NOT EXISTS [dbo].[cbg_master] (
  [geoid] NVARCHAR(100),
  [total_population] INT,
  [median_household_income] FLOAT,
  [median_age] FLOAT,
  [white_population] FLOAT,
  [black_population] FLOAT,
  [asian_population] FLOAT,
  [hispanic_population] FLOAT,
  [uni_degree] FLOAT,
  [income_q] NVARCHAR(50),
  [education_q] NVARCHAR(50),
  [age_q] NVARCHAR(50),
  [latitude] FLOAT,
  [longitude] FLOAT,
  [x_26919] FLOAT,
  [y_26919] FLOAT
);

-- CREATE TABLE for pois
CREATE TABLE IF NOT EXISTS [dbo].[pois] (
  [placekey] NVARCHAR(200),
  [location_name] NVARCHAR(500),
  [top_category] NVARCHAR(200),
  [sub_category] NVARCHAR(200),
  [naics_code] NVARCHAR(50),
  [latitude] FLOAT,
  [longitude] FLOAT,
  [poi_cbg] NVARCHAR(100),
  [wkt_area_sq_meters] INT
);

-- CREATE TABLE for cbg_poi_distance
CREATE TABLE IF NOT EXISTS [dbo].[cbg_poi_distance] (
  [placekey] NVARCHAR(200),
  [geoid] NVARCHAR(100),
  [distance_m] FLOAT
);

-- CREATE TABLE for cbg_poi_visits
CREATE TABLE IF NOT EXISTS [dbo].[cbg_poi_visits] (
  [geoid] NVARCHAR(100),
  [placekey] NVARCHAR(200),
  [visit_count] INT
);

-- CREATE TABLE for category_parameters
CREATE TABLE IF NOT EXISTS [dbo].[category_parameters] (
  [top_category] NVARCHAR(200),
  [naics_code] NVARCHAR(50),
  [alpha] FLOAT,
  [beta] FLOAT,
  [correlation] FLOAT
);

-- CREATE TABLE for Competitor_Summary
CREATE TABLE IF NOT EXISTS [dbo].[Competitor_Summary] (
  [geoid] NVARCHAR(100),
  [top_category] NVARCHAR(200),
  [total_u_existing] FLOAT
);

-- CREATE TABLE for category_demand
CREATE TABLE IF NOT EXISTS [dbo].[category_demand] (
  [geoid] NVARCHAR(100),
  [top_category] NVARCHAR(200),
  [total_category_visits] INT
);

-- CREATE TABLE for migration_summary
CREATE TABLE IF NOT EXISTS [dbo].[migration_summary] (
  [table_name] NVARCHAR(100) NOT NULL,
  [row_count] INT NOT NULL
);

