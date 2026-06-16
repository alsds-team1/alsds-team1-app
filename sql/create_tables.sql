-- Azure SQL Schema (compatible with SQL Server 2016+)

-- Table: cbg_master
IF OBJECT_ID('[dbo].[cbg_master]', 'U') IS NULL
CREATE TABLE [dbo].[cbg_master] (
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
GO

-- Table: pois
IF OBJECT_ID('[dbo].[pois]', 'U') IS NULL
CREATE TABLE [dbo].[pois] (
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
GO

-- Table: cbg_poi_distance
IF OBJECT_ID('[dbo].[cbg_poi_distance]', 'U') IS NULL
CREATE TABLE [dbo].[cbg_poi_distance] (
  [placekey] NVARCHAR(200),
  [geoid] NVARCHAR(100),
  [distance_m] FLOAT
);
GO

-- Table: cbg_poi_visits
IF OBJECT_ID('[dbo].[cbg_poi_visits]', 'U') IS NULL
CREATE TABLE [dbo].[cbg_poi_visits] (
  [geoid] NVARCHAR(100),
  [placekey] NVARCHAR(200),
  [visit_count] INT
);
GO

-- Table: category_parameters
IF OBJECT_ID('[dbo].[category_parameters]', 'U') IS NULL
CREATE TABLE [dbo].[category_parameters] (
  [top_category] NVARCHAR(200),
  [naics_code] NVARCHAR(50),
  [alpha] FLOAT,
  [beta] FLOAT,
  [correlation] FLOAT
);
GO

-- Table: Competitor_Summary
IF OBJECT_ID('[dbo].[Competitor_Summary]', 'U') IS NULL
CREATE TABLE [dbo].[Competitor_Summary] (
  [geoid] NVARCHAR(100),
  [top_category] NVARCHAR(200),
  [total_u_existing] FLOAT
);
GO

-- Table: category_demand
IF OBJECT_ID('[dbo].[category_demand]', 'U') IS NULL
CREATE TABLE [dbo].[category_demand] (
  [geoid] NVARCHAR(100),
  [top_category] NVARCHAR(200),
  [total_category_visits] INT
);
GO

-- Table: migration_summary
IF OBJECT_ID('[dbo].[migration_summary]', 'U') IS NULL
CREATE TABLE [dbo].[migration_summary] (
  [table_name] NVARCHAR(100) NOT NULL,
  [row_count] INT NOT NULL
);
GO

IF OBJECT_ID('[dbo].[cbg_geometries]', 'U') IS NULL
CREATE TABLE [dbo].[cbg_geometries] (
    [geoid] NVARCHAR(100),
    [geometry] NVARCHAR(MAX)
);
GO