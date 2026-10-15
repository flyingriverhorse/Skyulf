-- Source tables for the NYC taxi example projects, built from samples.nyctaxi.trips.
-- Run in the Databricks SQL editor (or a notebook) with two parameters:
--   catalog: the example's "catalog" answer, for example workspace
--   schema:  the example's "schema" answer, for example taxi_fare
-- Create-only: a statement fails when its table exists, so nothing is overwritten.

CREATE SCHEMA IF NOT EXISTS IDENTIFIER(:catalog || '.' || :schema);

-- One stable trip_id per trip, shared by training, scoring and labels.
CREATE TABLE IDENTIFIER(:catalog || '.' || :schema || '.trips_base') AS
SELECT
  CAST(row_number() OVER (
    ORDER BY tpep_pickup_datetime, tpep_dropoff_datetime, trip_distance, fare_amount,
      pickup_zip, dropoff_zip
  ) AS BIGINT) AS trip_id,
  tpep_pickup_datetime AS pickup_ts,
  tpep_dropoff_datetime AS dropoff_ts,
  CAST(hour(tpep_pickup_datetime) AS DOUBLE) AS pickup_hour,
  CAST(dayofweek(tpep_pickup_datetime) AS DOUBLE) AS pickup_dow,
  trip_distance,
  CAST(
    (unix_timestamp(tpep_dropoff_datetime) - unix_timestamp(tpep_pickup_datetime)) / 60.0
    AS DOUBLE
  ) AS duration_min,
  CAST(pickup_zip AS STRING) AS pickup_zip,
  CAST(dropoff_zip AS STRING) AS dropoff_zip,
  fare_amount,
  CAST(CASE WHEN fare_amount > 15 THEN 1 ELSE 0 END AS INT) AS high_fare,
  CAST(CASE WHEN trip_distance > 5 THEN 1 ELSE 0 END AS INT) AS long_trip,
  month(tpep_pickup_datetime) AS pickup_month
FROM samples.nyctaxi.trips;

-- January trains, scores and labels the first champion. Change data feed lets
-- training and scoring read only rows added later.
CREATE TABLE IDENTIFIER(:catalog || '.' || :schema || '.trips_train')
TBLPROPERTIES (delta.enableChangeDataFeed = true) AS
SELECT trip_id, pickup_ts, pickup_hour, pickup_dow, trip_distance, duration_min, pickup_zip,
  dropoff_zip, fare_amount, high_fare, long_trip
FROM IDENTIFIER(:catalog || '.' || :schema || '.trips_base')
WHERE pickup_month = 1;

CREATE TABLE IDENTIFIER(:catalog || '.' || :schema || '.trips_score')
TBLPROPERTIES (delta.enableChangeDataFeed = true) AS
SELECT trip_id, pickup_ts, pickup_hour, pickup_dow, trip_distance, duration_min, pickup_zip,
  dropoff_zip, fare_amount, high_fare, long_trip
FROM IDENTIFIER(:catalog || '.' || :schema || '.trips_base')
WHERE pickup_month = 1;

-- Actual outcomes for monitoring; a fare is known when the trip ends.
CREATE TABLE IDENTIFIER(:catalog || '.' || :schema || '.trips_labels')
TBLPROPERTIES (delta.enableChangeDataFeed = true) AS
SELECT trip_id, fare_amount, high_fare, long_trip, dropoff_ts AS result_available_at
FROM IDENTIFIER(:catalog || '.' || :schema || '.trips_base')
WHERE pickup_month = 1;

-- Latest-value zone features for taxi_fare_online: one row per pickup zone, a
-- declared primary key, change data feed and a DOUBLE epoch freshness field.
-- zone_updated_at records when the features were computed; refresh the table
-- (and its online publication) within the project's max_age_seconds.
CREATE TABLE IDENTIFIER(:catalog || '.' || :schema || '.zone_features') (
  pickup_zip STRING NOT NULL,
  zone_avg_distance DOUBLE,
  zone_trip_count DOUBLE,
  zone_updated_at DOUBLE,
  CONSTRAINT zone_features_pk PRIMARY KEY (pickup_zip)
)
TBLPROPERTIES (delta.enableChangeDataFeed = true);

INSERT INTO IDENTIFIER(:catalog || '.' || :schema || '.zone_features')
SELECT pickup_zip, avg(trip_distance), CAST(count(*) AS DOUBLE),
  CAST(unix_timestamp(current_timestamp()) AS DOUBLE)
FROM IDENTIFIER(:catalog || '.' || :schema || '.trips_base')
WHERE pickup_month = 1 AND pickup_zip IS NOT NULL
GROUP BY pickup_zip;

-- Retraining scenario, run later on its own: uncomment to add February.
-- INSERT INTO IDENTIFIER(:catalog || '.' || :schema || '.trips_train')
-- SELECT trip_id, pickup_ts, pickup_hour, pickup_dow, trip_distance, duration_min, pickup_zip,
--   dropoff_zip, fare_amount, high_fare, long_trip
-- FROM IDENTIFIER(:catalog || '.' || :schema || '.trips_base') WHERE pickup_month = 2;
-- INSERT INTO IDENTIFIER(:catalog || '.' || :schema || '.trips_score')
-- SELECT trip_id, pickup_ts, pickup_hour, pickup_dow, trip_distance, duration_min, pickup_zip,
--   dropoff_zip, fare_amount, high_fare, long_trip
-- FROM IDENTIFIER(:catalog || '.' || :schema || '.trips_base') WHERE pickup_month = 2;
-- INSERT INTO IDENTIFIER(:catalog || '.' || :schema || '.trips_labels')
-- SELECT trip_id, fare_amount, high_fare, long_trip, dropoff_ts AS result_available_at
-- FROM IDENTIFIER(:catalog || '.' || :schema || '.trips_base') WHERE pickup_month = 2;
