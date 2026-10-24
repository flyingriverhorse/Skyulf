-- Demo data for examples/lakeflow/bronze_observations. Run on a SQL warehouse.
CREATE SCHEMA IF NOT EXISTS workspace.lakeflow_demo;

-- 1000 entities x 3 month ends; about 40% have a birth date (a target), and some
-- labeled rows have no region_code, so the code arrives as floats in training.
CREATE OR REPLACE TABLE workspace.lakeflow_demo.raw_events AS
SELECT
  CAST(e.id AS BIGINT) AS entity_id,
  d.observed_date,
  CASE
    WHEN pmod(e.id * 7 + month(d.observed_date), 5) < 2
    THEN date_add(DATE'1960-01-01', CAST(pmod(e.id * 37, 12000) AS INT))
  END AS birth_date,
  CASE
    WHEN pmod(e.id, 10) IN (0, 3) THEN NULL
    ELSE element_at(array(12, 34, 81), CAST(pmod(e.id, 3) AS INT) + 1)
  END AS region_code,
  CAST(10 + pmod(e.id * 13, 490) AS INT) AS employee_count,
  CASE WHEN pmod(e.id, 17) = 0 THEN NULL ELSE CAST(pmod(e.id * 101, 10000) AS DOUBLE) / 10 END
    AS revenue
FROM range(1, 1001) AS e(id)
CROSS JOIN (
  SELECT explode(array(DATE'2026-07-31', DATE'2026-08-31', DATE'2026-09-30')) AS observed_date
) AS d;

-- The bronze source the project reads: a Lakeflow streaming table.
CREATE OR REFRESH STREAMING TABLE workspace.lakeflow_demo.bronze_observations
AS SELECT * FROM STREAM workspace.lakeflow_demo.raw_events;
