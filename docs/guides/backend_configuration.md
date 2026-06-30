# Backend Configuration Reference

All settings are loaded from the `.env` file in the project root (or environment variables).
They are case-sensitive. Start by copying the example:

```bash
cp .env.example .env
```

---

## Application (Core)

| Variable | Default | Description |
|---|---|---|
| `FASTAPI_ENV` | `development` | Environment mode: `development`, `production`, or `testing` |
| `APP_NAME` | `Skyulf` | Application name shown in API docs |
| `APP_VERSION` | *(set by code)* | Application version — do not override |
| `DEBUG` | `false` | Enable debug mode (enables API docs, verbose logging, auto-reload) |
| `HOST` | `127.0.0.1` | Bind address (`0.0.0.0` to expose on all interfaces) |
| `PORT` | `8000` | Bind port |
| `WORKERS` | `1` | Number of uvicorn worker processes (production only) |
| `SENTRY_DSN` | *(unset)* | Sentry error tracking DSN — leave unset to disable |

> **Note:** `FASTAPI_ENV=development` is the default. Production mode is enabled with `FASTAPI_ENV=production`.

---

## Security & Auth

| Variable | Default | Description |
|---|---|---|
| `SECRET_KEY` | *(random per restart)* | **Required in production.** Used for JWT signing. Set a stable value or tokens break on restart and across workers. Generate: `python -c "import secrets; print(secrets.token_urlsafe(64))"` |
| `ALGORITHM` | `HS256` | JWT signing algorithm |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `480` | JWT access token lifetime (8 hours) |
| `REFRESH_TOKEN_EXPIRE_DAYS` | `7` | JWT refresh token lifetime |
| `MAX_LOGIN_ATTEMPTS` | `5` | Failed login attempts before lockout |
| `ACCOUNT_LOCKOUT_DURATION_MINUTES` | `30` | Account lockout duration |
| `AUTH_FALLBACK_ENABLED` | `false` | Enable simple username/password fallback auth (dev only) |
| `AUTH_FALLBACK_USERNAME` | `admin` | Fallback auth username (only used when `AUTH_FALLBACK_ENABLED=true`) |
| `AUTH_FALLBACK_PASSWORD` | *(unset)* | **Required if fallback auth is enabled.** Must be set explicitly — no default. |
| `ALLOW_USER_REGISTRATION` | `true` | Allow new user registration via API |
| `CORS_ORIGINS` | `http://localhost:3000,http://localhost:8080` | Comma-separated list of allowed CORS origins |
| `ALLOWED_HOSTS` | `localhost,127.0.0.1` | Comma-separated list of trusted host headers |

> **Production checklist:**
> - Set `SECRET_KEY` to a strong random value
> - Set `FASTAPI_ENV=production`
> - Set `CORS_ORIGINS` to your actual frontend domain
> - Set `AUTH_FALLBACK_ENABLED=false` (or omit — it defaults to false)

---

## Database

| Variable | Default | Description |
|---|---|---|
| `DB_TYPE` | `sqlite` | Database backend: `sqlite` or `postgres` |
| `DATABASE_URL` | *(auto-constructed)* | Full SQLAlchemy async URL — set directly to override component fields |
| `DB_PATH` | `mlops_database.db` | SQLite database file path (relative to project root) |
| `DB_ECHO` | `false` | Log all SQL statements to console (verbose, use for debugging only) |
| `DB_POOL_SIZE` | `10` | PostgreSQL connection pool size |
| `DB_MAX_OVERFLOW` | `20` | Maximum PostgreSQL overflow connections |

**PostgreSQL component fields** (used when `DB_TYPE=postgres` and `DATABASE_URL` is not set):

| Variable | Description |
|---|---|
| `DB_USER` | Database username |
| `DB_PASSWORD` | Database password |
| `DB_HOST` | Database host |
| `DB_PORT` | Database port (default: 5432) |
| `DB_NAME` | Database name |
| `DB_SSLMODE` | SSL mode (`require`, `disable`, etc.) |
| `DB_SSLROOTCERT` | Path to SSL root certificate |

---

## Background Jobs (Celery)

| Variable | Default | Description |
|---|---|---|
| `USE_CELERY` | `false` | Enable Celery for background training jobs. When `false`, jobs run in FastAPI BackgroundTasks (no Redis needed — recommended for local dev) |
| `CELERY_BROKER_URL` | `redis://127.0.0.1:6379/0` | Redis URL for the Celery message broker |
| `CELERY_RESULT_BACKEND` | `redis://127.0.0.1:6379/0` | Redis URL for storing Celery task results |
| `CELERY_TASK_DEFAULT_QUEUE` | `mlops-training` | Default Celery queue name |
| `TUNING_N_JOBS` | `1` | Parallelism for hyperparameter search (1 = sequential, safe in FastAPI; -1 = all CPUs, safe in Celery workers only) |
| `TUNING_PARALLEL_BACKEND` | *(unset)* | joblib backend override: `threading` (safe in FastAPI), `loky` (safe in Celery only) |
| `ERROR_LOG_RETENTION_DAYS` | `30` | Days to retain error events in the DB before auto-deletion |

---

## File Uploads

| Variable | Default | Description |
|---|---|---|
| `UPLOAD_DIR` | `uploads/data` | Directory for uploaded dataset files |
| `MAX_UPLOAD_SIZE` | `10737418240` (10 GB) | Maximum upload file size in bytes |
| `ALLOWED_EXTENSIONS` | `.csv,.xlsx,.xls,.parquet,.json,.txt,.pkl,.feather,.h5,.hdf5` | Comma-separated list of allowed file extensions |
| `TRAINING_ARTIFACT_DIR` | `uploads/models` | Directory for trained model artifacts |
| `TEMP_DIR` | `temp/processing` | Temporary processing directory |
| `EXPORT_DIR` | `exports/data` | Data export directory |
| `MODELS_DIR` | `uploads/models` | Model storage directory |

---

## Cache / Redis

| Variable | Default | Description |
|---|---|---|
| `CACHE_TYPE` | `filesystem` | Cache backend: `filesystem` or `redis` |
| `CACHE_DEFAULT_TIMEOUT` | `3600` | Default cache TTL in seconds (1 hour) |
| `CACHE_TTL` | `300` | Short-lived cache TTL (5 minutes) |
| `REDIS_URL` | *(unset)* | Redis URL for cache backend (e.g. `redis://127.0.0.1:6379/0`) |

---

## Logging

| Variable | Default | Description |
|---|---|---|
| `LOG_LEVEL` | `INFO` | Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `LOG_FILE` | `logs/fastapi_app.log` | Log file path |
| `LOG_MAX_SIZE` | `52428800` (50 MB) | Maximum log file size before rotation |
| `LOG_BACKUP_COUNT` | `5` | Number of rotated log files to retain |
| `LOG_ROTATION_TYPE` | `size` | Log rotation strategy: `size` or `time` |
| `LOG_ROTATION_WHEN` | `midnight` | When to rotate (used with `LOG_ROTATION_TYPE=time`) |
| `LOG_ROTATION_INTERVAL` | `1` | Rotation interval (used with `LOG_ROTATION_TYPE=time`) |

---

## AWS / S3 (Optional)

| Variable | Default | Description |
|---|---|---|
| `AWS_ACCESS_KEY_ID` | *(unset)* | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | *(unset)* | AWS secret key |
| `AWS_SESSION_TOKEN` | *(unset)* | AWS session token (for temporary credentials) |
| `AWS_DEFAULT_REGION` | `us-east-1` | AWS region |
| `AWS_ENDPOINT_URL` | *(unset)* | Custom S3 endpoint (e.g. MinIO) |
| `AWS_BUCKET_NAME` | *(unset)* | S3 bucket for uploads |
| `S3_ARTIFACT_BUCKET` | *(unset)* | S3 bucket for ML artifacts |

---

## Snowflake (Optional)

| Variable | Default | Description |
|---|---|---|
| `FEATURE_SNOWFLAKE` | `false` | Enable Snowflake data source integration |
| `SNOWFLAKE_ACCOUNT` | *(unset)* | Snowflake account identifier |
| `SNOWFLAKE_USER` | *(unset)* | Snowflake username |
| `SNOWFLAKE_PASSWORD` | *(unset)* | Snowflake password |
| `SNOWFLAKE_WAREHOUSE` | *(unset)* | Snowflake warehouse |
| `SNOWFLAKE_DATABASE` | *(unset)* | Snowflake database |
| `SNOWFLAKE_SCHEMA` | *(unset)* | Snowflake schema |
| `SNOWFLAKE_ROLE` | *(unset)* | Snowflake role |

---

## LLM Integrations (Optional)

| Variable | Default | Description |
|---|---|---|
| `DEFAULT_LLM_PROVIDER` | `openai` | Default provider: `openai`, `deepseek`, `anthropic`, or `local` |
| `OPENAI_API_KEY` | *(unset)* | OpenAI API key |
| `OPENAI_DEFAULT_MODEL` | `gpt-4` | Default OpenAI model |
| `DEEPSEEK_API_KEY` | *(unset)* | DeepSeek API key |
| `ANTHROPIC_API_KEY` | *(unset)* | Anthropic (Claude) API key |
| `LOCAL_LLM_URL` | `http://localhost:11434` | Local LLM server URL (e.g. Ollama) |
| `LOCAL_LLM_MODEL` | `llama3` | Local model name |

---

## API Documentation

| Variable | Default | Description |
|---|---|---|
| `API_DOCS_ENABLED` | *(follows DEBUG)* | `true` to always show docs, `false` to always hide |
| `API_DOCS_URL` | `/docs` | Swagger UI path |
| `API_REDOC_URL` | `/redoc` | ReDoc path |
| `API_OPENAPI_URL` | `/openapi.json` | OpenAPI schema path |

> In production (`FASTAPI_ENV=production`), API docs are hidden by default (`DEBUG=false`). Set `API_DOCS_ENABLED=true` to override.

---

## Data Ingestion Feature Flags

| Variable | Default | Description |
|---|---|---|
| `ENABLE_LINEAGE` | `true` | Track data lineage for ingested datasets |
| `ENABLE_SCHEMA_DRIFT` | `true` | Detect schema changes on re-upload |
| `ENABLE_RETENTION` | `true` | Enable data retention policies |
