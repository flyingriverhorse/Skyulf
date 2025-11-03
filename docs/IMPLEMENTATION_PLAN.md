# Implementation Plan — Privacy-Preserving MLOps Building Blocks

**Duration**: 6 months  
**Budget**: €30k  
**Goal**: Add privacy, federation, and portability to existing MLOps platform

---

## Milestone 1: Differential Privacy Core (Month 1–2)

### What We're Building
A system to train ML models while protecting individual data points from being reverse-engineered.

### Why It Matters
- Hospitals/governments need to train on sensitive data without leaking patient/citizen info
- Proves mathematically how much privacy is preserved (ε/δ guarantees)
- Required for GDPR-compliant ML in many EU use cases

### Technical Steps

#### 1.1 DP-SGD Training Wrapper
**What**: Wrap existing scikit-learn models to add differential privacy during training.

**Implementation** (following your node-based architecture):
```
Location: core/feature_engineering/nodes/modeling/dp_training.py

Function: apply_dp_training(df, config, pipeline_id, node_id, storage)
  - Reads config: epsilon, delta, clipping_norm, target_column, model_type
  - Fits model with DP-SGD (clips gradients, adds noise)
  - Tracks ε spent, stores in TransformerStorage
  - Returns df unchanged, attaches metadata

Location: core/privacy/dp_sgd.py (helper module)
  - DPSGDWrapper class (sklearn-compatible)
  - PrivacyAccountant for ε/δ tracking
  
Node Catalog Entry (node_catalog.json):
{
  "type": "dp_training",
  "label": "Train with Differential Privacy",
  "category": "modeling",
  "tags": ["privacy", "training"],
  "parameters": [
    {
      "name": "epsilon",
      "label": "Privacy Budget (ε)",
      "type": "number",
      "default": 1.0,
      "min": 0.1,
      "max": 10.0,
      "step": 0.1,
      "description": "Lower = more private, less accurate"
    },
    {
      "name": "delta",
      "label": "Failure Probability (δ)",
      "type": "number",
      "default": 1e-5,
      "description": "Typically 1/n²"
    },
    {
      "name": "clipping_norm",
      "label": "Gradient Clipping Norm",
      "type": "number",
      "default": 1.0
    },
    {
      "name": "model_type",
      "label": "Model Type",
      "type": "select",
      "options": ["logistic_regression", "linear_svm", "mlp"]
    }
  ]
}

API Route (routes.py):
POST /ml-workflow/api/nodes/dp-training/apply
Body: {pipeline_id, node_id, config: {epsilon, delta, ...}}
Response: {status, epsilon_spent, accuracy, metadata}

Celery Task (model_training_tasks.py):
@celery_app.task
def train_with_dp(job_id, pipeline_id, config):
    # Long-running DP training
    # Records ε in privacy ledger
    # Stores model in TransformerStorage
```

**Tests**:
- `tests/nodes/modeling/test_dp_training.py`
  - Verify noise added (non-deterministic output)
  - Check ε/δ tracking in storage
  - Ensure accuracy degrades with lower ε

---

#### 1.2 Privacy Ledger
**What**: Database that records every training run's privacy cost.

**Implementation** (following your SQLAlchemy + async DB patterns):
```
Location: core/database/models.py (add new model)

SQLAlchemy Model:
class PrivacyLedger(Base):
    __tablename__ = "privacy_ledger"
    
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    pipeline_id: Mapped[Optional[str]]
    training_job_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("training_jobs.id"))
    dataset_source_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("data_sources.id"))
    epsilon: Mapped[float]
    delta: Mapped[float]
    training_samples: Mapped[int]
    model_type: Mapped[str]
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    metadata_: Mapped[Optional[Dict]] = mapped_column(JSON, name="metadata")

Location: core/privacy/ledger.py (service layer)

async def record_privacy_budget(
    session: AsyncSession,
    pipeline_id: str,
    epsilon: float,
    delta: float,
    dataset_source_id: uuid.UUID,
    training_samples: int,
    metadata: Optional[Dict] = None
) -> PrivacyLedger:
    # Insert record, return ledger entry

async def get_total_budget(
    session: AsyncSession,
    dataset_source_id: uuid.UUID
) -> Dict[str, float]:
    # Sum ε across all runs for dataset
    # Return {"total_epsilon": X, "run_count": Y}

async def check_budget_exceeded(
    session: AsyncSession,
    dataset_source_id: uuid.UUID,
    max_epsilon: float
) -> bool:
    # True if cumulative ε > threshold

API Routes (core/privacy/routes.py - NEW):
router = APIRouter(prefix="/api/privacy", tags=["privacy"])

POST /api/privacy/record
GET /api/privacy/budget/{dataset_source_id}
GET /api/privacy/report (export CSV/JSON)

Pydantic Schemas (core/privacy/schemas.py - NEW):
class PrivacyLedgerEntry(BaseModel):
    id: uuid.UUID
    epsilon: float
    delta: float
    created_at: datetime
    ...

class PrivacyBudgetSummary(BaseModel):
    dataset_source_id: uuid.UUID
    total_epsilon: float
    run_count: int
    entries: List[PrivacyLedgerEntry]
```

**Tests**:
- `tests/privacy/test_ledger.py`
  - Record multiple runs, verify cumulative ε
  - Test budget warnings/blocking
  - Verify async DB operations

---

#### 1.3 Data Minimization Helpers
**What**: Automatic tools to reduce data before training.

**Implementation** (as feature engineering nodes):
```
Location: core/feature_engineering/nodes/data_consistency/privacy_minimization.py

Node 1: Blur Sensitive Text
Function: apply_blur_sensitive_text(df, config, pipeline_id, node_id, storage)
Config: {columns, patterns: ["email", "ssn", "phone"], redact_char: "*"}

Node 2: Clip Outliers (privacy-focused)
Function: apply_privacy_clip_outliers(df, config, pipeline_id, node_id, storage)
Config: {columns, method: "std", n_std: 3, clip_values_stored: True}
Uses TransformerStorage to persist clip bounds

Node 3: Drop High-Uniqueness Columns
Function: apply_drop_high_uniqueness(df, config, pipeline_id, node_id, storage)
Config: {uniqueness_threshold: 0.9, auto_detect: True}

Node Catalog Entries (add to node_catalog.json):
{
  "type": "blur_sensitive_text",
  "label": "Redact Sensitive Text",
  "category": "data_consistency",
  "tags": ["privacy", "pii", "gdpr"],
  "parameters": [
    {
      "name": "columns",
      "label": "Columns to Redact",
      "type": "multi_select",
      "source": {"type": "column_list", "filter": "text"}
    },
    {
      "name": "patterns",
      "label": "Pattern Types",
      "type": "multi_select",
      "options": ["email", "ssn", "phone", "credit_card"]
    }
  ]
}

API Routes (existing routes.py):
POST /ml-workflow/api/nodes/blur-sensitive-text/apply
POST /ml-workflow/api/nodes/privacy-clip-outliers/apply
POST /ml-workflow/api/nodes/drop-high-uniqueness/apply

Recommendations Endpoint (NEW):
GET /ml-workflow/api/recommendations/privacy-risks
Response: {
  high_uniqueness_columns: [...],
  potential_pii_columns: [...],
  quasi_identifiers: [...]
}
```

**Tests**:
- `tests/nodes/data_consistency/test_privacy_minimization.py`
  - Verify redaction patterns work
  - Check clipping preserves distribution shape
  - Ensure uniqueness detection accurate

---

#### 1.4 Risk Heuristics Report
**What**: Automated report flagging re-identification risks.

**Implementation** (follows your recommendations pattern):
```
Location: core/feature_engineering/recommendations/privacy_risk.py

Class: PrivacyRiskAnalyzer
Methods:
- analyze_risks(df, dataset_source_id) -> PrivacyRiskReport
  Checks:
  - High uniqueness columns (>90% unique values)
  - Quasi-identifiers (age+zip+gender combinations)
  - k-anonymity score
  - Potential PII patterns (regex for emails, SSN, phone)
  
Pydantic Schemas (schemas.py):
class PrivacyRiskColumn(BaseModel):
    column: str
    risk_level: Literal["high", "medium", "low"]
    uniqueness_percentage: float
    reason: str
    recommended_action: str  # "drop", "generalize", "blur"

class PrivacyRiskReport(BaseModel):
    dataset_source_id: uuid.UUID
    generated_at: datetime
    k_anonymity_score: Optional[int]
    high_risk_columns: List[PrivacyRiskColumn]
    quasi_identifiers: List[List[str]]  # Combinations
    recommendations: List[str]
    overall_risk: Literal["critical", "high", "medium", "low"]

API Route (routes.py):
GET /ml-workflow/api/recommendations/privacy-risks
Query: dataset_source_id
Response: PrivacyRiskReport

# Mirrors existing pattern:
# GET /ml-workflow/api/recommendations/drop-columns
# GET /ml-workflow/api/recommendations/label-encoding

Integration:
- Called by UI when user opens privacy tab
- Results feed into blur/drop/clip node recommendations
- Cached in eda_cache/ like other recommendations
```

**Tests**:
- `tests/recommendations/test_privacy_risk.py`
  - Detect known quasi-identifiers (age+zip)
  - Calculate k-anonymity correctly
  - Flag SSN/email patterns

---

#### 1.5 Security & Threat Model
**What**: Document threats and mitigations (STRIDE).

**Implementation** (documentation + config enforcement):
```
Location: docs/THREAT_MODEL.md (NEW document)

Structure:
1. Threat Categories (STRIDE):
   - Spoofing: Model poisoning via fake federated updates
   - Tampering: Gradient manipulation, ledger bypass
   - Repudiation: Untracked privacy budget use
   - Info Disclosure: Membership inference, privacy leaks
   - DoS: Resource exhaustion via training jobs
   - Elevation: Unauthorized training/data access

2. Mitigations (mapped to existing auth system):
   - Signed model updates (JWT tokens from core.auth)
   - Gradient clipping + noise (DP-SGD)
   - Mandatory ledger writes (DB constraints)
   - DP guarantees + auditing (PrivacyLedger)
   - Rate limits (Celery queue limits, existing)
   - RBAC (core.user_management roles)

3. Secure Defaults (config.py):
   - PRIVACY_MIN_EPSILON = 0.1
   - PRIVACY_REQUIRE_LEDGER = True
   - PRIVACY_AUDIT_LOG_ENABLED = True
   - FEDERATED_REQUIRE_SIGNATURES = True

Location: config.py (add privacy settings)
class Settings(BaseSettings):
    # Existing settings...
    
    # Privacy settings
    PRIVACY_MIN_EPSILON: float = 0.1
    PRIVACY_MAX_EPSILON: float = 10.0
    PRIVACY_REQUIRE_LEDGER: bool = True
    PRIVACY_AUDIT_LOG_ENABLED: bool = True
    
    # Federated settings
    FEDERATED_REQUIRE_SIGNATURES: bool = True
    FEDERATED_MAX_CLIENTS: int = 100
    FEDERATED_TIMEOUT_SECONDS: int = 300

Location: core/privacy/security_checks.py (NEW)
def validate_privacy_config(epsilon, delta):
    """Enforce minimum epsilon, valid delta range."""
    settings = get_settings()
    if epsilon < settings.PRIVACY_MIN_EPSILON:
        raise ValueError(f"ε must be >= {settings.PRIVACY_MIN_EPSILON}")
    if delta <= 0 or delta >= 1:
        raise ValueError("δ must be in (0, 1)")
```

**Tests**:
- `tests/privacy/test_security_checks.py`
  - Reject ε < 0.1 by default
  - Enforce ledger writes
  - Validate JWT signatures (federated)

---

### Milestone 1 Deliverables
- ✅ DPSGDWrapper with ε/δ tracking
- ✅ Privacy ledger (DB + API)
- ✅ Minimization CLI + helpers
- ✅ Risk analysis API
- ✅ Threat model doc
- ✅ 15+ unit tests
- ✅ Example notebook: `examples/01_differential_privacy.ipynb`

---

## Milestone 2: Federated Learning Adapter (Month 3–4)

### What We're Building
Let multiple sites train the same model without sharing raw data—only share encrypted model updates.

### Why It Matters
- Hospitals can collaborate without violating patient privacy
- Companies can pool insights without exposing trade secrets
- Enables cross-border ML under strict data sovereignty laws

### Technical Steps

#### 2.1 Aggregator Service
**What**: Central server that collects and averages model updates.

**Implementation** (follows your Celery + DB patterns):
```
Location: core/database/models.py (add new models)

SQLAlchemy Models:
class FederatedRound(Base):
    __tablename__ = "federated_rounds"
    
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    pipeline_id: Mapped[str]
    round_number: Mapped[int]
    status: Mapped[str]  # "waiting", "aggregating", "completed"
    min_clients: Mapped[int]
    max_clients: Mapped[int]
    timeout_seconds: Mapped[int]
    global_model_path: Mapped[Optional[str]]
    started_at: Mapped[datetime]
    completed_at: Mapped[Optional[datetime]]

class FederatedClientUpdate(Base):
    __tablename__ = "federated_client_updates"
    
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    round_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("federated_rounds.id"))
    client_id: Mapped[str]
    model_weights_path: Mapped[str]  # uploads/federated/{round_id}/{client_id}.pkl
    num_samples: Mapped[int]
    signature: Mapped[str]  # JWT
    submitted_at: Mapped[datetime]
    verified: Mapped[bool] = mapped_column(default=False)

Location: core/federated/aggregator.py (service layer)

async def create_federated_round(
    session: AsyncSession,
    pipeline_id: str,
    min_clients: int,
    max_clients: int
) -> FederatedRound:
    # Create round, return round_id

async def submit_client_update(
    session: AsyncSession,
    round_id: uuid.UUID,
    client_id: str,
    model_weights: bytes,
    num_samples: int,
    signature: str
):
    # Save weights to uploads/federated/
    # Verify JWT (use core.auth)
    # Insert FederatedClientUpdate
    # Trigger aggregation if min_clients reached

Location: core/federated/tasks.py (NEW Celery tasks)

@celery_app.task
def aggregate_federated_round(round_id: str):
    """FedAvg aggregation - runs async in background."""
    1. Load all client updates for round
    2. Verify JWT signatures
    3. Weighted average: Σ(n_k / n_total * weights_k)
    4. Save global model to uploads/federated/{round_id}/global.pkl
    5. Update round status = "completed"
    6. Store in TransformerStorage

API Routes (core/federated/routes.py - NEW)
router = APIRouter(prefix="/api/federated", tags=["federated"])

POST /api/federated/rounds/create
POST /api/federated/rounds/{round_id}/submit
GET /api/federated/rounds/{round_id}/status
GET /api/federated/rounds/{round_id}/model

Pydantic Schemas (core/federated/schemas.py - NEW)
class FederatedRoundCreate(BaseModel):
    pipeline_id: str
    min_clients: int = 2

class FederatedClientSubmit(BaseModel):
    client_id: str
    model_weights: str  # base64
    num_samples: int
    signature: str  # JWT
```

**Tests**:
- `tests/federated/test_aggregator.py`
  - 3 clients submit → verify weighted avg
  - Invalid JWT → reject
  - Timeout → mark round failed

---

#### 2.2 Secure Client
**What**: Lightweight client that trains locally and submits updates.

**Implementation** (standalone CLI tool + API client):
```
Location: scripts/federated_client.py (NEW standalone script)

Class: FederatedClient
Methods:
- fetch_global_model(server_url, round_id) -> model_weights
- train_local(model, df, epochs) -> new_weights
- sign_update(weights, private_key) -> JWT (uses python-jose)
- submit_update(server_url, round_id, weights, signature)

CLI:
python scripts/federated_client.py \
  --server https://aggregator.example.com \
  --data local_train.csv \
  --epochs 5 \
  --client-id hospital_A \
  --api-key ${API_KEY} \
  --round-id ${ROUND_ID}

Workflow:
1. GET /api/federated/rounds/{round_id}/model
2. Train on local CSV
3. Sign with JWT (using api-key from core.auth)
4. POST /api/federated/rounds/{round_id}/submit
5. Poll status until aggregation complete
6. Repeat for next round

Configuration: clients/client_config.json
{
  "client_id": "hospital_A",
  "server_url": "https://aggregator.example.com",
  "api_key_env": "FEDERATED_API_KEY",
  "data_path": "./local_data.csv",
  "model_storage": "./client_models/"
}
```

**Tests**:
- `tests/federated/test_client.py`
  - Mock server responses
  - Verify JWT signing
  - Test network retry logic

---

#### 2.3 Flower Integration (Optional)
**What**: Plug into Flower framework to leverage battle-tested FL.

**Implementation** (optional backend, reduces custom code):
```
Location: core/federated/flower_adapter.py (NEW)

Why Flower:
- Production-ready secure aggregation
- Built-in encryption for model updates
- Saves ~50% dev time vs custom implementation
- Actively maintained (10k+ GitHub stars)

Adapter Pattern:
import flwr as fl

class SkyulfFlowerClient(fl.client.NumPyClient):
    def __init__(self, data_source_id, session):
        self.data_source_id = data_source_id
        self.session = session
    
    def get_parameters(self):
        # Load from TransformerStorage
        return model_to_numpy(self.model)
    
    def fit(self, parameters, config):
        # Train using existing training pipeline
        df = load_data_source(self.data_source_id)
        model = train_model(df, parameters)
        return model_to_numpy(model), len(df), {}

Server (runs alongside FastAPI):
flwr.server.start_server(
    server_address="0.0.0.0:8080",
    strategy=fl.server.strategy.FedAvg(
        min_fit_clients=2,
        min_available_clients=2
    ),
    config=fl.server.ServerConfig(num_rounds=10)
)

Client CLI:
python scripts/flower_client.py \
  --server aggregator:8080 \
  --data-source-id ${DATA_SOURCE_ID}

Integration Decision:
- Month 3: Build custom aggregator (learning + control)
- Month 4: Add Flower as optional backend if time permits
- Document tradeoffs in docs/federated_architecture.md
```

**Why**: Optional fallback that reduces risk if custom impl hits issues.

---

#### 2.4 Docker Compose Demo
**What**: One-command setup to run aggregator + 3 clients.

**Implementation** (follows your docker patterns):
```
Location: docker/federated/docker-compose.yml (NEW)

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: skyulf_mlflow
      POSTGRES_USER: skyulf
      POSTGRES_PASSWORD: dev_password
    volumes:
      - federated_db:/var/lib/postgresql/data
  
  redis:
    image: redis:7-alpine
    
  aggregator:
    build:
      context: ../..
      dockerfile: Dockerfile
    command: python -m uvicorn main:app --host 0.0.0.0 --port 8000
    ports: ["8000:8000"]
    environment:
      - DATABASE_URL=postgresql+asyncpg://skyulf:dev_password@postgres/skyulf_mlflow
      - REDIS_URL=redis://redis:6379
    depends_on: [postgres, redis]
    volumes:
      - ../../uploads:/app/uploads
  
  celery_worker:
    build:
      context: ../..
      dockerfile: Dockerfile
    command: celery -A celery_worker.celery_app worker --loglevel=info
    environment:
      - DATABASE_URL=postgresql+asyncpg://skyulf:dev_password@postgres/skyulf_mlflow
      - REDIS_URL=redis://redis:6379
    depends_on: [postgres, redis]
    volumes:
      - ../../uploads:/app/uploads
  
  client_1:
    build:
      context: ../..
      dockerfile: Dockerfile
    command: python scripts/federated_client.py \
      --server http://aggregator:8000 \
      --data /data/client1.csv \
      --client-id hospital_A \
      --api-key demo_key_1
    volumes:
      - ./demo_data:/data
    depends_on: [aggregator]
  
  client_2:
    # Same pattern for client 2
  
  client_3:
    # Same pattern for client 3

volumes:
  federated_db:

Location: docker/federated/demo_data/ (synthetic datasets)
- client1.csv (hospital A data)
- client2.csv (hospital B data)
- client3.csv (hospital C data)

Run:
cd docker/federated
docker-compose up

# Logs show:
# - Aggregator starts on :8000
# - 3 clients connect, train locally
# - After 10 rounds, final model accuracy displayed
```

**Tests**:
- `tests/integration/test_federated_demo.py`
  - Spin up compose stack
  - Verify 3 clients submit updates
  - Check final global model exists
  - Accuracy >= baseline

---

### Milestone 2 Deliverables
- ✅ Aggregator service (FedAvg baseline)
- ✅ Secure client with JWT signing
- ✅ Flower adapter (optional)
- ✅ Docker compose demo
- ✅ 10+ integration tests
- ✅ Example: `examples/02_federated_learning.ipynb`

---

## Milestone 3: Interoperable Packaging (Month 5)

### What We're Building
Export models to standard formats so they work across different systems.

### Why It Matters
- Avoid vendor lock-in (can switch from sklearn to TensorFlow)
- Deploy same model to edge devices, web servers, embedded systems
- Regulatory compliance (auditors can inspect packaged models)

### Technical Steps

#### 3.1 ONNX Export/Import
**What**: Convert trained models to ONNX format.

**Implementation** (adds export to existing TrainingJob):
```
Location: core/feature_engineering/nodes/modeling/model_export.py (NEW)

Functions:
def export_training_job_to_onnx(
    training_job_id: uuid.UUID,
    output_path: str,
    sample_input: Optional[pd.DataFrame] = None
):
    """Export trained model to ONNX format."""
    # Load model from TrainingJob.model_artifact_path
    # Use skl2onnx for sklearn models
    # Use torch.onnx.export for PyTorch
    # Save to uploads/exports/{training_job_id}/model.onnx

def import_onnx_model(onnx_path: str) -> ONNXInferenceWrapper:
    """Wrap onnxruntime.InferenceSession with .predict() interface."""
    # Returns sklearn-compatible wrapper

Location: core/database/models.py (extend TrainingJob)
class TrainingJob(Base):
    # Existing fields...
    onnx_export_path: Mapped[Optional[str]]
    onnx_exported_at: Mapped[Optional[datetime]]

API Routes (add to routes.py):
POST /ml-workflow/api/training-jobs/{job_id}/export/onnx
Response: {onnx_path, file_size, export_time}

GET /ml-workflow/api/training-jobs/{job_id}/download/onnx
Response: application/octet-stream (ONNX file)

POST /ml-workflow/api/training-jobs/{job_id}/validate/onnx
Body: {test_data_source_id}
Response: {predictions_match: true, max_diff: 1e-6}

Pydantic Schemas (schemas.py):
class ONNXExportRequest(BaseModel):
    sample_rows: int = 10  # For shape inference

class ONNXExportResponse(BaseModel):
    onnx_path: str
    file_size_bytes: int
    export_time_seconds: float
    validation_passed: bool
```

**Tests**:
- `tests/nodes/modeling/test_model_export.py`
  - Export sklearn LogisticRegression → reimport → same predictions
  - Verify ONNX file structure
  - Test validation endpoint

---

#### 3.2 MLflow-Compatible Bundles
**What**: Package model + metadata in MLflow format (no external server needed).

**Implementation** (extends existing model artifacts):
```
Location: core/feature_engineering/nodes/modeling/model_bundle.py (NEW)

Bundle Structure (in uploads/bundles/{training_job_id}/):
model_bundle/
  MLmodel              # MLflow descriptor (YAML)
  model.pkl            # Trained model (joblib)
  transformers/        # Pipeline transformers from TransformerStorage
    scaler_age.pkl
    encoder_city.pkl
  conda.yaml           # Environment dependencies
  requirements.txt     # pip dependencies
  input_example.json   # Sample input for validation
  metadata.json        # Custom: privacy budget, dataset info, user

Functions:
async def create_mlflow_bundle(
    session: AsyncSession,
    training_job_id: uuid.UUID,
    output_dir: str
) -> str:
    """Package trained model + transformers into MLflow format."""
    # Load TrainingJob from DB
    # Load transformers from TransformerStorage
    # Generate MLmodel YAML
    # Copy artifacts to bundle directory
    # Return bundle path

async def load_mlflow_bundle(bundle_path: str) -> Dict:
    """Load model and metadata from bundle."""
    # Validate MLmodel YAML
    # Load model.pkl
    # Load transformers
    # Return {model, transformers, metadata}

API Routes (add to routes.py):
POST /ml-workflow/api/training-jobs/{job_id}/bundle/create
Response: {bundle_path, bundle_size_mb}

GET /ml-workflow/api/training-jobs/{job_id}/bundle/download
Response: application/zip (bundle.zip)

GET /ml-workflow/api/training-jobs/{job_id}/bundle/metadata
Response: MLmodel YAML contents

Pydantic Schemas (schemas.py):
class MLflowBundleMetadata(BaseModel):
    training_job_id: uuid.UUID
    created_at: datetime
    model_type: str
    sklearn_version: str
    python_version: str
    privacy_epsilon: Optional[float]
    transformers_included: List[str]
```

**Tests**:
- `tests/nodes/modeling/test_model_bundle.py`
  - Create bundle → load in isolated env → predictions match
  - Verify MLmodel YAML structure
  - Test dependency resolution

---

#### 3.3 Provenance Log
**What**: Tamper-evident record of how model was created.

**Implementation** (auto-generated from TrainingJob):
```
Location: core/database/models.py (add ModelProvenance)

SQLAlchemy Model:
class ModelProvenance(Base):
    __tablename__ = "model_provenance"
    
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    training_job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("training_jobs.id"))
    created_at: Mapped[datetime]
    pipeline_id: Mapped[str]
    dataset_source_id: Mapped[uuid.UUID]
    dataset_hash: Mapped[str]  # SHA256 of training data
    dataset_rows: Mapped[int]
    algorithm: Mapped[str]
    hyperparameters: Mapped[Dict] = mapped_column(JSON)
    transformers_applied: Mapped[List[str]] = mapped_column(JSON)
    privacy_epsilon: Mapped[Optional[float]]
    privacy_delta: Mapped[Optional[float]]
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    signature: Mapped[Optional[str]]  # RSA-SHA256 signature
    provenance_json: Mapped[Dict] = mapped_column(JSON)  # Full W3C PROV

Location: core/feature_engineering/nodes/modeling/provenance.py (NEW)

async def create_provenance_record(
    session: AsyncSession,
    training_job: TrainingJob,
    df_hash: str
) -> ModelProvenance:
    """Auto-generate provenance from training job."""
    # Extract all metadata from TrainingJob
    # Query PrivacyLedger for ε/δ if DP used
    # Query TransformerStorage for pipeline details
    # Generate W3C PROV JSON
    # Sign with private key (from config)
    # Store in DB

def sign_provenance(provenance_dict: Dict, private_key_path: str) -> str:
    """Sign provenance JSON with RSA key."""
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    # Return base64(signature)

async def verify_provenance(
    session: AsyncSession,
    provenance_id: uuid.UUID,
    public_key_path: str
) -> bool:
    """Verify provenance signature."""

API Routes (add to routes.py):
GET /ml-workflow/api/training-jobs/{job_id}/provenance
Response: ModelProvenance JSON (W3C PROV format)

POST /ml-workflow/api/training-jobs/{job_id}/provenance/verify
Body: {public_key}
Response: {valid: true, verified_at: "..."}

GET /ml-workflow/api/training-jobs/{job_id}/provenance/download
Response: provenance.json file

Pydantic Schemas (schemas.py):
class ProvenanceRecord(BaseModel):
    id: uuid.UUID
    training_job_id: uuid.UUID
    dataset_hash: str
    algorithm: str
    privacy_epsilon: Optional[float]
    signature: str
    w3c_prov: Dict  # Full W3C PROV-DM format
```

**Tests**:
- `tests/nodes/modeling/test_provenance.py`
  - Create provenance → verify signature
  - Tamper with JSON → verify fails
  - Validate W3C PROV structure

---

#### 3.4 Software Bill of Materials (SBOM)
**What**: List all dependencies for security audits.

**Implementation** (auto-generated with bundles):
```
Location: core/feature_engineering/nodes/modeling/sbom.py (NEW)

Format: CycloneDX JSON (standard for security audits)

Functions:
def generate_sbom_for_bundle(bundle_path: str) -> Dict:
    """Generate CycloneDX SBOM from bundle requirements."""
    # Parse requirements.txt from bundle
    # Query PyPI for license info
    # Include Python version, OS
    # Return CycloneDX JSON

def generate_sbom_from_training_job(
    training_job_id: uuid.UUID
) -> Dict:
    """Generate SBOM from current environment."""
    import pkg_resources
    installed = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    # Filter to packages actually used by model
    # Generate CycloneDX format

CycloneDX Structure:
{
  "bomFormat": "CycloneDX",
  "specVersion": "1.4",
  "version": 1,
  "components": [
    {
      "type": "library",
      "name": "scikit-learn",
      "version": "1.4.0",
      "licenses": [{"license": {"id": "BSD-3-Clause"}}],
      "purl": "pkg:pypi/scikit-learn@1.4.0"
    },
    ...
  ]
}

API Routes (add to routes.py):
GET /ml-workflow/api/training-jobs/{job_id}/sbom
Response: CycloneDX JSON

GET /ml-workflow/api/training-jobs/{job_id}/sbom/download
Response: sbom.json file

POST /ml-workflow/api/training-jobs/{job_id}/sbom/validate
Response: {valid: true, vulnerabilities: [...]}

Integration with MLflow Bundle:
- SBOM auto-generated when bundle created
- Saved to model_bundle/sbom.json
- Included in provenance metadata

CLI (optional):
python scripts/generate_sbom.py \
  --training-job-id ${JOB_ID} \
  --output sbom.json
```

**Tests**:
- `tests/nodes/modeling/test_sbom.py`
  - Generate SBOM → validate CycloneDX schema
  - Verify license detection
  - Test vulnerability scanning (mock)

---

#### 3.5 Validation CLI
**What**: Command to check bundle integrity.

**Implementation** (CLI + API endpoint):
```
Location: scripts/validate_bundle.py (NEW standalone script)

CLI:
python scripts/validate_bundle.py \
  --bundle ./model_bundle \
  --verify-signature \
  --check-dependencies \
  --public-key keys/public.pem

Validation Checks:
1. ✓ MLmodel file exists and valid YAML
2. ✓ model.pkl exists and loads with joblib
3. ✓ requirements.txt parseable
4. ✓ All dependencies installable (pip check)
5. ✓ Provenance signature valid (if --verify-signature)
6. ✓ SBOM complete and valid CycloneDX
7. ✓ Input example shape matches model
8. ✓ Transformers directory complete

Output:
✓ Bundle validation PASSED
  - MLmodel: valid
  - Model artifact: 2.4 MB
  - Dependencies: 12 packages, all available
  - Provenance: signature valid
  - SBOM: 12 components listed
  
Exit code: 0 (success) or 1 (failure)

Location: core/feature_engineering/nodes/modeling/bundle_validator.py (NEW)

class BundleValidator:
    def validate(self, bundle_path: str) -> ValidationReport
    def check_mlmodel(self, mlmodel_path: str) -> bool
    def check_dependencies(self, requirements_path: str) -> bool
    def verify_provenance(self, provenance_json: Dict, public_key: str) -> bool

API Route (add to routes.py):
POST /ml-workflow/api/bundles/validate
Body: {bundle_path or bundle_upload}
Response: {
  valid: true,
  checks: {
    mlmodel: {passed: true},
    model_artifact: {passed: true, size_mb: 2.4},
    dependencies: {passed: true, count: 12},
    provenance: {passed: true, signature_valid: true},
    sbom: {passed: true, components: 12}
  },
  errors: []
}

Pydantic Schemas (schemas.py):
class BundleValidationRequest(BaseModel):
    bundle_path: Optional[str]
    verify_signature: bool = False
    public_key: Optional[str]

class BundleValidationReport(BaseModel):
    valid: bool
    checks: Dict[str, Dict]
    errors: List[str]
    warnings: List[str]
```

**Tests**:
- `tests/nodes/modeling/test_bundle_validator.py`
  - Valid bundle → all checks pass
  - Missing MLmodel → error
  - Invalid signature → error
  - Uninstallable dependency → warning

---

### Milestone 3 Deliverables
- ✅ ONNX export/import
- ✅ MLflow-compatible bundler
- ✅ Provenance log with signatures
- ✅ SBOM generator
- ✅ Validation CLI
- ✅ 12+ tests
- ✅ Example: `examples/03_model_packaging.ipynb`

---

## Milestone 4: Hardening & Documentation (Month 6)

### What We're Building
Production-ready deployment guides, security baselines, and tutorials.

### Why It Matters
- Real users need step-by-step instructions
- Security auditors need threat models and hardening checklists
- Partners/pilots need smooth onboarding

### Technical Steps

#### 4.1 CI/CD Hardening
**What**: Enforce quality gates on every PR.

**Implementation** (extends existing .github/workflows/ci.yml):
```
Location: .github/workflows/ci.yml (extend existing)

Current CI (already done):
- pytest + coverage
- flake8 linting
- mypy type checking

Add to CI:
jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - name: Install security tools
        run: |
          pip install bandit safety
      
      - name: SAST (Bandit)
        run: bandit -r core/ -ll  # Only high/medium severity
      
      - name: Dependency vulnerabilities (Safety)
        run: safety check --json
      
      - name: License compliance
        run: |
          pip install pip-licenses
          pip-licenses --format=json --fail-on="GPL"
  
  test:
    # Existing test job
    steps:
      # Add coverage threshold
      - name: Coverage check
        run: pytest --cov --cov-fail-under=70

Location: .github/workflows/security.yml (NEW - runs nightly)

name: Security Scan
on:
  schedule:
    - cron: '0 2 * * *'  # 2 AM daily
  workflow_dispatch:

jobs:
  full-security-scan:
    runs-on: ubuntu-latest
    steps:
      - name: Trivy vulnerability scan
        uses: aquasecurity/trivy-action@master
      - name: OWASP dependency check
        # Scan for known vulnerabilities
      - name: Report to security dashboard
        # Send results to monitoring

Pre-commit Hooks (.pre-commit-config.yaml - NEW):
repos:
  - repo: https://github.com/psf/black
    rev: 23.0.0
    hooks:
      - id: black
  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
        args: ['-ll']
```

**Tests**:
- CI passes on clean code
- CI fails on security issues
- Coverage threshold enforced

---

#### 4.2 Admin Guide
**What**: How to deploy and configure the platform.

**Implementation** (comprehensive deployment docs):
```
Location: docs/ADMIN_GUIDE.md (NEW)

Table of Contents:

1. Deployment Options
   a. Docker Compose (recommended for PoC)
      - docker-compose.yml walkthrough
      - Environment variables (.env.example)
      - Volume mounts (uploads/, logs/, data/)
   
   b. Kubernetes (production)
      - Helm chart (charts/skyulf-mlflow/)
      - Values.yaml configuration
      - Ingress + TLS setup
   
   c. Bare metal (advanced)
      - systemd service files
      - Nginx reverse proxy config
      - PostgreSQL + Redis setup

2. Configuration Reference
   - config.py settings (all env vars documented)
   - DATABASE_URL connection strings
   - REDIS_URL for Celery
   - PRIVACY_MIN_EPSILON, FEDERATED_* settings
   - CORS_ORIGINS, ALLOWED_HOSTS

3. Security Hardening Checklist
   ✓ HTTPS enabled (Let's Encrypt or custom cert)
   ✓ Firewall rules (only 443/80 public)
   ✓ Database encrypted at rest
   ✓ JWT secret rotated (SECRET_KEY env var)
   ✓ RBAC roles configured (admin, analyst, viewer)
   ✓ Audit logging enabled (PRIVACY_AUDIT_LOG_ENABLED=true)
   ✓ Rate limiting on APIs
   ✓ Security headers (HSTS, CSP, X-Frame-Options)

4. Backup & Recovery
   - Database backups:
     pg_dump schedule (daily automated)
   - Model artifacts backup:
     uploads/ → S3/MinIO sync
   - Disaster recovery runbook

5. Monitoring & Observability
   - Health endpoint: GET /health
   - Prometheus metrics: /metrics
   - Grafana dashboards (JSON exports in docs/grafana/)
   - Alerting rules:
     * Celery queue depth > 100
     * Privacy budget > 80% of max
     * Training job failures > 5%
   - Log aggregation (ELK or Loki)

6. Upgrade Guide
   - Database migrations (alembic upgrade head)
   - Zero-downtime rolling updates
   - Rollback procedure

Appendix:
- Troubleshooting common issues
- Performance tuning (Celery workers, DB pool)
- Network diagrams
```

---

#### 4.3 Developer Guide
**What**: How to extend the platform.

**Implementation** (onboarding for contributors):
```
Location: docs/DEVELOPER_GUIDE.md (NEW)

Table of Contents:

1. Architecture Overview
   - Component diagram (FastAPI + Celery + Redis + PostgreSQL)
   - Node-based feature engineering system
   - TransformerStorage pattern
   - Data flow: upload → EDA → nodes → training → export
   - Async DB patterns (SQLAlchemy 2.0)

2. Development Setup
   - Prerequisites (Python 3.10+, Docker, Git)
   - Clone repo + install deps (uv or pip)
   - Configure .env for local dev
   - Run migrations: alembic upgrade head
   - Start services:
     * uvicorn main:app --reload
     * celery -A celery_worker.celery_app worker
     * docker run redis:7-alpine
     * docker run postgres:15

3. Adding New Feature Engineering Nodes
   Step 1: Implement node function
     Location: core/feature_engineering/nodes/<category>/<node_name>.py
     Signature: apply_<node>(df, config, pipeline_id, node_id, storage)
   
   Step 2: Add to node catalog
     Location: node_catalog.json
     Define: type, label, parameters, default_config
   
   Step 3: Wire API route
     Location: core/feature_engineering/routes.py
     Add: POST /ml-workflow/api/nodes/<node>/apply
   
   Step 4: Add Pydantic schemas
     Location: core/feature_engineering/schemas.py
   
   Step 5: Write tests
     Location: tests/nodes/<category>/test_<node>.py

4. Adding Privacy/Federated Features
   - Follow same node pattern
   - Use existing DB models (extend if needed)
   - Celery tasks for long-running work
   - Store results in TransformerStorage or DB

5. Database Migrations
   - Create migration:
     alembic revision --autogenerate -m "Add privacy ledger"
   - Review generated migration in migrations/versions/
   - Apply: alembic upgrade head
   - Rollback: alembic downgrade -1

6. API Documentation
   - OpenAPI auto-generated at /docs
   - Add docstrings to route functions
   - Pydantic models auto-document request/response

7. Testing
   - Unit tests: tests/nodes/, tests/privacy/
   - Integration tests: tests/integration/
   - Run: pytest -v
   - Coverage: pytest --cov --cov-report=html

8. Code Style & Standards
   - Black formatter (line length 120)
   - isort for imports
   - flake8 for linting
   - mypy for type checking
   - Pre-commit hooks enforce automatically

9. Pull Request Process
   - Fork repo + create feature branch
   - Write tests first (TDD)
   - Ensure CI passes (pytest + lint + mypy)
   - Update docs if adding features
   - Request review from maintainers
```

---

#### 4.4 Tutorials
**What**: Step-by-step walkthroughs for common tasks.

**Implementation** (hands-on guides with screenshots):
```
Location: docs/tutorials/ (NEW directory)

Tutorial 1: Privacy-Preserving Training (01_privacy_training.md)
---
Goal: Train a model with differential privacy and track budget.

Steps:
1. Upload dataset via /api/data/sources/upload
2. Run EDA: GET /ml-workflow/api/recommendations/privacy-risks
3. Apply blur node to sensitive columns
4. Add DP training node to pipeline:
   {type: "dp_training", epsilon: 1.0, delta: 1e-5}
5. Execute training job (Celery)
6. Check privacy ledger: GET /api/privacy/budget/{dataset_id}
7. Export to ONNX: POST /ml-workflow/api/training-jobs/{id}/export/onnx
8. Validate: python scripts/validate_bundle.py

Expected outcome: Model trained with ε=1.0, accuracy ~85%, privacy budget tracked.

Tutorial 2: Federated Learning (02_federated_setup.md)
---
Goal: Train model across 3 simulated hospitals without sharing data.

Steps:
1. Start aggregator: docker-compose -f docker/federated/docker-compose.yml up
2. Prepare 3 client datasets (split covertype_dataset.csv into 3)
3. Client 1: python scripts/federated_client.py --data client1.csv
4. Clients 2-3: same pattern
5. Watch aggregation logs (10 rounds)
6. Download global model: GET /api/federated/rounds/{id}/model
7. Evaluate on test set

Expected outcome: Final accuracy > individual clients, data never shared.

Tutorial 3: Model Packaging & Deployment (03_model_deployment.md)
---
Goal: Package trained model for production deployment.

Steps:
1. Train model (any pipeline)
2. Create MLflow bundle: POST /ml-workflow/api/training-jobs/{id}/bundle/create
3. Download bundle.zip: GET .../bundle/download
4. Extract and inspect:
   - MLmodel (YAML descriptor)
   - model.pkl (trained model)
   - transformers/ (scalers, encoders)
   - provenance.json (signed record)
   - sbom.json (dependencies)
5. Validate: python scripts/validate_bundle.py --bundle ./bundle/
6. Deploy to edge device (example Dockerfile)
7. Test inference endpoint

Expected outcome: Portable, validated model bundle ready for deployment.

Tutorial 4: Security Audit (04_security_audit.md)
---
Goal: Perform security audit on a trained model.

Steps:
1. Generate SBOM: GET /ml-workflow/api/training-jobs/{id}/sbom
2. Check for vulnerabilities: python scripts/check_vulnerabilities.py
3. Verify provenance signature: POST .../provenance/verify
4. Review threat model: docs/THREAT_MODEL.md
5. Check audit logs: SELECT * FROM privacy_ledger;
6. Validate privacy budget spent
7. Run penetration tests (optional)

Expected outcome: Security report confirming model integrity and privacy compliance.

Each tutorial includes:
- Prerequisites (data, env setup)
- Step-by-step commands
- Expected outputs (screenshots/logs)
- Troubleshooting section
- Links to API docs
```

---

#### 4.5 Demo & Pilot
**What**: End-to-end demo with synthetic data + small partner pilot.

**Implementation** (reproducible demo + real-world pilot):
```
Demo Package:
Location: demo/ (NEW directory)

Structure:
demo/
  README.md                    # One-command quickstart
  demo_data/
    healthcare_synthetic.csv   # 10k rows, no real PII
    client_1.csv               # Split for federated demo
    client_2.csv
    client_3.csv
  pipelines/
    dp_training_pipeline.json  # Pre-configured DP pipeline
    federated_pipeline.json    # Pre-configured FL pipeline
  scripts/
    01_upload_data.py          # Upload demo data via API
    02_run_dp_training.py      # Execute DP training
    03_run_federated.py        # Launch federated demo
    04_validate_outputs.py     # Check results
  docker-compose.demo.yml      # All-in-one demo stack
  results/                     # Output artifacts (gitignored)

Quickstart:
# One-command demo
cd demo
docker-compose -f docker-compose.demo.yml up

# Logs show:
# ✓ Server started
# ✓ Data uploaded
# ✓ DP training: ε=1.0, accuracy=85%
# ✓ Privacy budget tracked
# ✓ Federated: 3 clients, 10 rounds
# ✓ Global model accuracy=88%
# ✓ ONNX exported
# ✓ Bundle validated

demo/README.md:
# Privacy-Preserving MLOps Demo

## What This Demo Shows
- Train model with differential privacy
- Federated learning across 3 sites
- ONNX export + bundle validation
- Privacy budget tracking

## Run It (5 minutes)
1. `docker-compose up`
2. Open http://localhost:8000/demo
3. Watch logs for results
4. Explore outputs in results/

---

Pilot Program:
Partner: Small municipality or healthcare SME

Use Case Options:
1. Municipality: Analyze citizen survey data (privacy-sensitive)
2. Healthcare: Multi-site patient outcome analysis (HIPAA/GDPR)
3. Finance: Credit risk modeling across branches (data sovereignty)

Pilot Plan:
Week 1:
  - Onboarding call (architecture walkthrough)
  - Deploy to partner's on-prem server
  - Load partner's real dataset (anonymized)
  - Configure privacy budgets

Week 2:
  - Train model with DP
  - Validate privacy guarantees
  - Export model for production
  - Collect feedback

Deliverables from Pilot:
- Deployment report (issues, resolution time)
- Feedback survey (usability, performance, security)
- Testimonial (if positive results)
- Case study write-up (with permission)
- Improvements backlog

Success Metrics:
- Model deployed successfully
- Privacy budget stayed under threshold
- Partner satisfied (NPS > 8)
- No data leaks/incidents
```

---

### Milestone 4 Deliverables
- ✅ CI/CD security checks
- ✅ Admin guide (deployment, hardening)
- ✅ Developer guide (architecture, contrib)
- ✅ 4 tutorials
- ✅ End-to-end demo
- ✅ Pilot with partner (feedback report)
- ✅ Tagged release v1.0.0

---

## Summary: Month-by-Month Breakdown

| Month | Focus | Key Deliverable | What You Can Show |
|-------|-------|----------------|-------------------|
| 1 | DP Core | DPSGDWrapper + ledger | "Train model with ε=1.0, see privacy budget" |
| 2 | DP Complete | Risk analysis + threat model | "Automated risk report flags sensitive columns" |
| 3 | FL Aggregator | Central server + API | "3 clients submit updates, server aggregates" |
| 4 | FL Client | Secure client + demo | "One-command federated training on docker" |
| 5 | Packaging | ONNX + MLflow + provenance | "Export model, reimport anywhere, verify signature" |
| 6 | Hardening | Docs + pilot + release | "Full admin guide, tutorials, pilot testimonial" |

---

## What You Already Have (Leverage)

From existing codebase:
- ✅ FastAPI app with async DB
- ✅ Celery for background tasks
- ✅ Feature engineering pipelines
- ✅ Model training utilities
- ✅ SQLAlchemy models
- ✅ Basic tests (pytest)
- ✅ Authentication/authorization

What You'll Add (New):
- Privacy layer (DP, ledger)
- Federation layer (aggregator, client)
- Packaging layer (ONNX, MLflow, provenance)
- Documentation layer (guides, tutorials)

---

## Success Metrics

After 6 months, you should have:
1. **Working Code**: All features pass CI, 80%+ test coverage
2. **Demo**: One-command demo anyone can run locally
3. **Docs**: Admin + dev guides, 4 tutorials
4. **Pilot**: At least 1 partner using it with feedback
5. **Release**: v1.0.0 tagged on GitHub with signed artifacts
6. **Community**: 5+ GitHub stars, 2+ external contributors

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| DP math complexity | Use proven libraries (Opacus, TensorFlow Privacy) |
| FL security holes | Leverage Flower, add signature verification |
| ONNX compatibility issues | Test with multiple runtimes (onnxruntime, TensorRT) |
| Partner pilot delays | Have backup synthetic demo ready |
| Scope creep | Ship MVPs first, polish later |

---

## Next Steps to Start

1. **Week 1**: Set up dev environment, create `core/privacy/` folder structure
2. **Week 2**: Implement basic DPSGDWrapper (no noise yet), write tests
3. **Week 3**: Add noise + ε/δ tracking, validate with known DP examples
4. **Week 4**: Build privacy ledger DB schema + API endpoints
5. Continue iterating through plan above...

Ready to start? Pick one:
- A) Start with DP wrapper implementation now
- B) Review/adjust this plan first
- C) Create skeleton folder structure for all milestones
