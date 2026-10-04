# Model-set quality gates and automatic activation

Date: 2026-09-29. Base: `90808117`. Implemented; live acceptance passed in report114.

Follow-up [114](114-model-set-challenger-delivery.md) extends this delivery with
challenger/previous_challenger and rejection. Its final acceptance/commit status
supersedes the earlier local-only status and missing-role limitations below.

## Contract

The user requested single-model-style automatic promotion for complete sets,
independent metric limits for each component, and an explanation of registry tags.

- Bundle setup adds `model_set_promotion_policy`: `manual_approval` by default,
  or `automatic`. Existing branch-level policies remain manual so components
  cannot independently move aliases while the complete set is being evaluated.
- Each branch's workflow supplies `metric`, `quality_threshold`, optional
  `quality_gates`, and `min_improvement`. Regression, classification and ensemble
  components use their own task-appropriate metric; scores are not ranked across targets.
- Before training, capture the champion SET once and pin its component versions.
  Independent component aliases do not define the replacement baseline.
- The first set requires all absolute gates to pass. A replacement additionally
  requires every component to strictly improve over its counterpart. Ties fail.
  No subset is promoted. Layout changes require a new set name.
- Freeze comparison digests and expected champion version into the set identity.
  At approval, replay the pinned training snapshot and holdout membership, load
  saved policies, and recompute metrics. Editable project configuration cannot
  change the candidate's gates.
- Both manual and automatic approval use these gates and the existing functional
  validation of all components/business rules under the common alias admission.
- A failed quality gate returns an inspectable candidate decision and preserves
  the champion. Missing/corrupt evidence, stale baseline, and execution errors
  fail the job. Manual approval cannot override a failed policy.
- Persist the decision artifact and inspection tags. Successful quality evidence
  is included in the existing durable set-validation proof; rollback verifies it.
- Scoring remains a separate job. Component aliases remain unchanged.
- Old sets keep their original hashes and remain readable/scoreable. Updated
  Bundle approval requires newly packaged quality evidence. The low-level legacy
  functional-only SDK remains available only for packages without quality pins.

## Code map

- `inference/_model_set_manifest.py`, `inference/model_set.py`: canonical optional
  quality pins, old artifact compatibility.
- `integrations/databricks/model_set_quality.py`: frozen evidence replay and all
  component decisions.
- `integrations/databricks/model_set_release.py`: baseline pinning, durable decision,
  manual/automatic release orchestration.
- `integrations/mlflow/model_set_lifecycle.py`: quality callback requirement,
  durable successful proof and rollback verification.
- `integrations/databricks/local_branches.py`, `branch_notebook.py`,
  `model_set_project.py`: branch baseline, package and Bundle operator wiring.
- Bundle schema, model-set factory, branch examples, README and SDK guide:
  configuration and behavior documentation.

## Verification

- Real local MLflow/SQLite training test, pandas AND Polars: weak regression,
  classification and voting ensemble models create automatic set v1; improved
  versions create automatic champion v2; tied next versions fail all components;
  manual approval of the failed set is blocked; rollback restores set v1.
  Separate component champions deliberately remain at v1 while the set baseline
  advances to v2, proving comparison does not follow component aliases.
- Artifact/project/MLflow lifecycle/quality/real-auto suite: **87 passed, 1 skipped**.
  Windows symlink creation requires additional privilege; no cloud claim is derived
  from this suite.
- Real Databricks CLI generation and branch/prompt tests: **31 passed**. Both
  compute types generate manual and automatic policies; still exactly two jobs.
  Sandbox initially prevented CLI process launch; local-generation-only execution
  with escalation passed. That initial local verification did not deploy or run paid tasks.
- New quality tests also reject stale baseline, absent threshold, extra gate
  failure, incomplete proof, and functional-only approval bypass.
- Initial regression exposed optional-null serialization changing canonical
  digest reconstruction. Omit the absent field when saving; both tampering tests
  and an explicit old-shape/hash compatibility test now pass.
- Branch training/scoring/final quality tests: **94 passed** (including six newly
  added proof/next-action cases; some quality tests overlap the preceding suite).
- Final rollback/validation replay after extracting the evidence-count helper:
  **3 passed**; no validation conditions changed.
- Full CI scopes passed: Ruff check, Ruff format (1116 files), Ty, Lizard CCN <= 10
  across both backend and Core. No full repository pytest, MkDocs or frontend rebuild
  was run for this Core/template change.

Final live acceptance now passes against Unity Catalog/Delta on both engines,
including all current quality, tag and challenger/rejection changes. See
[report114](114-model-set-challenger-delivery.md) for run IDs and limitations.

## Tags written by these lifecycle paths

Not every row applies to every object. The set's new quality tags live on its
registered MODEL VERSION; training tags live on component versions/runs. Metrics
are stored separately as MLflow metrics and JSON evidence, not only as tags.
MLflow/Databricks can also add environment-specific system tags; those are not a
fixed Skyulf-defined inventory. User-supplied custom tags are likewise open-ended.

| Tag | Location / scope | Meaning |
| --- | --- | --- |
| `model_set_model_count` | Set version, at Bundle registration | Number of saved component models |
| `model_set_<branch>_name` | Set version, at Bundle registration | Qualified registered name of that component; long names continue in `_name_2`, `_name_3`, etc. to preserve the 256-byte value bound |
| `model_set_<branch>_version` | Set version, at Bundle registration | Exact component version embedded in this set |
| `model_set_<branch>_type` | Set version, at Bundle registration | Selected estimator/ensemble type, unwrapped from any tuner |
| `model_set_promotion_policy` | Set version after a quality decision | `automatic` or `manual_approval` used for this decision |
| `model_set_quality_status` | Set version | Latest quality result: `passed` / `failed`; does not itself mean champion |
| `model_set_quality_sha256` | Set version | Canonical decision payload fingerprint |
| `model_set_quality_artifact` | Set version | `runs:/.../decision.json` containing per-model metrics, gates, baseline and reasons |
| `model_set_validation_sha256` | Approved set version | Fingerprint of durable functional plus quality validation |
| `model_set_validation_artifact` | Approved set version | Artifact URI used to verify validation and rollback |
| `promotion_status` | Model/set version | Historical promotion status, e.g. `promoted`, `not_promoted`; NOT the current alias |
| `promotion_<event_id>` | Destination model/set version | Durable alias receipt: action, from_version, proof, parent, state, previous |
| `champion_current_event` | Registered model/set | Event/version marker for the controlled current champion |
| `pending_alias_event` | Registered model/set, temporary | Alias mutation intent awaiting verified completion; blocks unsafe retry |
| `challenger_current_event` | Registered model/set | Marker for the current nominated challenger |
| `previous_challenger_current_event` | Registered model/set | Marker for saved previous-challenger history |
| `challenger_history_<event_id>` | Model/set version | Prior/new challenger history and prepared/committed state |
| `validation_status` | Model/set candidate version | `pending`, `passed`, `rejected` or `error` evaluation state |
| `validation_reason` | Model/set candidate version | Readable explanation of that evaluation state |
| `quality_gate_event` | Single-model candidate version | Event associated with the recorded metric gates |
| `quality_gate_<metric>` | Single-model candidate version | JSON with metric, direction, threshold, value, passed, event_id |
| `approval_status` | Explicitly rejected model/set version | `rejected`; distinct from a failed quality evaluation |
| `approval_reason` | Explicitly rejected model/set version | Operator's rejection reason |
| `task` | Component training run + version | `training`; this field is not classification/regression |
| `model_type` | Component training run + version | Estimator type, including an ensemble type when selected |
| `engine` | Component training run + version | `pandas` or `polars` execution engine |
| `split_strategy` | Component training run + version | Selected training/holdout split strategy |
| `train_data_destination` | Component training run + version | Source Delta table for training |
| `test_data_destination` | Component training run + version | Source Delta table for heldout evaluation |
| `train_data_version` | Component training run + version | Pinned Delta snapshot version used for training |
| `test_data_version` | Component training run + version | Pinned Delta snapshot version used for evaluation |
| `candidate_date_tag` | Component training run + version | Candidate training date in UTC |
| `train_start` | Component run + version, when set | Training window start |
| `test_start` | Component run + version, when set | Holdout window start |
| `data_end` | Component run + version, when set | Data window cutoff |
| `result_cutoff` | Component run + version, when set | Label/result availability cutoff |
| `risk_category` | Component run + version, when supplied | User-provided risk classification |
| `mlflow.parentRunId` | Component child run | Links it to the coordinated branch-training parent run |
| `skyulf.training.branch` | Component child run | Branch name from branches.py |
| `skyulf.training.plan_sha256` | Parent + child training runs | Fingerprint of the shared pinned training plan |
| `skyulf.training.status` | Parent training run | Branch-training progress: running / complete / failed |
| `skyulf.lifecycle.job_id` | Single-model lifecycle run | Databricks job identity |
| `skyulf.lifecycle.job_run_id` | Single-model lifecycle run | Concrete Databricks job invocation |
| `skyulf.lifecycle.status` | Single-model lifecycle run | Overall incomplete/finished/failed lifecycle state |
| `skyulf.lifecycle.request` | Single-model lifecycle run | Fingerprint of the prepared immutable request |
| `skyulf.lifecycle.<phase>.attempt` | Single-model lifecycle run | Phase started/failed marker protecting retries |
| `skyulf.lifecycle.<phase>.receipt` | Single-model lifecycle run | Fingerprint of completed phase evidence |
| `skyulf.lifecycle.registration_intent` | Single-model lifecycle run | Registration was started |
| `skyulf.lifecycle.registration` | Single-model lifecycle run | Fingerprint of the saved registration receipt |

`champion`, `previous_champion`, `challenger` and `previous_challenger` are ALIASES,
not tags. With follow-up114, sets support all four. After rollback, v2 can still say
`promotion_status=promoted` because it was promoted in the past; the `champion`
alias identifies the version currently active.

`model_set_digest`, `local_pipeline_digest` and `bundle_digest` are MLmodel
artifact metadata fields, not model-version tags. Existing `skyulf_artifact_kind`
and `skyulf_execution_scope` likewise describe artifact kind/execution scope.
No tag rename/migration was introduced in this feature.

Follow-up: new Bundle set registrations now attach the component inventory tags
listed above, before either manual or automatic approval. Existing remote versions
are not backfilled. Local project suite: 13 passed; expanded regression/classification/
ensemble tag checks: 3 passed (one overlapping earlier case). Real MLflow registration
checks verify tags on both saved set versions. Full Ruff/format/Ty/CCN10 and diff
checks passed. This tag addition has not been deployed to Databricks.
