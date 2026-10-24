# Databricks Bundle guide index

For all Databricks, MLflow, Spark and preprocessing guides with short page
descriptions, open the [integrations documentation index](integrations.md).

<span id="databricks-bundle-configuration-reference"></span>

Start with the [end-to-end project guide](databricks_bundle_walkthrough.md) to
create a project, train a model and publish predictions. Before you generate one,
[choose a project shape](databricks_choose.md): layout, prediction path and what
runs on Spark. Use the topic pages below for a specific change. Generated project Markdown contains short instructions
and links to these references.

The project pins its Core version in `deployment/requirements.txt`. These public
pages describe the current template; use the [upgrade guide](databricks_project_upgrade.md)
when the installed release differs. Training fits in one bounded Python process;
[distributed inference](databricks_scoring.md) scores an admitted saved model on Spark workers.

## Configuration and operations

| Task | Detailed guide |
| --- | --- |
| <span id="choose-a-layout-and-setup-mode"></span>Choose a layout and setup mode | [Open guide](databricks_project_setup.md#choose-a-layout-and-setup-mode) |
| <span id="files-you-edit"></span>Files you edit | [Open guide](databricks_project_setup.md#files-you-edit) |
| <span id="what-a-configuration-change-requires"></span>What a configuration change requires | [Open guide](databricks_project_setup.md#what-a-configuration-change-requires) |
| <span id="training-and-inference-boundaries"></span>Training and inference boundaries | [Open guide](databricks_spark_scoring.md) |
| <span id="distributed-inference-settings"></span>Distributed inference settings | [Open guide](databricks_spark_scoring.md) |
| <span id="monitoring-investigation-and-training-lineage"></span>Monitoring investigation and training lineage | [Open guide](databricks_monitoring_dashboard.md) |
| <span id="guided-setup-and-offline-preview"></span>Guided setup and offline preview | [Open guide](databricks_project_setup.md#guided-setup-and-offline-preview) |
| <span id="custom-preprocessing-recipes"></span>Custom preprocessing recipes | [Open guide](databricks_custom_steps.md#custom-preprocessing-recipes) |
| <span id="writing-your-own-step"></span>Writing your own step | [Open guide](databricks_custom_steps.md#writing-your-own-step) |
| <span id="write-your-own-pre-split-filter"></span>Write your own pre-split filter | [Open guide](databricks_custom_pre_split.md) |
| <span id="preprocessing-diagnostics"></span>Preprocessing diagnostics | [Open guide](preprocessing_context.md) |
| <span id="source-layout-and-existing-projects"></span>Source layout and existing projects | [Open guide](databricks_project_setup.md) |
| <span id="local-input-limits"></span>Local input limits | [Open guide](databricks_training_data.md#local-input-limits) |
| <span id="record-identity-and-result-availability"></span>Record identity and result availability | [Open guide](databricks_training_data.md#record-identity-and-result-availability) |
| <span id="choose-the-evaluation-split-and-result-availability"></span>Choose the evaluation split and result availability | [Open guide](databricks_training_data.md#choose-the-evaluation-split-and-result-availability) |
| <span id="search-and-cross-validation"></span>Search and cross-validation | [Open guide](databricks_tuning.md) |
| <span id="optional-classification-decision-thresholds"></span>Optional classification decision thresholds | [Open guide](databricks_decision_thresholds.md) |
| <span id="ensemble-recipes"></span>Ensemble recipes | [Open guide](databricks_tuning.md) |
| <span id="explicit-training-sampling"></span>Explicit training sampling | [Open guide](databricks_training_data.md#explicit-training-sampling) |
| <span id="source-windows-are-independent-of-splitting-and-scheduling"></span>Source windows are independent of splitting and scheduling | [Open guide](databricks_training_data.md#source-windows-are-independent-of-splitting-and-scheduling) |
| <span id="source-date-formats-and-timezones"></span>Source date formats and timezones | [Open guide](databricks_training_data.md#source-date-formats-and-timezones) |
| <span id="configuration-validation-and-migration"></span>Configuration validation and migration | [Open guide](databricks_project_upgrade.md#configuration-validation-and-migration) |
| Upgrade integration code | [Open guide](databricks_project_upgrade.md#upgrade-integration-code) |
| <span id="where-the-workflow-lives"></span>Where the workflow lives | [Open guide](databricks_training_reports.md) |
| Understand files in an experiment's Artifacts tab | [Artifact reference](databricks_artifacts.md) |
| <span id="independent-scoring-and-promotion-policies"></span>Independent scoring and promotion policies | [Open guide](databricks_lifecycle.md#independent-scoring-and-promotion-policies) |
| <span id="operator-actions-through-run-now"></span>Operator actions through Run now | [Open guide](databricks_lifecycle.md#operator-actions-through-run-now) |
| <span id="approve-an-existing-candidate-without-training"></span>Approve an existing candidate without training | [Open guide](databricks_lifecycle.md#approve-an-existing-candidate-without-training) |
| <span id="reject-a-candidate-or-roll-back-a-promotion"></span>Reject a candidate or roll back a promotion | [Open guide](databricks_lifecycle.md#reject-a-candidate-or-roll-back-a-promotion) |
| <span id="recent-challenger-history"></span>Recent challenger history | [Open guide](databricks_lifecycle.md#recent-challenger-history) |
| Recover an interrupted alias change | [Open guide](databricks_lifecycle_recovery.md) |
| Train, approve and score a multi-target model set | [Open guide](databricks_model_sets.md) |
| Set the model_set block | [Open guide](databricks_model_sets.md#set-the-model_set-block) |
| Choose a model-set output mode | [Open guide](databricks_model_sets.md#choose-the-output-mode) |
| Compose model-set outputs with declarative rules | [Open guide](databricks_model_sets.md#declarative-composition) |
| Model-set quality policy and replacement rule | [Open guide](databricks_model_sets.md#quality-policy-and-replacement-rule) |
| Score a model set | [Open guide](databricks_model_sets.md#score-the-set) |
| Find and read SHAP explanations | [Open guide](databricks_explanations.md) |
| <span id="what-is-created"></span>What is created | [Open guide](databricks_deployment.md) |
| <span id="additional-quality-gates"></span>Additional quality gates | [Open guide](databricks_lifecycle.md#additional-quality-gates) |
| <span id="reading-notebook-results-and-failures"></span>Reading notebook results and failures | [Open guide](databricks_training_reports.md) |
| <span id="first-run"></span>First run | [Open guide](databricks_bundle_walkthrough.md) |
| <span id="runtime-package-and-compute-configuration"></span>Runtime package and compute configuration | [Open guide](databricks_deployment.md) |
| <span id="explicit-retry-policy"></span>Explicit retry policy | [Open guide](databricks_deployment.md) |
| <span id="job-resource-files"></span>Job resource files | [Open guide](databricks_deployment.md#job-resource-files) |
| Remove a deployed project | [Open guide](databricks_teardown.md) |
| Fit, inspect and score from Python | [Open guide](databricks_sdk.md) |
| Publish one calendar period of predictions to Delta | [Open guide](databricks_batch.md) |

<span id="optional-basic-model-cross-validation"></span>
For basic or nested cross-validation, see
[CV and total work](databricks_tuning.md#cv-nested-cv-and-total-work).

## Monitoring and serving

Detailed setup asks `prediction_delivery`: `batch` creates the score job,
`endpoint` creates only the serving job, and `both` creates both. Quick setup
defaults to `batch`. All choices keep monitoring. Logged endpoints created by
the serving job enroll automatically after verification; observations still
require requests and a monitoring run. See [delivery choices and setup](databricks_serving.md#choose-how-predictions-are-produced).

| Task | Detailed guide |
| --- | --- |
| Understand how monitoring runs and what to read first | [Monitoring overview](databricks_monitoring.md) |
| Create or upgrade the shared monitoring store | [Store setup](databricks_monitoring_setup.md) |
| Enroll model identities and prepare reference data | [Enrollment](databricks_monitoring_enrollment.md) |
| Match later outcomes to predictions | [Delayed labels](databricks_monitoring_labels.md) |
| Investigate input distribution and data quality | [Drift](databricks_drift.md) |
| Detect observed model performance loss | [Performance](databricks_performance.md) |
| Configure guarded automatic retraining | [Retraining](databricks_retraining.md) |
| Read and filter saved observations | [Dashboard](databricks_monitoring_dashboard.md) |
| Create an endpoint and query it | [Serving](databricks_serving.md) |
| Publish and look up online feature values | [Online features](databricks_online_features.md) |
| Observe requests, latency and incomplete logs | [Serving monitoring](databricks_serving_monitoring.md) |
| Increase challenger traffic each day | [Daily rollout](databricks_serving_rollout.md) |

## Preprocessing step catalog

The [preprocessing catalog](preprocessing_recipes.md) explains recipes and phases
and indexes every registered step. Each family has its own page:

| Family | Page |
| --- | --- |
| Missing values, cleaning and types | [Open page](preprocessing_recipes_cleaning.md) |
| Scaling and numeric transformations | [Open page](preprocessing_recipes_scaling.md) |
| Bounds and outliers | [Open page](preprocessing_recipes_outliers.md) |
| Categorical encoding | [Open page](preprocessing_recipes_encoding.md) |
| Binning | [Open page](preprocessing_recipes_binning.md) |
| Feature generation and selection | [Open page](preprocessing_recipes_features.md) |
| Time and geography | [Open page](preprocessing_recipes_time_geo.md) |
| Text and embeddings | [Open page](preprocessing_recipes_text.md) |
| Split, sampling and diagnostic Core nodes | [Open page](preprocessing_recipes_sampling.md) |
| Custom factory contracts and ready functions | [Open page](preprocessing_recipes_custom.md) |

## Reference pages

For exact fields and selectable operations, use the
[training configuration](databricks_training_config.md),
[model catalog](databricks_model_catalog.md), [preprocessing catalog](preprocessing_recipes.md),
[scoring reference](databricks_scoring.md) and [feature table guide](databricks_features.md). The
[glossary](databricks_glossary.md) defines shared terms;
[troubleshooting](databricks_troubleshooting.md) maps validation messages to fixes.
