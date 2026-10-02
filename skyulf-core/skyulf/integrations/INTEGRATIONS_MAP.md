# Integrations haritası (mlflow + databricks)

Mermaid diyagramları VS Code'da "Markdown Preview" (Mermaid destekli) ile görüntülenir.
Kaynak: `skyulf-core/skyulf/integrations/` (databricks 52 modül, mlflow 13 modül).

## 1. Modül haritası

```mermaid
flowchart TB
    subgraph CORE["skyulf core (inference, preprocessing, modeling)"]
        C1[Pipeline / Estimators]
        C2[inference.project_code]
    end

    subgraph ML["integrations/mlflow"]
        direction TB
        ML_C[_client.py<br/>_model_metadata.py]
        ML_T[tracking.py]
        ML_R[registry.py]
        ML_M[model.py / local_model.py<br/>pyfunc paketleme]
        ML_V[validation.py<br/>ModelComparisonReport,<br/>quality gates, karar kuralı]
        ML_CH[challenger.py<br/>rejection.py]
        ML_P[promotion.py<br/>alias, receipt, rollback]
        ML_MS[model_set.py<br/>model_set_challenger.py<br/>model_set_lifecycle.py]
        ML_C --> ML_T & ML_R
        ML_R --> ML_V --> ML_CH --> ML_P
        ML_M --> ML_R
        ML_P --> ML_MS
    end

    subgraph DB["integrations/databricks"]
        direction TB
        subgraph CFG["Yapılandırma ve proje"]
            D_CFG[workflow_config.py<br/>local_sdk.py<br/>project*.py<br/>_project_*.py]
        end
        subgraph TRN["Eğitim"]
            D_RT[local_retraining.py<br/>local_pre_split.py<br/>local_cv.py<br/>local_search*.py]
            D_CMP[local_competition.py<br/>competition_*.py<br/>training_nodes.py]
            D_BR[local_branches.py<br/>branch_tasks.py<br/>branch_notebook.py]
            D_EV[local_training_evidence.py<br/>local_explanations.py<br/>explanation_report.py]
        end
        subgraph LIFE["Lifecycle (Bundle)"]
            D_LT[lifecycle_tasks.py]
            D_LS[_lifecycle_state.py<br/>_lifecycle_data.py]
            D_JR[job_runtime.py<br/>job_output.py<br/>training_node_*.py]
        end
        subgraph SCORE["Skorlama ve yayın"]
            D_WF[local_workflow.py]
            D_INC[local_incremental.py<br/>local_history.py<br/>scoring_pre_split.py]
            D_PUB[local_publish.py<br/>prediction_output.py<br/>delta.py / admission.py<br/>delta_admission.py]
        end
        subgraph MSET["Model set"]
            D_MS[model_set_project.py<br/>model_set_stages.py<br/>model_set_quality.py<br/>model_set_release.py<br/>model_set_batch.py<br/>model_set_output.py]
        end
        CFG --> TRN
        TRN --> LIFE
        LIFE --> SCORE
        TRN --> MSET
        MSET --> SCORE
    end

    C1 --> TRN
    C1 --> SCORE
    C2 --> LIFE
    TRN -->|kayıt, karşılaştırma| ML_R
    LIFE -->|faz kanıtı, tag| ML_T
    LIFE -->|alias kararı| ML_P
    MSET -->|set nominate / release| ML_MS
```

## 2. Lifecycle faz akışı (`PHASE_PREDECESSORS`)

```mermaid
flowchart LR
    prepare --> load_data --> prepare_dataset
    prepare --> train
    train --> select_best_model
    train --> evaluate_register --> compare --> decide
    prepare -.-> operator
    prepare -.-> finalize
    prepare -.-> result

    subgraph grouped["Gruplanmış task'lar"]
        TR["train_register<br/>(train + evaluate_register)"]
        CD["compare_decide<br/>(compare + decide)"]
    end
```

Kesikli oklar: yalnızca `prepare` sonrası çalışma şartı (sıralama dışarıda zorlanır).
Her faz önce `attempt` tag'i yazar, sonra receipt üretir; belirsiz sonuçta retry
reddedilir (operatör incelemesi gerekir).

## 3. Champion / challenger / alias durumu

```mermaid
stateDiagram-v2
    [*] --> Registered: register_candidate
    Registered --> NoChampion: ilk model
    NoChampion --> Champion: initialize_champion<br/>(quality + operatör onayı)
    Registered --> Challenger: stage_challenger<br/>(eligible)
    Registered --> Rejected: operatör reddi
    Challenger --> Champion: promote_candidate<br/>(candidate_improved)
    Challenger --> Rejected: reject
    Champion --> Champion: rollback_promotion<br/>(önceki sürüme)
    Champion --> [*]
```

Karar nedenleri: `no_champion`, `quality_gate_failed`, `candidate_improved`,
`insufficient_improvement`. Planlanan: `guardrail_regression`,
`candidate_improved_secondary` (bkz. `GUARDRAIL_PLAN.md`).

## 4. Uçtan uca veri akışı

```mermaid
sequenceDiagram
    participant Op as Operatör/Bundle
    participant LT as lifecycle_tasks
    participant TR as local_retraining
    participant ML as mlflow (registry/promotion)
    participant DL as Delta (publish)

    Op->>LT: prepare (config + pinned snapshot)
    LT->>TR: train (split, pre-split, CV/search)
    TR-->>LT: fitted artifact + evidence
    LT->>ML: evaluate_register (run, model version)
    LT->>ML: compare (candidate vs champion, holdout)
    ML-->>LT: ModelComparisonReport
    LT->>ML: decide (stage_challenger / promote)
    Op->>ML: operator onayı / red
    LT->>DL: scoring publish (guarded, receipt)
    DL-->>Op: committed receipt
```

## 5. İyileştirme önerileri

| # | Öneri | Gerekçe | Öncelik |
|---|---|---|---|
| 1 | Çoklu metrik / guardrail kuralı | Tek metrik kararı diğer metriklerin düşüşünü yakalamıyor (`GUARDRAIL_PLAN.md`) | Yüksek |
| 2 | Drift izleme ve drift'ten retraining fazı | `profiling/drift.py` integrations'a bağlı değil; faz grafiğine `drift_check → retrain_trigger` eklenmeli | Yüksek |
| 3 | `local_retraining.py` (1700 satır) bölünmeli | Pre-split doğrulama yardımcıları (`_validate_*`, `_pre_split_*`) zaten `local_pre_split.py`'ye ait; sample/filter doğrulaması ve candidate fit/log ayrı modüllere alınabilir. CCN ve okunabilirlik | Yüksek |
| 4 | `promotion.py` (1035) ve `lifecycle_tasks.py` (875) bölünmeli | Alias event/receipt/rollback işleri ve faz yürütücüleri ayrı sorumluluklar | Orta |
| 5 | Faz grafiğini koddan üreten test | Dokümandaki diyagram ile `PHASE_PREDECESSORS` zamanla ayrışır; test Mermaid'i üretip doküman ile kıyaslasın | Orta |
| 6 | Belirsiz sonuç için operatör runbook'u | "Retry reddedilir, operatör incelemesi gerekir" ama nasıl inceleneceği tek yerde yazılı değil | Orta |
| 7 | `databricks/` düz yapısı alt paketlere ayrılabilir | 52 modül tek klasörde (`local_*` 25 adet). `training/`, `scoring/`, `model_set/`, `bundle/` gibi. `INTERNAL_API.md` alias yasakladığı için taşıma tek PR'da ve importlarla birlikte yapılmalı | Düşük (riskli) |
| 8 | Model-set için ortak karar kuralı | `evaluate_model_set_quality` her component'in geçmesini ister; guardrail/tie-break kuralı component bazında aynı çalışmalı | Orta |
| 9 | Local ve Bundle yolu arasında tekrar kontrolü | `local_workflow`/`local_sdk` ile `lifecycle_tasks` aynı eğitim/kanıt mantığını paylaşmalı (`local_training_evidence` doğru yön); yeni özellikler iki yola da uygulanmalı | Orta |

Önerilen sıra: 1 → 3 (guardrail eklemeden önce yer aç) → 2 → 8 → 5/6.
