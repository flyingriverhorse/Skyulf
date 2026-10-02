# Champion kararı: primary metric + guardrail metrikleri

Durum: **taslak, onay bekliyor.** Kapsam: `skyulf-core/skyulf/integrations/` (mlflow + databricks),
Bundle wizard şeması (frontend) ve docs.

## 1. Problem

`mlflow/validation.py::_comparison_decision` yalnızca seçilen tek metrik için
`improvement >= min_improvement` kuralını uygular. Diğer metrikler sadece mutlak
`quality_gates` eşiğine bakar, champion'a göre kıyaslanmaz. Sonuçlar:

- Primary biraz artarken başka bir metrik belirgin düşse bile candidate geçer.
- Primary eşit kalıp başka bir metrik artarsa candidate reddedilir.

## 2. Hedef davranış

Primary metric kararı belirler. Guardrail metrikleri champion'a göre kayıp
toleransı kontrol eder. İsteğe bağlı tie-break, primary eşitken guardrail
iyileşmesini kabul eder.

```
                 candidate + champion holdout metrikleri
                                  │
                      ┌───────────▼───────────┐
                      │ champion var mı?      │── hayır ─▶ no_champion
                      └───────────┬───────────┘
                                  │ evet
                      ┌───────────▼───────────┐
                      │ mutlak quality_gates  │── geçmedi ─▶ quality_gate_failed   (mevcut)
                      └───────────┬───────────┘
                                  │ geçti
                      ┌───────────▼───────────┐
                      │ guardrail kontrolü    │── biri toleransın
                      │ Δg ≥ -max_regression? │   altında ─▶ guardrail_regression  (YENİ)
                      └───────────┬───────────┘
                                  │ hepsi tolerans içinde
                      ┌───────────▼───────────┐
                      │ primary Δ ≥           │── evet ─▶ candidate_improved       (mevcut)
                      │ min_improvement (>0)? │
                      └───────────┬───────────┘
                                  │ hayır
                      ┌───────────▼───────────┐
                      │ primary Δ == 0 VE     │── evet ─▶ candidate_improved_secondary (YENİ)
                      │ bir guardrail Δg ≥    │
                      │ secondary_min_improv? │
                      └───────────┬───────────┘
                                  │ hayır
                                  ▼
                      insufficient_improvement                                       (mevcut)
```

`Δ` yön duyarlıdır: loss/error metriklerinde `champion - candidate`, diğerlerinde
`candidate - champion`. Pozitif değer iyileşmedir. Mevcut `_metric_improvement`
mantığı yeniden kullanılır.

### Örnekler

Primary `heldout_roc_auc`, `min_improvement=0.005`; guardrail `heldout_recall` ve
`heldout_f1` için `max_regression=0.01`. Champion: AUC 0.900, recall 0.80, F1 0.75.

| # | Candidate (AUC / recall / F1) | Δ primary | Δ recall | Δ F1 | Sonuç |
|---|---|---|---|---|---|
| A | 0.910 / 0.80 / 0.75 | +0.010 | 0 | 0 | `candidate_improved` |
| B | 0.915 / 0.76 / 0.75 | +0.015 | -0.04 | 0 | `guardrail_regression` |
| C | 0.900 / 0.83 / 0.75 | 0 | +0.03 | 0 | `candidate_improved_secondary` |
| D | 0.900 / 0.80 / 0.75 | 0 | 0 | 0 | `insufficient_improvement` |
| E | 0.898 / 0.85 / 0.75 | -0.002 | +0.05 | 0 | `insufficient_improvement` |

E kritik: tie-break yalnızca primary tam eşitken çalışır; primary sessizce
kötüleşemez.

## 3. Config

```python
"quality_policy": {
    "metric": "heldout_roc_auc",
    "min_improvement": 0.005,
    "quality_gates": {"heldout_recall": 0.70},          # mutlak alt sınır (mevcut)
    "guardrails": {                                      # YENİ: champion'a göre
        "heldout_recall": {"max_regression": 0.01},
        "heldout_f1": {"max_regression": 0.01},
    },
    "secondary_min_improvement": 0.01,                   # YENİ, opsiyonel
}
```

Kararlar:

- **Tolerans birimi:** mutlak (0.01 = 1 puan), metrik başına, yön duyarlı.
- **`secondary_min_improvement` yoksa** tie-break kapalıdır.
- **`guardrails` yoksa** bugünkü davranış aynen korunur (geriye uyumlu).
- Guardrail metriği primary ile aynı olamaz; görev tipine uygun olmalı; `max_regression >= 0` ve sonlu olmalı.
- "Eşit" karşılaştırması: Δ'nın tam sıfır olması. Kayan nokta gürültüsü için
  ayrı bir `tie_tolerance` eklemeyiz (basitlik); gerekirse sonra değerlendirilir.

## 4. Uygulama adımları (TDD sırasıyla)

1. **`mlflow/validation.py`**
   - Önce A–E senaryoları + doğrulama hataları için testler yaz.
   - `validate_quality_policy`: `guardrails`, `secondary_min_improvement` doğrulaması.
   - `_comparison_decision`: yukarıdaki akış; CCN ≤ 10 için guardrail değerlendirmesini
     ayrı yardımcıya çıkar (`_guardrail_deltas`, `_guardrail_violations`).
   - `ModelComparisonReport`: `guardrails` ve `guardrail_deltas` alanları (varsayılan `None`).
2. **Digest uyumluluğu** (`comparison_payload` / `comparison_digest`)
   - Yeni alanlar `None` iken payload'a eklenmez; eski kayıtların digest'i değişmez.
   - Test: guardrail'siz rapor için digest eski değerle aynı.
3. **`mlflow/promotion.py`**
   - `_validate_report_approval` (≈ satır 954): `reason` olarak
     `candidate_improved` **ve** `candidate_improved_secondary` kabul edilmeli,
     yoksa eligible model promote aşamasında reddedilir.
   - `_staging_decision` reason haritasına iki yeni neden eklenir.
4. **`databricks/model_set_quality.py` / model-set akışı**
   - Aynı karar kuralı her component için uygulanır; `failed_components` sebebi
     `guardrail_regression` olarak raporlanır.
5. **`databricks/workflow_config.py`**
   - `_validate_workflow_quality` yeni alanları doğrular; `preview_workflow_config`
     guardrail'ları özetler; `migrate_workflow_config` eski config'i değiştirmeden bırakır.
6. **Frontend (Bundle wizard şeması)**
   - `guardrails` ve `secondary_min_improvement` alanları; backend/Core ile seçenek
     listeleri karşılaştırılır (node-config senkron kuralı).
7. **Docs / changelog**
   - `docs/guides/mlflow_registry.md`, `docs/guides/databricks_bundle.md`.
   - `changelog/<major>.<minor>.x.md` girişi.

## 5. Doğrulama kapıları

- `ruff check .`, `ruff format --check ...`, tam `ty check` kapsamı (CI ile aynı).
- Etkilenen pytest: `skyulf-core/tests/integrations` + `test_internal_api_boundary.py`.
- Lizard CCN ≤ 10, pydantic model docstring'i eklenirse OpenAPI snapshot testi.
- Frontend: vitest, `lint`, `complexity:check`, `build`, `size-check`.

## 6. Riskler

| Risk | Önlem |
|---|---|
| Eski digest/receipt'lerin bozulması | Alan yokken payload aynı; digest regresyon testi |
| Promote aşamasında yeni reason'ın reddedilmesi | `promotion.py` güncellemesi + uçtan uca test |
| Tie-break ile primary'nin gizlice düşmesi | Yalnızca Δ == 0 iken; senaryo E testi |
| `local_retraining.py` / `promotion.py` büyüklüğü | Yeni mantık `validation.py` yardımcılarında kalır |

## 7. Drift ile ilişki

Drift izleme ve drift'ten tetiklenen retraining (random/temporal veri ekleme) diğer
bilgisayardaki push edilmemiş işte. Bu plan ondan bağımsızdır; retraining sonucu
oluşan candidate da aynı karar kuralından geçer. Drift işi merge edildikten sonra
`drift_check → retrain_trigger` fazı `PHASE_PREDECESSORS` içine eklenir.
