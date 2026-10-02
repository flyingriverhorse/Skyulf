# Bundle'da düzenlenebilir integration kodu

Durum: **taslak.** Hedef: kod core'da kalır, Bundle projesinde de gerekirse elle
değiştirilebilir olur; kanıt zinciri (digest/receipt) bozulmaz.

## 1. Karar: iki aşama

| Aşama | İçerik | Ne zaman |
|---|---|---|
| **1. Hook'lar** | Sık değişen noktalar projede küçük, düzenlenebilir dosyalar | Önce, guardrail ve dosya bölme sonrası |
| **2. Tam kopya** (opsiyonel) | `src/skyulf_integrations/` init seçeneği ile | Yalnızca Aşama 1 yetmediği somut bir ihtiyaç çıkarsa |

Gerekçe: Tam kopya (~19k satır) her projede çoğalır; core düzeltmeleri ulaşmaz, kopya
core testlerinin dışında kalır. Hook'lar ise mevcut `src/modeling/*.py` ve
`src/features/` kalıbıyla aynıdır ve zaten `load_project_module` +
`project_source_digest` ile model kanıtına girer. Tam emin olunmadığı için
Aşama 2'yi şimdilik yapmıyoruz; karar kriterleri §4'te.

## 2. Aşama 1: hook'lar

Her hook, varsayılan olarak core'u çağıran ince bir dosyadır. Kullanıcı değiştirmezse
davranış core ile birebir aynıdır.

| Dosya (Bundle) | Core karşılığı | Değiştirilebilen |
|---|---|---|
| `src/lifecycle/decision.py` | `mlflow/validation.py::_comparison_decision` | champion kuralı, guardrail, tie-break |
| `src/lifecycle/drift.py` | drift izleme / retrain tetikleyici (henüz yok) | eşikler, tetikleme koşulu, yeni veri penceresi |
| `src/lifecycle/quality.py` | `quality_gates`, `evaluate_quality_gates` | ek metrik ve eşik mantığı |

Gerekenler:

1. Core'da karar fonksiyonlarını **enjekte edilebilir** yap: `compare_registered_local_models(..., decision=default_decision)`.
   `decision`, `(improvement, guardrail_deltas, quality_passed, policy) -> (eligible, reason)` imzalı olur.
2. `_project_files.py` hook dosyalarını mevcut düzen ve eski düzen için bulur
   (`build_preprocessing` ile aynı yöntem); yoksa core varsayılanı kullanılır.
3. Hook'un **kaynak digest'i** karşılaştırma raporuna ve `candidate_comparison.json`
   kanıtına yazılır (`decision_source_sha256`). Varsayılan hook için `null`.
4. Kullanıcı hook'u `reason` olarak yalnızca bilinen kodlardan birini döndürebilir
   (`promotion.py::_validate_report_approval` ve UI açıklamaları bu kümeye dayanır);
   bilinmeyen kod hata verir.
5. Şablon: `skyulf-core/templates/databricks/template/{{.project_name}}/src/lifecycle/*.py.tmpl`
   ve `README.md.tmpl` içinde "ne zaman değiştirilir" notu.
6. Doğrulama: `src/tools/smoke.py` hook'ları import etmeden de çalışan statik kontrol
   (`project_checks.py` kuralı: "never import or execute project hooks") ile imza ve reason
   kümesini denetler.

## 3. Aşama 2: tam kopya (ihtiyaç doğarsa)

- Init seçeneği `vendor_integrations` (varsayılan `false`) → `src/skyulf_integrations/`.
- Job'larda wheel'den **önce** `sys.path`'e eklenir.
- `UPSTREAM.json`: core sürümü + dosya başına sha256.
- `tools/sync_integrations.py`: core'dan yeniler, elle değişen dosyaları diff ile listeler,
  üzerine yazmadan önce onay ister.
- Run ve model tag'leri: `integrations_source=wheel|vendored`, `integrations_modified=true|false`.
- `smoke.py`: kopyanın core API sürümüyle uyumunu doğrular.
- `INTERNAL_API.md` kuralı geçerli kalır: kopyada da alias/forwarding yok.

## 4. Aşama 2'ye geçiş kriterleri

Şunlardan en az biri doğruysa Aşama 2'yi aç:

- Bir müşteri ihtiyacı hook ile çözülemeyen bir akış değişikliği gerektiriyor
  (ör. faz eklemek/çıkarmak, alias protokolünü değiştirmek).
- Wheel build'i olmadan Databricks'te hızlı yama gerekiyor.
- Aynı özelleştirme ≥ 3 projede hook dışına taşmış.

Aksi halde Aşama 1 yeterlidir.

## 5. Sıra

1. Guardrail kuralı (`GUARDRAIL_PLAN.md`) ve büyük dosyaların bölünmesi
   (`INTEGRATIONS_MAP.md` §5, öneri 3–4).
2. Karar fonksiyonunu enjekte edilebilir yap (core, geriye uyumlu).
3. Hook şablonları + digest kanıtı + smoke/preview kontrolleri.
4. Drift hook'u (drift işi merge edildikten sonra).
5. Aşama 2 kararı (§4).

## 6. Riskler

| Risk | Önlem |
|---|---|
| Elle değişen karar kodu kanıtı bozar | Hook digest'i kanıta yazılır; reason kümesi kapalı |
| Kullanıcı hook'u güvenlik riski | Mevcut "trusted project Python" modeliyle aynı; statik kontrol import etmez |
| Core ve Bundle sürüm kayması | Şablon wheel sürümünü pinler; hook imza sürümü (`HOOK_API_VERSION`) kontrol edilir |
| Frontend/Bundle wizard senkronu | Hook varlığı wizard'da bilgi alanı; seçenek listeleri node-config kuralıyla karşılaştırılır |
