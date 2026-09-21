# Doğrulama ve release kapıları

Bu belge gelecek uygulamanın kabul ölçütlerini tanımlar. Tarihsel rapordaki
9.815 passed / 68 skipped / 1 failed sonucu bu planın test sonucu değildir.
O çalışmadaki ağ bağımlı embedding testi offline tek tekrarında geçmişti;
güncel baseline SM-00'da yeniden ölçülür.

## Ortam matrisi

| Lane | Amaç | Geçme koşulu |
| --- | --- | --- |
| Base Python | Spark/MLflow olmadan mevcut core | Import + local regression |
| OSS Spark local[2] | Gerçek partition/action ve parity | Spark suite çalışır, toplu skip yok |
| MLflow local store | Run/model/registry lifecycle | Gerçek save/load; subprocess packaging |
| Delta integration | Atomik dönem yayını | Gerçek tablo, retry ve conflict |
| Databricks seçili runtime | UC/worker/wheel/job entegrasyonu | SM-16 platform evidence |
| Connect/serverless | İsteğe bağlı ortam profili | Ayrı smoke; başka lane sonucu yeterli değil |

OSS Spark, DBR ve Connect ayrı dependency profilleri olarak ele alınır. DBR'ın
runtime paketlerini OSS test extra'sıyla körlemesine değiştirme. Sürüm sabitleme
SM-00/SM-16 çıktısıdır. Python >=3.12 mevcut paket gereksinimidir.

## Davranış matrisi

| Sözleşme | Test verisi ve assertion |
| --- | --- |
| Train-only state | Train [1,null,3], held-out [100,null]; fill 2 |
| StandardScaler | [1,3] → mean=2, var=1, scale=1; dört flag kombinasyonu |
| Null/NaN | All-null, NaN, constant, empty; açık local semantiği |
| Kolon/dtype | Farklı sıra, missing/extra, dot/backtick; silent coercion yok |
| Row identity | Repartition 1/2/7, shuffle; key eşleşmesi; null/duplicate ret |
| State transport | 3 fit engine × 3 apply engine; supported operation'lar |
| Persistence | Codec/bundle round-trip; eski pickle local regression |
| Native execution | FE apply'da Python UDF/ham data collect yok |
| Distributed Python | Batch size 2/17/default; aynı key için aynı tahmin |
| Temporal | Rolling [1..6] chunk karşı örneği; generic worker yolunda ret |
| Classification | String/multiclass, classes_ sırası, proba, threshold precedence |
| Lifecycle | Worker fit yok; state mutation yok; concurrent run id isolation |
| MLflow optional | Paket yokken disabled çalışır; enabled açık dependency hatası |
| Registry | Alias run başında sabitlenir; arada alias değişimi sonucu etkilemez |
| Batch output | Aynı ay retry, başka ay korunur, empty guard, concurrent conflict |
| Packaging | Temiz subprocess/worker wheel; repo cwd bağımlılığı yok |

Float tolerance fixture'ın sayısal ölçeğine göre gerekçelendirilir; küçük FE
fixture'larında rtol=1e-10/atol=1e-12. Category/key/schema için exact equality.
Row sırasına göre list equality Spark kanıtı değildir.

## Materialization kanıtı

Static toPandas araması tek başına yeterli değildir. Üretim input'unda collection
metodlarını guard eden test, gerçek Spark plan ve ölçek ölçümü birlikte kullanılır.
Aggregate state toplama izinli ve state_max_bytes limitlidir. Testin küçük
beklenen sonucunu collect etmesi üretim driver fallback'i sayılmaz. Native FE
planında Python UDF olmamalı; model inference planında Python yürütmesi beklenir.
Yürütme testleri action çağırmalı; erken ret testleri action olmadığını doğrulamalı.

## Komutlar

Önce görev dosyasındaki dar pytest komutu. Interpreter ilgili lane'in venv'i
olmalı; `.venv` ile `.venv-spark` karıştırılmaz.

```powershell
.venv-spark/Scripts/python.exe -m pytest skyulf-core/tests/spark -q
.venv/Scripts/python.exe -m pytest skyulf-core/tests -q
.venv/Scripts/python.exe -m ruff check skyulf-core/skyulf skyulf-core/tests
.venv/Scripts/python.exe -m ruff format --check skyulf-core/skyulf skyulf-core/tests
```

Type gate: repo'nun güncel ty check kapsamını uygula; optional modüllerde paket
yokken type import hatalarını maskelemeden dependency lane'i ayır. Backend
değişirse ilgili root integration testleri; frontend değişirse typecheck, vitest,
eslint ve build. Yalnız Markdown planı için uygulama suite'i tekrar çalıştırmak
veya mkdocs build yapmak gerekmez; doküman linkleri doğrulanır.

## Kapılar

- G1 / SM-07: native FE ve portable state; local API regresyonu yok. **Tamamlandı
  (2026-09-21):** 261 Spark lane testi; 9969 local geçti, 223 skip.
  [Komutlar ve sınırlar](OPEN_QUEUE.md).
- G2 / SM-11: iki inference yolu + classification + worker/ölçek kanıtı.
- G3 / SM-14: opsiyonel MLflow model/run/registry; UC gerçek kanıtı G4'te.
- G4 / SM-16: gerçek Databricks job + UC load + idempotent Delta batch.
- G5 / SM-17: node support matrisi gerçeğe uygun; unsupported açık.
- G6 / SM-18: backend/Canvas supported akışlar ve legacy artifact uyumu.
- G7 / SM-19: endpoint/SQL parity; optional streaming ayrı sonuç.
- G8 / SM-20: template validation ve gerçek deploy/run kanıtı ayrı.

Failure nedenini ürün hatası, ortam eksikliği veya önceden var olan failure
olarak kanıtla kaydet. Skip'i pass gibi yazma. Yanlış tahmin, leakage veya satır
kaybı varsa ilgili capability kapalı kalır.
