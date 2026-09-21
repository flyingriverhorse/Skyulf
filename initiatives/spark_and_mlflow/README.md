# Spark ve MLflow initiative

Tarih: 2026-09-21. Durum: **SM-00–SM-05 tamamlandı; sıradaki görev SM-06.**
Hedef sürüm [0.9.0 — Unreleased](../../changelog/0.9.x.md).
Geliştirme sürümü 0.9.0; yayın yapılmadı. [Güncel baseline](BASELINE.md).

Amaç: Skyulf Core'un mevcut pandas/Polars davranışını koruyarak gerçek Spark
feature engineering, dağıtık Python inference ve opsiyonel MLflow/Unity Catalog
entegrasyonu sağlamak. Aylık batch teslimatından sonra endpoint'ler, en son
template/Databricks Bundle gelir.

## Okuma ve uygulama sırası

1. [OPEN_QUEUE.md](OPEN_QUEUE.md): kısa, sıralı görev kuyruğu ve sonraki iş.
2. [ARCHITECTURE.md](ARCHITECTURE.md): kapsam, değişmez kurallar ve dosya sınırları.
3. [01-core-spark-plan.md](01-core-spark-plan.md): SM-00–SM-07, engine ve fit/apply.
4. [02-inference-plan.md](02-inference-plan.md): SM-08–SM-11, iki inference yolu.
5. [03-mlflow-batch-delivery-plan.md](03-mlflow-batch-delivery-plan.md):
   SM-12–SM-20, MLflow, registry, batch ve son teslimatlar.
6. [VALIDATION.md](VALIDATION.md): test matrisi, ortamlar ve tamamlanma kanıtı.

Tarihsel araştırma:

- [Hazırlık incelemesi](reports/2026-09-21-spark-databricks-readiness.md)
- [Kaynak/node/test envanteri](reports/2026-09-21-spark-core-inventory.md)

Bu iki rapor `initiatives/reports/` içinden taşındı. Rapordaki sayılar ve test
sonuçları raporun incelediği commit'e aittir; güncel dalın sonucu değildir.
Plan hazırlanırken HEAD `2397c11648c27645a4ee332412a503c394c1c2f2` idi.
Özellikle güncel `engines/registry.py` pandas varsayılanı ve sınıf değişkeni
kullanıyor; eski rapordaki Polars/ContextVar ifadesi güncel varsayım yapılmamalı.
SM-00 mevcut davranışı yeniden kaydeder. Bu initiative global varsayılan engine'i
değiştirmeyi gerektirmez; ileride wizard önerisi ayrı bir kullanıcı tercihidir.

Güncelleme: 090 ve yerel master 0.8.24 merge noktası
`97536eae20e5422220f9824bc591ceb05576ee50` üzerine ilerletildi. Yukarıdaki
pandas-default notu önceki checkout'u anlatır; güncel kaynak Polars/ContextVar
davranışını geri içeriyor. Güncel tekrar: 231 baseline + 2 Spark smoke geçti.
Kullanıcı belgeleri: [Spark rehberi](../../docs/user_guide/spark.md).

## İlk teslimat

Kolon seçimi → SimpleImputer(mean/constant) → StandardScaler → kayıt/yükleme →
Spark apply. Sonra aynı feature'larla mevcut bir Python modelinin Spark worker
tahmini. Spark verisini driver'a indirmek hiçbir aşamanın gizli çözümü değildir.
Median/quantile, kategorik state ve zaman pencereleri ilk teslimattan ayrı tutulur.

## Çalışma kuralları

- Sonraki görev **SM-06**. Bir sonraki satıra geçmeden önce bağımlılıklarını bitir.
- Her görevde test → beklenen başarısızlık → dar uygulama → test/gate kanıtı.
- Yeni testler docstring ve gerçek assertion içerir. Uygulama sırasında kaynak
  yollarını yeniden doğrula; planlanan API'ler bugün mevcut API gibi kullanılmaz.
- Dosya yazılmış olması, testlerin geçmesi ve Databricks'te çalışması ayrı durumlar.
- Queue'da DONE için komut, sonuç, runtime ve commit/çalışma ağacı kaydı gerekir.
- Commit/push bu planın otomatik adımı değildir; yalnız kullanıcı istediğinde
  güncel kontrollerle DCO commit hazırlanır.
- Önce core/SDK; backend/Canvas değişikliği SM-18. Endpoint SM-19,
  template SM-20. Streaming ilk release'in zorunlu koşulu değildir.

## Diğer initiative'lerle ilişki

[Önceki MLflow planı](../analysis/skyulf-core-mlflow-integration-plan.md)
tarihsel girdidir. Bu çalışma için sıra ve karar kaynağı burasıdır: global fit
callback veya backend job'a zorunlu bağlı tracking yerine açık run kapsamı;
ilk teslimatta UI yok. Joblib/ONNX, Ray ve deep-learning planları bu çalışmanın
önkoşulu değildir. Paylaşılan artifact/pipeline dosyalarındaki eşzamanlı
değişiklikler her görev başında kontrol edilir.
