# Spark ve MLflow — Open Queue

Güncelleme: 2026-09-21. **SM-00 tamamlandı. Sonraki görev: SM-01. Hedef: 0.9.0.**
Bu dosya kısa çalışma sırasıdır; detaylar bağlantılı planlarda.

Durumlar: READY = başlanabilir; WAIT = önceki görev bekleniyor;
LATER = son aşama; ACTIVE = yürütülüyor; BLOCKED = somut dış engel;
DONE = kanıtla tamamlandı. Şu an ACTIVE görev yok; SM-01 başlanabilir.

| Sıra | Görev | Bağımlılık | Durum | Kısa bitiş ölçütü |
| --- | --- | --- | --- | --- |
| SM-00 | Baseline ve Spark test ortamı | — | DONE | Güncel master: 231 baseline; 2 Spark smoke; [kanıt](BASELINE.md) |
| SM-01 | Execution/capability kuralları | SM-00 | READY | Desteklenmeyen engine/operation erken ret |
| SM-02 | Spark engine ve conversion sınırı | SM-01 | WAIT | Spark tanınır; gizli local conversion yok |
| SM-03 | Dispatcher, schema, row keys | SM-02 | WAIT | Key/target korunur; local API çalışır |
| SM-04 | Versioned portable state | SM-03 | WAIT | Limit/version kontrolü; state round-trip |
| SM-05 | Native SimpleImputer | SM-04 | WAIT | Mean/constant train-only fit, Spark apply |
| SM-06 | Native StandardScaler | SM-05 | WAIT | Population variance ve flag parity |
| SM-07 | Uçtan uca FE kapısı | SM-06 | WAIT | Üç engine çapraz fit/apply; kayıt/yükleme |
| SM-08 | Ortak inference bundle | SM-07 | WAIT | Raw/features ayrımı; model metadata round-trip |
| SM-09 | Native FE + worker model | SM-08 | WAIT | Spark tahmini local ile key bazında aynı |
| SM-10 | Worker Python FE + model | SM-09 | WAIT | Batch-safe pipeline; window gibi yollar açık ret |
| SM-11 | Classification ve ölçek kapısı | SM-10 | WAIT | Class/proba/threshold parity; worker kanıtı |
| SM-12 | Opsiyonel MLflow tracking | SM-11 | WAIT | Off bağımsız; gerçek run lifecycle ve izolasyon |
| SM-13 | MLflow model packaging | SM-12 | WAIT | Temiz ortamda pyfunc yükleme ve parity |
| SM-14 | Registry/Unity Catalog adapter | SM-13 | WAIT | URI/signature; alias → sabit version |
| SM-15 | Aylık batch + Delta sink | SM-14 | WAIT | Dönem izolasyonu, retry ve conflict kontrolü |
| SM-16 | Gerçek Databricks kapısı | SM-15 | WAIT | Job/run kanıtı; UC load; iki inference yolu |
| SM-17 | Kalan node aileleri | SM-16 | WAIT | Aile bazlı port; her kayıt supported/unsupported |
| SM-18 | Backend/Canvas ve custom FE | SM-17 | LATER | Capability UI/API; DAG/artifact uyumu |
| SM-19 | HTTP/SQL erişimi; optional streaming | SM-18 | LATER | Desteklenen erişimlerde batch ile aynı tahmin |
| SM-20 | Template / Databricks Bundle | SM-19 | LATER | Seçilebilir working runner; validate ve smoke |

## Detay planlar

- SM-00–SM-07: [Core/Spark](01-core-spark-plan.md)
- SM-08–SM-11: [Inference](02-inference-plan.md)
- SM-12–SM-20: [MLflow/batch/son teslimatlar](03-mlflow-batch-delivery-plan.md)
- Ortak kurallar: [Mimari](ARCHITECTURE.md), [Doğrulama](VALIDATION.md)

SM-17 başlangıcında 17a column/cast/date, 17b quantile/category,
17c group/window, 17d OOF alt görevlerine ayrılır. Her alt görevin kendi
test/review kapısı vardır. Tüm node'ların native olması zorunlu değildir;
gerçekten desteklenen kapsam ilan edilerek G5 kapatılır.
SM-19 streaming isteğe bağlıdır; HTTP/SQL bittiğinde streaming unsupported
olarak açıkça kaydedilebilir, template'i belirsiz süre bloke etmez.

## Bir görevi kapatma kaydı

Her DONE satırına aynı dosyada aşağıdaki bilgileri içeren tarihli kayıt ekle:

```text
Task ID / tarih / incelenen commit / değişiklik commit'i veya uncommitted
Değişen dosyalar ve sağlanan davranış
Çalıştırılan exact komutlar / passed-failed-skipped / runtime sürümleri
Olumsuz senaryo kanıtı (unsupported, leakage, retry vb.)
Bilinen sınırlamalar / sonraki görev
```

Komutlar çalıştırılmadan kutular işaretlenmez. Spark lane'inde tamamı skip olmuş
suite DONE kanıtı değildir. Platform erişimi gerektiğinde BLOCKED nedeni somut
olarak yazılır; yerel test sonucu gerçek UC/Databricks sonucu yerine geçmez.

## Devam oturumu için kısa talimat

Önce README, mimari ve bu kuyruğu oku. Güncel git durumunu ve ilgili kaynakları
doğrula; kullanıcı değişikliklerini koru. İlk READY görevi ACTIVE yap, kendi
test döngüsüyle tamamla ve kanıtı yaz. Sonraki bağımlılığı aç. Endpoint veya
template'e önden başlama. SM-00 test/runtime hazırlığıdır; native node uygulaması
ve MLflow entegrasyonu henüz yok.

## SM-00 — 2026-09-21 tamamlanma kaydı

- Başlangıç HEAD: `2397c11648c27645a4ee332412a503c394c1c2f2`; değişiklikler uncommitted.
- Test fixture/smoke, optional extra/requirements, Spark CI lane ve 0.9.0
  Unreleased changelog oluşturuldu. Detaylı ortam ve komutlar [BASELINE.md](BASELINE.md).
- Local baseline: 206 passed, 5 mevcut numeric warning.
- PySpark 4.0.3 + Temurin Java 17.0.20.1 + Python 3.12.10: 2 passed, temiz kapanış.
- Spark olmayan base ortam: 2 skipped; zorunlu Spark modu: beklenen 2 setup error,
  exit 1. Eksik dependency başarılı CI gibi raporlanmıyor.
- Ruff check/format ve scoped ty geçti. GitHub Actions ve Databricks çalıştırılmadı.
- Sonraki görev SM-01; bu kanıt native Spark node desteği anlamına gelmez.
