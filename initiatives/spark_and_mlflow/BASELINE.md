# SM-00 — Güncel baseline ve Spark runtime kanıtı

Tarih: 2026-09-21. Başlangıç HEAD:
`2397c11648c27645a4ee332412a503c394c1c2f2`. Değişiklikler uncommitted.
Başlangıç sürümü 0.8.20; son geliştirme sürümü 0.9.0, Unreleased.

Çalışma ağacında önceden .gitignore değişiklikleri, initiative silmeleri ve
untracked plan/test klasörleri vardı; bunlar geri alınmadı. Worktree yaratma
`.git/refs/heads` yazma izni nedeniyle başarısız oldu; bu görev mevcut dizinde
dar kapsamla yürütüldü. Commit/push/release yapılmadı.

## İlk checkout'un kaynak davranışı

- EngineRegistry default pandas, sınıf değişkeni; pandas/Polars input detection var.
- Spark enum/engine henüz yok. SM-00 extra yalnız runtime dependency hazırlığı.
- Calculator.fit ve Applier.apply, engine-keyed dispatcher kullanıyor.
- FeatureEngineer fit_transform/transform mevcut local API; Spark girişi yok.
- SkyulfPipeline save/load ve core serializer seam aynı şey sayılmamalı;
  native state taşınması henüz uygulanmadı.
- Eski envanterdeki 0.8.24/Polars-default sonuçları bu baseline'a taşınmadı.

## Ortamlar

| Paket | Mevcut .venv | Ayrı .venv-spark |
| --- | --- | --- |
| Python | 3.12.10 | 3.12.10 |
| pandas | 2.3.2 | 2.3.2 |
| Polars | 1.44.1 | 1.44.2 |
| NumPy | 1.26.4 | 1.26.4 |
| PyArrow | 24.0.0 | 24.0.0 |
| scikit-learn | 1.8.0 | 1.9.1 |
| pytest | 9.1.1 | 9.1.1 |
| PySpark | kurulu değil | 4.0.3 |
| MLflow | kurulu değil | kurulu değil |

Spark test hedefi Python 3.12 + Spark 4.0.3 + Java 17. Runtime uyumu için
[Spark 4.0.3 kurulum belgesi](https://spark.apache.org/docs/4.0.3/api/python/getting_started/install.html)
esas alındı. Test edilen Java: Eclipse Temurin 17.0.20.1+1.
JDK resmi [Adoptium API](https://adoptium.net/installation/ci-scripts) üzerinden
indirildi ve API'nin SHA-256 değeriyle doğrulandı. Konumu
`.cache/spark-jdk/jdk-17.0.20.1+1`; sistem PATH/JAVA_HOME ayarları değiştirilmedi.
Bu kurulum Databricks Runtime veya Connect desteği iddiası değildir.

## Tekrar çalıştırma

```powershell
uv venv .venv-spark --python .venv/Scripts/python.exe --no-python-downloads
uv pip install --python .venv-spark/Scripts/python.exe -r requirements-spark.txt
$env:JAVA_HOME = (Resolve-Path '.cache/spark-jdk/jdk-17.0.20.1+1').Path
$env:SKYULF_REQUIRE_SPARK = '1'
.venv-spark/Scripts/python.exe -m pytest skyulf-core/tests/spark -q --tb=short -o addopts= -p no:cacheprovider --basetemp .cache/sm00-spark-final
```

Venv zaten varsa ilk komut tekrar çalıştırılmaz. JAVA_HOME yeni makinedeki Java 17
konumuna göre ayarlanır. Bu env değerleri yalnız terminal sürecinde geçerlidir.
Repo pytest ayarları için requirements-spark pytest-asyncio da içerir.
Sonuç: **2 passed in 7.23s**, JVM alt süreçleri başarıyla kapandı.

İlk sandbox denemesi 2 passed ardından PySpark'ın Windows taskkill temizliğinde
Access denied yazdı. İzinli tekrar temiz bitti; son process kontrolünde Skyulf
Spark Java süreci bulunmadı. Son test pytest config uyarısı da üretmedi.

## Baseline komutu

```powershell
.venv/Scripts/python.exe -m pytest skyulf-core/tests/unit/test_core_serialization.py skyulf-core/tests/unit/test_core_model_registry.py skyulf-core/tests/unit/test_artifact_shapes.py skyulf-core/tests/unit/test_engines_registry.py skyulf-core/tests/unit/test_engines_sklearn_bridge.py skyulf-core/tests/unit/test_engine_parity.py skyulf-core/tests/integration/test_scaling.py skyulf-core/tests/integration/test_pipeline_integration_preprocessing.py -q --tb=short -o addopts= -p no:cacheprovider --basetemp .cache/sm00-baseline
```

**206 passed, 5 warnings in 4.47s.** Warning'ler all-NaN scaler fixture'larının
mevcut sklearn/NumPy hesaplamalarından. İlk denemelerde iki test dosyası adı
güncel checkout'ta yoktu; doğru adlar keşfedilip komut düzeltildi. Sistem temp
klasörünün erişim hatası repo içi basetemp kullanılarak giderildi. Bunlar ürün
regresyonu olarak kaydedilmedi. Tam core suite bu görevde tekrar çalıştırılmadı.

## Negatif test ve statik kontroller

- Yeni smoke testleri fixture eklenmeden önce beklenen missing-fixture hatası verdi.
  Bu test altyapısı bootstrap kontrolüdür; native node TDD kanıtı değildir.
- Base .venv, `SKYULF_REQUIRE_SPARK` yok: 2 skipped.
- Base .venv, `SKYULF_REQUIRE_SPARK=1`: 2 beklenen setup error, exit 1.
- `ruff check skyulf-core/tests/spark skyulf-core/setup.py`: geçti.
- `ruff format --check skyulf-core/tests/spark skyulf-core/setup.py`: geçti.
- `ty check skyulf-core/tests/spark --python .venv-spark`: geçti.
- CI workflow YAML parse edildi; dedicated lane require flag'i doğrulandı.
- Root pyproject, core setup ve frontend package/lock sürümleri 0.9.0 olarak
  eşleşiyor. `npm run sync-version` ve `npm run check-version` başarılı.
- `npm run build`: tsc + Vite başarılı (20.43s); circular chunk ve empty
  vendor-react chunk uyarıları var. İlk sandbox denemesinin üst dizin erişim
  engelinden sonra izinli build tamamlandı; generated static çıktı yeniden üretildi.
- `git diff --check`: başarılı. 0.9.0 changelog Unreleased; yayın/commit yok.

SM-00 yalnız test foundation kapısıdır. Spark node'ları, portable state, inference
bundle ve MLflow adapter'ları sonraki görevlerdir. Yeni GitHub Actions workflow'u
yazıldı; uzak CI çalıştırılmadı. Sonraki READY görev: SM-01.

## Master güncellemesi sonrası yeniden doğrulama

Kullanıcının talimatıyla origin fetch edildi; 090 ve yerel master
`97536eae20e5422220f9824bc591ceb05576ee50` commit'ine fast-forward edildi.
Bu commit 0.8.24 PR #175 merge'idir. Başlangıçta yerel master 70 commit gerideydi.
Stash üzerinden çalışma korundu; version conflict'leri 0.9.0 olarak çözüldü.
Yeni tabanda Polars default ve ContextVar engine scope tekrar mevcut.

Yukarıdaki baseline komutuna `skyulf-core/tests/unit/test_engine_context.py`
eklenerek `.cache/sm00-upstream-baseline` basetemp'iyle tekrar çalıştırıldı:
**231 passed, 5 warnings (3.26s)**. Spark lane:
`.cache/sm00-upstream-spark` basetemp'iyle **2 passed (7.38s)**, temiz JVM kapanışı.
Tam repo ty check ve scoped Ruff geçti. `mkdocs build --strict` yeni Spark
kullanıcı rehberiyle başarılı. Spark optional imports test fixture'da lazy
importlib yüklemesiyle çözülür; base ortamda PySpark zorunlu hale getirilmez.
