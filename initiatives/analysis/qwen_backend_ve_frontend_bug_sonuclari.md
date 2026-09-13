Beş ajanın hepsi döndü ve en kritik 7 yeni bulguyu kendim doğruladım. HEAD bu arada yine ilerledi: 8053a84c.

   Kapsam güncellemesi
   ┌────────────────────────┬───────────────────────────────────┬──────────────────────────────────────────────────────────────────────┐
   │ Katman                 │ 1. tur                            │ Şimdi                                                                │
   ├────────────────────────┼───────────────────────────────────┼──────────────────────────────────────────────────────────────────────┤
   │ skyulf-core/skyulf/    │ ~%95                              │ ~%95                                                                 │
   │ (38.6k satır)          │                                   │                                                                      │
   ├────────────────────────┼───────────────────────────────────┼──────────────────────────────────────────────────────────────────────┤
   │ backend/ (30.0k satır) │ ~%50-60                           │ ~%90 — ml_pipeline (16.7k) derin + monitoring 2.1k satırın tamamı,   │
   │                        │                                   │ database, data_ingestion, data/catalog, realtime, health, config,    │
   │                        │                                   │ middleware, exceptions, eda                                          │
   ├────────────────────────┼───────────────────────────────────┼──────────────────────────────────────────────────────────────────────┤
   │ frontend/src/ (66.0k   │ ~%25                              │ ~%70 — components/eda (16 sekme), components/pages                   │
   │ satır)                 │                                   │ (Experiments/Inference/Deployments), pages/drift, modules/nodes (tüm │
   │                        │                                   │ ayar panelleri), pipelineConversion/                                 │
   ├────────────────────────┼───────────────────────────────────┼──────────────────────────────────────────────────────────────────────┤
   │ Hâlâ sıfır             │ testler (644 dosya), docs/ (82),  │ aynı                                                                 │
   │                        │ CI (13), Docker/build, bağımlılık │                                                                      │
   │                        │ tutarlılığı, src/styles/          │                                                                      │
   └────────────────────────┴───────────────────────────────────┴──────────────────────────────────────────────────────────────────────┘
   Toplam: ~115 doğrulanmış bulgu (57 + 58 yeni).

   ---

   YENİ KRİTİK BULGULAR

   58. handle_success job'ın metriklerini yanlış node'dan alıyor; sarkan bir yaprak tüm eğitim sonuçlarını çöpe atıyor — strategies.py:127
   (kendim doğruladım)
    last_node_id = list(result.node_results.keys())[-1]
   node_results topolojik sırada dolduruluyor ama Kahn kuyruğu ham canvas liste sırasıyla tohumlanıyor. PARTITION_TERMINAL_STEP_TYPES sadece
   training/tuning/data_preview'ı terminal sayıyor, yani split'a bağlı ama devamı gelmemiş bir scaler/DatasetProfile node'u partition
   edilmiyor ve çalışıyor. Canvas'ta o node trainer'dan sonra listeleniyorsa son sırada yer alıyor ve metrikleri job'ın metrikleri oluyor.
   Çalıştırılmış kanıt:
    job.metrics          = {'steps': {'step': {'details': {'scaled_columns': ['a']}}}, ...}
    job.best_score       = None      job.best_params = None
    job.results          = None      job.scoring     = None
    job.tuned_thresholds = None      job.thresholds_enabled = False
   Job completed / progress 100 görünüyor. Sonuç: Experiments satırı boş, tuning geçmişi göremiyor, _seed_tuned_thresholds hiçbir eşik
   tohumlamıyor → deployment varsayılan karar kuralına düşüyor, yani threshold tuning o job için sessizce devre dışı. Sıraya bağımlı olduğu
   için heisenbug gibi görünüyor: aynı graf node'ları farklı sırayla dizilince çalışıyor.

   59. PATCH /monitoring/jobs/{id}/description açıklamayı sessizce çöpe atıyor — monitoring/router.py:189-195 (kendim doğruladım)
    meta_raw = cast(dict, row.job_metadata or {})   # doluysa AYNI nesne
    meta_raw["description"] = body.description      # yerinde mutasyon
    row.job_metadata = cast(Any, meta_raw)          # aynı nesne geri atanıyor
   job_metadata düz JSON kolonu (MutableDict değil), o yüzden SQLAlchemy flush'ta current == committed_state karşılaştırıyor — aynı nesne →
   "değişmedi" → UPDATE yok. Endpoint 200 {"status":"ok"} dönüyor, session içi nesne yeni değeri gösteriyor (yani naive unit test de geçiyor),
    expire_on_commit sonrası değer kayboluyor. Sadece job_metadata dolu olan job'larda — yani her branch run'ında. Aynı dosyada
   update_drift_alert_disposition:788-793 doğru deseni kullanıyor (history = [*history, {...}]).

   60. AWS secret access key HTTP yanıt gövdesinde düz metin dönüyor ve error_events'a kalıcı yazılıyor — data_ingestion/service.py:406,529 +
   exceptions/handlers.py:99-108 (kendim doğruladım)
    raise SkyulfException(message=f"Database error: {str(e)}") from e
   SQLAlchemy StatementError'ın str()'i [SQL: ...] ve [parameters: (...)]] içeriyor; hide_parameters repo'da hiçbir yerde ayarlanmamış (grep:
   engine.py'de 0). S3 kaynağında bağlı parametre, storage_options.aws_secret_access_key dahil çağıranın JSON'u. SkyulfException.status_code
   varsayılanı 500 ve handler exc.message'ı redaksiyonsuz gövdeye koyuyor — main.py:482 ve middleware/error_handler.py:62-73
   redact_credentials çalıştırıyor, bu yol çalıştırmıyor (handlers.py'de 0 eşleşme). ≥500 olduğu için _record_error aynı metni
   error_events.message/traceback'e yazıyor, oradan da auth'suz GET /monitoring/errors servis ediyor. Çalıştırılmış kanıt: yanıt gövdesinde
   hem secret access key hem access key ID bulundu; mevcut redact_credentials ikisini de yakalardı, sadece çağrılmıyor. Herhangi bir DB
   aksaklığı (kilitli dosya, dolu disk, read-only mount) kalıcı secret sızıntısına dönüşüyor.

   61. Decomposition ağacı seçilen ölçüyü değil satır sayısını gösteriyor — components/eda/tabs/DecompositionTab.tsx:14 (kendim doğruladım)
    useState<string>('count')      // başlangıç
    Aggregation <select> seçenekleri: ["sum","mean","min","max"]   // 'count' YOK
    istek argümanları: [101,"revenue","count","",[]]
   Kullanıcı Analyze: revenue seçiyor, Aggregation dropdown'ı beliriyor, ama measureAgg hiç güncellenmiyor → istek UI'ın gösteremediği bir
   değerle gidiyor. Core sessizce düşüyor: decomposition.py:140 → agg_exprs.get(measure_agg, pl.len()). Ağaçta revenue diye etiketlenmiş satır
    sayıları render ediliyor; her değer ve oran yanlış büyüklük, hata yok.

   62. Tek bir sonlu-olmayan değer tüm EDA sayfasını çökertiyor — tabs/OutliersTab.tsx:59,61 + tabs/GeospatialTab.tsx:105,109

   OutlierPoint.explanation list[dict[str,Any]] | None — tipi garanti etmiyor, orjson sonlu olmayanı null yapıyor, ve .toFixed(2) doğrudan
   çağrılıyor. Çalıştırılmış: TypeError: Cannot read properties of null (reading 'toFixed'). Sekme başına ErrorBoundary yok (sadece App.tsx:26
    ve MainLayout.tsx:130, sayfa seviyesi) → tek sekme değil tüm EDA sayfası error fallback'e düşüyor. Aynı desen Geospatial'da:
   min_lat.toFixed(4) korumasız, ama 119. satırda centroid_lat?.toFixed(4) ?? '—' var — yazar biliyormuş, sadece birini korumuş.

   63. Kategorik drift sparkline'ı düz yeşil "stabil" çizgi çiziyor, aynı satır "Drifted" derken — pages/drift/_hooks/useDriftHistory.ts:35
   (kendim doğruladım)
    result[col].push(data.psi ?? 0);
   Kategorik kolonlarda backend summary'sinde psi yok (router.py:397 → metrics_map.get("psi")), core sadece psi_categorical üretiyor. null ??
   0 → her kontrol için uydurma 0.0. Sparkline.tsx:24 son değere göre renklendiriyor → #22c55e yeşil. Çalıştırılmış kanıt:
    region sparkline = [0,0,0]  nokta rengi #22c55e (yeşil, düz çizgi)
    age    sparkline = [0.02,0.05,0.48]  nokta rengi #ef4444 (kırmızı, doğru)
   Ama DriftTable.tsx:253-255 'psi' || 'psi_categorical' çözüyor → aynı satırda PSI hücresi 0.4800 kırmızı kalın, Status "Drifted". Her drift
   raporundaki her kategorik özellik etkileniyor. FE-7'den farklı dosya/farklı artefakt.

   64. null tahminler inference istatistiklerinde 0.0 oluyor — inference/inferenceData.ts:257-260 (kendim doğruladım)
    .map(p => (typeof p === 'number' ? p : Number(p))).filter(n => Number.isFinite(n))
   Number(null) === 0 ve Number.isFinite(0) === true → null'ar filtrelenmiyor, tam 0.0'a çevrilip sayılıyor. Çalıştırılmış:
   toNumericArray([1,null,3]) = [1,0,3], mean 1.333 (gerçek iki değerin mean'i 2.0). Tüketici predictionStats → mean/min/max +
   PredictionHistogram: tek bir null satır min'i 0.00'a çekiyor ve histogramın sol kenarına hayalet bar ekliyor. Modelin çıktısı hakkında
   gerçek diye sunuluyor.

   65. Advanced (tuned) mod kullanıcının hiperparametrelerini sessizce çöpe atıyor — pipelineConversion/training.ts:52-62 +
   ModelConfigSection.tsx:87 (kendim doğruladım)

   Advanced dalı run_mode, target_column, algorithm, execution_mode, tuning_config üretiyor — hyperparameters hiç yok. Karşılaştırma:
   ensemble.ts:165-184 her iki modda da tam yapıyı iletiyor, yani bu training-node'a özgü. Üstelik model dropdown'ı onChange({...config,
   model_type: v, search_space: {}}) yapıyor — arama uzayını sıfırlıyor. ClassificationNode/RegressionNode kendi validate'ini tanımlamadığı
   için nodeFactory.ts:64-67'nin varsayılanı (sadece target_column) geçerli; boş search space işaretlenmiyor. Çalıştırılmış:
    ADV validate(empty search_space) = {"isValid":true}
    ADV payload = {"run_mode":"tuned", ... "tuning_config":{... "search_space":{}}}
   Fixture'daki hyperparameters: {n_estimators:400, max_depth:7} payload'da tamamen yok. Backend tarafı da teyitli: _node_runners.py:639'da
   test_empty_search_space_silently_ignores_user_hyperparams adlı bir teste atıf var. Kullanıcı Basic modda 400/7 ayarlıyor, Advanced'a
   geçiyor, modeli tekrar seçiyor, n_trials=50 veriyor ve çalıştırıyor → job tuning run olarak raporlanıyor, hiçbir şey tune etmiyor,
   kütüphane varsayılanlarıyla eğitiyor.

   66-68. Sayısal input parse deliği — üç ayrı node ailesi, aynı kök neden

   Repo'da core/utils/numberInput.ts::parseIntSafe tam bu tehlike için yazılmış ve 12 panelde kullanılıyor, ama tutarsız uygulanmış — aynı
   dosya içinde bile:
   ┌────┬──────────────────────────────┬──────────────────────────────────────┬───────────────────────────────────────────────────┐
   │    │ Node                         │ Dosya                                │ Sonuç                                             │
   ├────┼──────────────────────────────┼──────────────────────────────────────┼───────────────────────────────────────────────────┤
   │ 66 │ Outlier (IQR multiplier,     │ OutlierControls.tsx:38,55,74,84      │ çıplak parseFloat → NaN → null; core              │
   │    │ z-score threshold, winsorize │                                      │ config.get("multiplier",1.5) anahtar mevcut       │
   │    │ percentilleri,               │                                      │ olduğu için varsayılanı uygulamıyor → TypeError:  │
   │    │ contamination)               │                                      │ unsupported operand type(s) for *: 'NoneType' and │
   │    │                              │                                      │ 'float'                                           │
   ├────┼──────────────────────────────┼──────────────────────────────────────┼───────────────────────────────────────────────────┤
   │ 67 │ TrainTestSplitter test_size  │ TrainTestSplitNode.tsx:41,240        │ validator if (test_size <= 0 || test_size >= 1) — │
   │    │                              │                                      │ NaN için ikisi de false → isValid:true. Aynı      │
   │    │                              │                                      │ dosyada validation_size aynı parse hatasına sahip │
   │    │                              │                                      │ ama `                                             │
   ├────┼──────────────────────────────┼──────────────────────────────────────┼───────────────────────────────────────────────────┤
   │ 68 │ Ensemble + training          │ EnsembleFormSections.tsx:88,169,226, │ Number('') NaN değil 0 → wire'a geçerli JSON ama  │
   │    │ panelleri (cv, n_jobs,       │ CrossValidationSection.tsx:70,149,   │ parametre için illegal değer: cv=0 → sklearn      │
   │    │ calibration_cv, cv_folds,    │ AdvancedTuningOptions.tsx:66,80      │ "k-fold cannot be less than 2", n_jobs=0 → joblib │
   │    │ cv_random_state, n_trials,   │                                      │ "n_jobs == 0 has no meaning", n_trials=0 → sıfır  │
   │    │ random_state)                │                                      │ denemeli tuning                                   │
   └────┴──────────────────────────────┴──────────────────────────────────────┴───────────────────────────────────────────────────┘
   Hepsinde: alan temizleniyor → node hata göstermiyor → graf validasyonu geçiyor → Run enabled → iş sunucuda kullanıcının tanımadığı bir
   Python hatasıyla ölüyor. pipelineConversion/'ın tek test dosyası var (split.test.ts, 2 test).

   ---

   YENİ YÜKSEK
   ┌────┬─────────────────────────┬────────────────────────────────────────┬─────────────────────────────────────────────────────────────┐
   │    │ Bulgu                   │ Dosya                                  │ Sonuç                                                       │
   ├────┼────────────────────────┼────────────────────────────────────────┼───────────────────────────────────────────────────────────────┤
   │ 69 │ _best_score_headline   │ _execution/summary.py:585-593,         │ Tüm trial'ları başarısız olmuş bir tuning run 'mse inf · 3    │
   │    │ iki kardeşinin sahip   │ :573-582                               │ trials' diye canvas kartına ve JobsDrawer'a yazılıyor —       │
   │    │ olduğu isfinite        │                                        │ olabilecek en kötü sonuç sınırsız iyi gibi.                   │
   │    │ filtresine sahip       │                                        │ grid_random.py:201 -inf ile başlatıyor, OC-200'de nan         │
   │    │ değil; neg_ scorer'da  │                                        │ çalıştırılmış olarak kayıtlı                                  │
   │    │ işaret çevirme -inf'i  │                                        │                                                               │
   │    │ inf olarak gösteriyor  │                                        │                                                               │
   ├────┼────────────────────────┼────────────────────────────────────────┼───────────────────────────────────────────────────────────────┤
   │ 70 │ WarningCaptureHandler  │ engine/_warning_capture.py:18-19,62-84 │ Çalıştırıldı: branch_0 4 satır aldı, branch_1 2 satır —       │
   │    │ process-global         │                                        │ bölünme schedule'a bağımlı, her satır handler'ın kendi node'u │
   │    │ logger'lara            │                                        │ ile etiketli. USE_CELERY=false'ta bir /preview ile bir /run   │
   │    │ bağlanıyor; eşzamanlı  │                                        │ aynı process'i paylaşıp kirleniyor                            │
   │    │ branch'ler birbirinin  │                                        │                                                               │
   │    │ uyarısını çalıyor ve   │                                        │                                                               │
   │    │ yanlış node'a          │                                        │                                                               │
   │    │ atfediyor              │                                        │                                                               │
   ├────┼────────────────────────┼────────────────────────────────────────┼───────────────────────────────────────────────────────────────┤
   │ 71 │ _to_py_literal regex'i │ _notebook_builders.py:126-138          │ \b sınırı string literal içinde de eşleşiyor. Çalıştırıldı:   │
   │    │ true/false/null'ı JSON │                                        │ {"aliases":{"null":"unknown"}} → {"None":"unknown"}, "value   │
   │    │ string değerlerinin    │                                        │ == null" → "value == None". Export edilen notebook temiz      │
   │    │ içinde de yeniden      │                                        │ çalışıyor ve yanlış sonuç üretiyor —                          │
   │    │ yazıyor                │                                        │ AliasReplacement/ValueReplacement/InvalidValueReplacement ile │
   │    │                        │                                        │ "null" sentinel'i kullanan CSV'lerde sıradan                  │
   ├────┼────────────────────────┼────────────────────────────────────────┼───────────────────────────────────────────────────────────────┤
   │ 72 │ tz-aware               │ monitoring/tasks.py:19,29,             │ Repo'nun kendi yardımcısı _normalize_since_for_naive_column   │
   │    │ datetime.now(UTC)      │ router.py:1208-1211                    │ (router.py:971-980) üç yerde doğru kullanılmış, bu ikisinde   │
   │    │ naive kolonlara        │                                        │ kullanılmamış. tasks.py:29'da # ty:                           │
   │    │ bağlanıyor →           │                                        │ ignore[invalid-argument-type] var — tip kontrolcüsü tam bu    │
   │    │ Postgres'te 500 +      │                                        │ satırı işaretlemiş ve uyarı düzeltilmek yerine susturulmuş.   │
   │    │ retention job kalıcı   │                                        │ Sonuç: GET /monitoring/errors/timeline her çağrıda 500;       │
   │    │ ölü                    │                                        │ günlük cleanup-error-events-daily 3 retry yakıp kalıcı ölüyor │
   │    │                        │                                        │ → error_events (satır başına 8 KB traceback) sınırsız büyüyor │
   ├────┼────────────────────────┼────────────────────────────────────────┼───────────────────────────────────────────────────────────────┤
   │ 73 │ /health/detailed event │ health/routes.py:80-91                 │ Çalıştırıldı: socket_connect_timeout=5, socket_timeout=None.  │
   │    │ loop üzerinde senkron  │                                        │ Redis TCP'yi kabul edip PING'e cevap vermezse (BGSAVE, DEBUG  │
   │    │ Redis ping'i, üstelik  │                                        │ SLEEP, yarı-açık proxy) tüm event loop süresiz donuyor —      │
   │    │ socket_timeout=None    │                                        │ auth'suz, payloadsız tetiklenebilir, load balancer'ın poll    │
   │    │                        │                                        │ ettiği endpoint. manager.py:138 zaten redis.asyncio           │
   │    │                        │                                        │ kullanıyor                                                    │
   ├────┼────────────────────────┼────────────────────────────────────────┼───────────────────────────────────────────────────────────────┤
   │ 74 │ /deployment/predict    │ deployment/api.py:83-110,              │ BE-1 ile aynı sınıf ama farklı ve daha kritik yol: production │
   │ 74 │ /deployment/predict   │ deployment/api.py:83-110,                 │ BE-1 ile aynı sınıf ama farklı ve daha kritik yol:          │
   ├────┼───────────────────────┼────────────────────────────────────────────┼────────────────────────────────────────────────────────────┤
   │ 74 │ /deployment/predict   │ deployment/api.py:83-110,                  │ BE-1 ile aynı sınıf ama farklı ve daha kritik yol:         │
   │    │ joblib.load + sklearn │ service.py:514,260-273,441-488             │ production serving hot path. S3 üzerinden s3fs ağ I/O'su   │
   │    │ transform/predict'i   │                                            │ da dahil. to_thread/run_in_executor grep'i bu dosyada 0    │
   │    │ event loop'ta         │                                            │                                                            │
   │    │ çalıştırıyor          │                                            │                                                            │
   ├────┼───────────────────────┼────────────────────────────────────────────┼────────────────────────────────────────────────────────────┤
   │ 75 │ Decomposition cache   │ eda/DecompositionTree.tsx:166,173,182      │ Çalıştırıldı: filtre değişimi sonrası çağrı sayısı 1→1,    │
   │    │ anahtarı filtreleri   │                                            │ gönderilen filtre hep EU. Reset butonu daha da             │
   │    │ içermiyor → filtre    │                                            │ kötüleştiriyor: remount modül-seviyesi treeCache'e çarpıp  │
   │    │ değişince eski        │                                            │ aynı bayat seviyeleri geri yüklüyor → ölçü kolonu/dataset  │
   │    │ değerler gösteriliyor │                                            │ değiştirmeden yenilemenin yolu yok                         │
   ├────┼───────────────────────┼────────────────────────────────────────────┼────────────────────────────────────────────────────────────┤
   │ 76 │ EDA filtreleri        │ useEdaPageController.ts:95-97 +            │ grep -rn "active_filters" frontend/src → hiçbir şey.       │
   │    │ mount'ta siliniyor ve │ useEDAStore.ts:138-147                     │ Sayfayı yenileyince filtreli hesaplanmış rapor boş filtre  │
   │    │ active_filters'dan    │                                            │ çipleriyle render ediliyor → kullanıcı profilin filtresiz  │
   │    │ hiç geri yüklenmiyor  │                                            │ olduğuna inanıyor. Decomposition sonra initialFilters=[]   │
   │    │                       │                                            │ gönderip filtresiz sorguluyor, yani sayıları diğer         │
   │    │                       │                                            │ sekmelerle çelişiyor                                       │
   ├────┼───────────────────────┼────────────────────────────────────────────┼────────────────────────────────────────────────────────────┤
   │ 77 │ VIF null (mükemmel    │ eda/OverviewCards.tsx:36,                  │ vif düz dict[str,float], FiniteFloat değil; tam            │
   │    │ collinearity) → "High │ variableRow/VariableStatistics.tsx:104-113 │ collinearity'de +inf → orjson null. .filter(v => v > 5) ve │
   │    │ VIF Features" kartı   │                                            │ null > 5 === false. Çalıştırıldı:                          │
   │    │ eksik sayıyor, en     │                                            │ {a:null,b:12.5,c:1.2,d:6.0} için kart 2 gösteriyor, gerçek │
   │    │ kötü özellik düşüyor  │                                            │ 3 — ve düşen tam olarak en kötü kolon                      │
   ├────┼───────────────────────┼────────────────────────────────────────────┼────────────────────────────────────────────────────────────┤
   │ 78 │ Correlation heatmap   │ eda/CorrelationHeatmap.tsx:12-17 vs :25    │ value > 0 ? red : blue → 0 mavi; Math.max(0.4,             │
   │    │ renk ölçeği: r=0 mavi │                                            │ Math.abs(value)) → taban. Çalıştırıldı: r=0 ile r=-0.4     │
   │    │ (negatif) render      │                                            │ aynı rgba, r=0.02 ile r=0.40 aynı rgba. Legend 0'ı gri-100 │
   │    │ ediliyor ve tüm       │                                            │ olarak kodluyor, null hücre de aynı gri. Renk bir          │
   │    │ |r|≤0.4 aynı          │                                            │ heatmap'in birincil kodlaması → zayıf korelasyon bandının  │
   │    │ opaklıkta             │                                            │ tamamı yanlış                                              │
   ├────┼───────────────────────┼────────────────────────────────────────────┼────────────────────────────────────────────────────────────┤
   │ 79 │ variable adlı bir     │ eda/CorrelationHeatmap.tsx:49-56,122       │ Çalıştırıldı: headers ["Variable","variable","value"],     │
   │    │ kolon satır etiketini │                                            │ row0 ["1","1","0.7"] (beklenen ["variable","1","0.7"]) +   │
   │    │ eziyor →              │                                            │ React duplicate-key uyarısı. Tam da truncation             │
   │    │ erişilebilirlik/CSV   │                                            │ bildiriminin kullanıcıyı yönlendirdiği tablo               │
   │    │ alternatifi etiketsiz │                                            │                                                            │
   │    │ matris oluyor         │                                            │                                                            │
   ├────┼───────────────────────┼────────────────────────────────────────────┼────────────────────────────────────────────────────────────┤
   │    │ erişilebilirlik/CSV  │                                               │ bildiriminin kullanıcıyı yönlendirdiği tablo            │
   │    │ alternatifi          │                                               │                                                         │
   │    │ etiketsiz matris     │                                               │                                                         │
   │    │ oluyor               │                                               │                                                         │
   ├────┼──────────────────────┼───────────────────────────────────────────────┼─────────────────────────────────────────────────────────┤
   │    │ etiketsiz matris    │                                                  │                                                       │
   │    │ oluyor              │                                                  │                                                       │
   ├────┼─────────────────────┼──────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
   │ 80 │ Histogram barına    │ eda/VariableRow.tsx:65-69                        │ Sadece rawBin.start kullanılıyor, rawBin.end hiç      │
   │    │ tıklamak bardaki    │                                                  │ okunmuyor; üstelik core'un sağ-kapalı (start,end]     │
   │    │ satırları değil >=  │                                                  │ konvansiyonu yüzünden bir bin kaymış. 10.00-20.00 (42 │
   │    │ tıklamak bardaki    │                                                   │ okunmuyor; üstelik core'un sağ-kapalı (start,end]    │
   │    │ satırları değil >=  │                                                   │ konvansiyonu yüzünden bir bin kaymış. 10.00-20.00    │
   │    │ start üzerindeki    │                                                   │ (42 satır) barına tıklayınca amount >= 10.00         │
   │    │ her şeyi            │                                                   │ ekleniyor → tüm ≥10 satırlar + değerin kendisi.      │
   │    │ filtreliyor         │                                                   │ Tooltip "Click to filter" diyor                      │
   ├────┼─────────────────────┼───────────────────────────────────────────────────┼──────────────────────────────────────────────────────┤
   │ 81 │ Uygulamanın kendi   │ inference/inferenceData.ts:232-254 vs :325-340    │ parseCsv koşulsuz , ve \r?\n ile bölüyor (docstring  │
   │    │ CSV export'u        │                                                   │ "no quoted-field handling" diyor), rowsToCsv ise     │
   │    │ yeniden içe         │                                                   │ RFC-uyumlu tırnaklıyor. Çalıştırıldı: 2 satırlık     │
   │    │ aktarılamıyor;      │                                                   │ export → 3 satır parse, pop kolonu "TX" string'i     │
   │    │ tırnaklı alanlar    │                                                   │ oluyor. excludedColumns strip'i header adına göre    │
   │    │ tüm sonraki         │                                                   │ eşleştiği için kayma sonrası target kolonu payload'a │
   │    │ kolonları           │                                                   │ feature olarak giriyor — ve "Loaded N rows from CSV" │
   │    │ kaydırıyor ve       │                                                   │ toast'ı başarı gösteriyor                            │
   │    │ target'ı feature    │                                                   │                                                      │
   │    │ olarak sızdırıyor   │                                                   │                                                      │
   ├────┼─────────────────────┼───────────────────────────────────────────────────┼──────────────────────────────────────────────────────┤
   │ 82 │ Feature Selection   │ featureSelection/SelectionParameters.tsx:41,60,16 │ threshold/alpha/param çıplak,                        │
   │    │ aynı dosyada        │ 0,191 vs :124,145,243,259                         │ k/percentile/max_features/step güvenli — aynı        │
   │    │ parseIntSafe ve     │                                                   │ dosyada. types.ts:20 threshold?: number | string     │
   │    │ çıplak parseFloat   │                                                   │ diyor ve "median" olabileceğini söylüyor ama widget  │
   │    │ karıştırıyor;       │                                                   │ <input type="number"> → yazınca NaN → null           │
   │    │ belgelenen "median" │                                                   │                                                      │
   │    │ sentinel'i          │                                                   │                                                      │
   │    │ girilemez           │                                                   │                                                      │
   └────┴─────────────────────┴───────────────────────────────────────────────────┴──────────────────────────────────────────────────────┘
   ---

   YENİ ORTA / DÜŞÜK
   ┌────┬───────────────────────────────────────────────────────────────────────────────────────┬────────────────────────────────────┐
   │    │ Bulgu                                                                                 │ Dosya                              │
   ├────┼───────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────┤
   │ 83 │ Notebook export: sanitize edilmemiş node_id yorum satırına interpolate ediliyor; \n   │ _notebook_builders.py:385-397      │
   │    │ derlenen Python enjekte ediyor (compile() başarılı)                                   │                                    │
   ├────┼───────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────┤
   │ 84 │ LocalArtifactStore.save atomik değil; SIGKILL/OOM/disk-dolu kesik .joblib bırakıyor,  │ artifacts/local.py:40-48           │
   │    │ deployment tam o anahtardan yüklüyor → kalıcı 400                                     │                                    │
   ├────┼───────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────┤
   │ 85 │ S3 read-cache dosya adı /→_ ile injective değil: data/train.csv ile data_train.csv    │ data/catalog.py:346-353            │
   │    │ aynı cache dosyası → sessiz yanlış dataset okuma                                      │                                    │
   ├────┼───────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────┤
   │ 86 │ FileSystemCatalog.save/S3Catalog.save pandas-only writer çağırıyor; SKYULF_ENGINE     │ data/catalog.py:210-218,493-498    │
   │    │ varsayılanı polars ve aynı sınıfın load() polars döndürüyor → AttributeError. Aynı    │                                    │
   │    │ sınıfta _write_to_cache doğru branch'liyor                                            │                                        │
   ├────┼───────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────┤
   │ 87 │ DataSourceRead config'i redakte ediyor ama source_metadata'yi olduğu gibi döndürüyor; │ schemas/ingestion.py:86 +              │
   │    │ döndürüyor; _handle_ingestion_failure oraya ham driver metni yazıyor → redaksiyon    │ data_ingestion/tasks.py:150-156        │
   │    │ sınırı yanlış kolonda                                                                │                                        │
   ├────┼─────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────┤
   │ 88 │ _run_migrations her exception'ı yutuyor ve yine de ✅ Database tables               │ database/engine.py:244-258,206-207     │
   │    │ created/updated basıyor; iki giriş artık var olmayan                                │                                        │
   │    │ basic_training_jobs/advanced_tuning_jobs tablolarını ALTER ediyor → her açılışta    │                                        │
   │    │ başarısız, operatörü gerçek hataya köreltiyor                                       │                                        │
   ├────┼─────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────┤
   │ 89 │ get_db_type_from_url tanımadığı her URL için SQLite'a düşüyor:                      │ database/adapter.py:54-71              │
   │    │ postgresql+psycopg2:// → sqlite. Ve bu tam olarak record_pipeline_error'ın ürettiği │                                        │
   │    │ yazım                                                                               │                                        │
   ├────┼─────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────┤
   │ 90 │ realtime/manager.broadcast sıralı ve per-client timeout'suz await send_text →       │ realtime/manager.py:68-79              │
   │    │ okumayı bırakan tek client tüm job-event dağıtımını ve Redis subscriber loop'unu    │                                        │
   │    │ donduruyor; /ws/jobs auth'suz ve _clients sınırsız                                  │                                        │
   ├────┼─────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────┤
   │ 91 │ _run_transformer hiç yazılmayan bir artifact_key kaydediyor; legacy inference       │ _node_runners.py:1137 vs :1147,        │
   │    │ inference bundle FileNotFoundError'ı warning'e yutup transformers: [] ile   │ _feature_eng.py:250-271                          │
   │    │ devam ediyor → preprocessing'siz deploy                                     │                                                  │
   ├────┼─────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────┤
   │ 92 │ ThresholdTuningService.save source anahtarını düşürüyor; save tüm dict'i    │ threshold_tuning_service.py:243-249 vs :155,166  │
   │    │ değiştirdiği için ilk manuel kayıtta "source":"training" provenance'ı        │                                                  │
   │    │ siliniyor                                                                    │                                                  │
   ├────┼──────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────┤
   │ 93 │ Drift CSV export ekrandan farklı: kategoriklerde PSI boş,                    │ drift/_utils/csvExport.ts:21,23,32,45            │
   │    │ feature_importances'ta olmayan kolona uydurma Risk yazıyor, " escape         │                                                  │
   │    │ edilmiyor                                                                    │                                                  │
   │    │ feature_importances'ta olmayan kolona uydurma Risk yazıyor, " escape    │                                                       │
   │    │ edilmiyor                                                               │                                                       │
   ├────┼─────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
   │ 94 │ Drift threshold min/max dekoratif; PSI=5 yazmak her kolonu Stable       │ ThresholdsPanel.tsx:31-41 + router.py:309-323         │
   │    │ yapıyor (typo ile temiz sağlık raporu), -1 her şeyi critical yapıyor.   │                                                       │
   │    │ Backend de Query(ge=,le=) olmadan alıyor                                │                                                       │
   ├─────┼────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
   │ 95  │ Threshold alanını temizlemek varsayılana değil önceki run'ın           │ ThresholdsPanel.tsx:35 + useDriftReport.ts:33-34      │
   │     │ verdict'lerine düşüyor; hangi eşiğin yürürlükte olduğunu gösteren      │                                                       │
   │     │ hiçbir gösterge yok                                                    │                                                       │
   ├─────┼────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
   │ 96  │ JSON editörünün satır/kolon tespiti güncel V8'de ölü kod: sadece       │ inference/inferenceData.ts:133-155                    │
   │     │ /position\s+(\d+)/ (2023 öncesi) ve Firefox biçimi eşleşiyor.          │                                                       │
   │     │ Çalıştırıldı (Node v24): null. Test legacy biçimi assert ettiği için   │                                                       │
   │     │ geçiyor                                                                │                                                       │
   ├─────┼────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
   │ 97  │ Relative-error histogramı ±2 dışındaki örnekleri clamp etmek yerine    │ ExperimentsPage/utils/regressionCharts.ts:42-51       │
   │     │ düşürüyor; tooltip "Capped at ±200%" diyor. Barlar split boyutuna      │                                                       │
   │     │ toplanmıyor, tam da ağır kuyruklu splitlerde                           │                                                       │
   ├─────┼────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
   │ 98  │ Sparkline'lar zaman noktası düşürüyor ve kendini normalize ediyor: 3   │ useDriftHistory.ts:31-37 + drift/Sparkline.tsx:13-25  │
   │     │ kontrol için 2 nokta, ve şekil ile renk çelişiyor (0.9→0.25 düşen      │                                                       │
   │     │ rampa kırmızı nokta, 0.19→0.21 çıkan rampa aynı kırmızı nokta)         │                                                       │
   ├─────┼────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
   │ 99  │ threshold = 0 tüm tahminleri tek sınıfa çökertiyor (value/0 =          │ ExperimentsPage/utils/classificationCharts.ts:150-163 │
   │     │ Infinity, strict > ile hep kazanır); hepsi 0 ise NaN → bestIdx=0 → tüm │                                                       │
   │     │ örnekler classes[0]                                                    │                                                       │
   ├─────┼────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
   │ 100 │ publish_job_event her event'te yeni Redis client + pool kurup atıyor;  │ realtime/events.py:52-56,86-89                        │
   │     │ 2000-trial'lık tuning ~2000 broker bağlantısı açıyor                   │                                                       │
   ├─────┼────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
   │ 101 │ Clustering sekmesi var olmayan bir plot'u anlatıyor: points hiç render │ tabs/ClusteringTab.tsx:18,40,49,98,103,119-125        │
   │     │ edilmiyor, #clustering-chart boş, ama caption "The plot shows a 2D     │                                                       │
   │     │ projection (PCA)" ve "Download Chart" butonu metin kartlarını PNG      │                                                       │
   │     │ yapıyor. Null centroid tüm PCA sekmesini çökertiyor                    │                                                       │
   ├─────┼────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
   │ 102 │ Altı sekme !!profile.X && <Tab/> döndürüyor → bölüm yoksa tamamen boş  │ edaPage/EdaProfileContent.tsx:109-217                 │
   │     │ içerik paneli; sidebar tıklanabilir kalıyor, "uygulanamaz" ile         │                                                       │
   │     │ "yükleniyor" ile "çöktü" ayırt edilemez                                │                                                       │
   ├─────┼────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
   │ 103 │ Decomposition hata yakalamaları sadece console.error: kök yükleme      │ eda/DecompositionTree.tsx:210-213,322-327,386-389     │
   │     │ başarısızsa 600px tamamen boş panel, split başarısızsa menü kapanıp    │                                                       │
   │     │ hiçbir şey olmamış gibi görünüyor → kullanıcı sonsuz tekrar deniyor    │                                                       │
   ├─────┼────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
   │ 104 │ Correlation PNG export'u ekrandakiyle uyuşmuyor: ikinci, sapmış renk   │ tabs/CorrelationsTab.tsx:31-33,66-74                  │
   │     │ ölçeği kopyası (0.2 vs 0.4 taban) + truncation bildirimi yok           │                                                       │
   ├─────┼────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
   │ 105 │ Number(bin.start).toFixed(2) — Number(null)===0 değil NaN, yani +inf   │ eda/DistributionChart.tsx:36-37                       │
   │     │ içeren kolonun son barı tooltip ve CSV'de 10.00 - 0.00 (sıfırdan       │                                                       │
   │     │ geriye giden aralık) diye etiketleniyor                                │                                                       │
   ├─────┼────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
   │ 106 │ TimeSeries seasonality bar'ı dataKey="count" ama core oraya bir mean   │ tabs/TimeSeriesTab.tsx:140,175,220,91                 │
   │ 106 │ TimeSeries seasonality bar'ı dataKey="count" ama core oraya bir mean  │ tabs/TimeSeriesTab.tsx:140,175,220,91                 │
   │     │ +inf içeren kolonun son barı tooltip ve CSV'de 10.00 - 0.00        │                                                           │
   │     │ (sıfırdan geriye giden aralık) diye etiketleniyor                  │                                                           │
   ├─────┼────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────┤
   │ 106 │ TimeSeries seasonality bar'ı dataKey="count" ama core oraya bir    │ tabs/TimeSeriesTab.tsx:140,175,220,91                     │
   │     │ mean alias'lıyor → tooltip ortalama değerde count: 3.42            │                                                           │
   │     │ gösteriyor, başlık "Average values" diyor. ACF formatter null'da   │                                                           │
   │     │ çöküyor; trend serisi listesi sadece trend[0]'ın anahtarlarından   │                                                           │
   │     │ türetiliyor                                                        │                                                           │
   ├─────┼────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────┤
   │ 107 │ CausalTab.tsx:15 "highest-variance" iddiası, core'daki NaN-sort    │ tabs/CausalTab.tsx:14,15,21                               │
   │     │ hatası yüzünden etkilenen raporlarda gerçeğin tam tersi; graf      │                                                           │
   │     │ içinde kullanıcının fark etmesini sağlayacak hiçbir gösterge yok   │                                                           │
   ├─────┼────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────┤
   │ 108 │ SampleData ızgarasının şeması tek satırdan türetiliyor: row 0'da   │ tabs/SampleDataTab.tsx:23,57                              │
   │     │ olup sonrakinde olmayan anahtar "undefined" yazıyor, row 0'da      │                                                           │
   │     │ olmayıp sonrakinde olan kolon ızgaradan tamamen kayboluyor         │                                                           │
   ├─────┼────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────┤
   │ 109 │ Outliers sekmesi "sıfır anomali bulundu" ile "analiz başarısız"ı   │ tabs/OutliersTab.tsx:12-14,72                             │
   │     │ aynı metinde birleştiriyor: temiz dataset "Outlier analysis did    │                                                           │
   │     │ not return any results" diyor. Null hücre değerleri literal null   │                                                           │
   │     │ yazıyor                                                            │                                                           │
   ├─────┼────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────┤
   │ 110 │ SearchSpaceInput sadece Number.isNaN ile doğruluyor →              │ trainingSettings/../SearchSpaceInput.tsx:53-58            │
   │     │ 1e400/Infinity yeşil geçiyor, JSON.stringify onu null yapıyor; hex │                                                           │
   │     │ (0x10) sessizce ondalık kabul ediliyor                             │                                                           │
   ├─────┼────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────┤
   │ 111 │ FE-1'in kök nedeni üç değil dört elle tutulan liste ve ikisi       │ connectionValidation.ts:29, connectionPolicy.ts:41,       │
   │     │ birbirine ters: training iki conversion listesinde var, iki        │ ensemble.ts:11, connectedModels.ts:18                     │
   │     │ connection listesinde yok                                          │                                                           │
   ├─────┼────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────┤
   │ 112 │ Ensemble "Base Model Params" paneli, bağlı kaynak modeli           │ ensembleSettings/connectedModels.ts:86-89,105-107         │
   │     │ düzenledikten sonra bayat hiperparametre gösteriyor (changedFields │                                                           │
   │     │ boşsa base_estimator_params hiç yazılmıyor). Payload doğru — ama   │                                                           │
   │     │ kullanıcı 3'ü geri yazıp yine 10 çalıştığını görünce hiçbir        │                                                           │
   │     │ açıklama bulamıyor                                                 │                                                           │
   ├─────┼────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────┤
   │ 113 │ UI-only isExpanded (akordeon durumu) sunucuya gönderilip           │ featureGeneration/types.ts:10 +                           │
   │     │ kaydedilen pipeline artefaktına yazılıyor; aynı toptan-node.data   │ pipelineConversion/preprocessing.ts:245-252               │
   │     │ iletimi definitionType'ı da feature_selection/outlier              │                                                           │
   │     │ parametrelerine sızdırıyor                                         │                                                           │
   ├─────┼────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────┤
   │ 114 │ _find_deployment_context scalar_one_or_none() kullanıyor; iki      │ monitoring/router.py:412-424                              │
   │     │ aktif deployment aynı job'a bakarsa MultipleResultsFound blanket   │                                                           │
   │     │ except'e düşüp (None,None) döndürüyor → drift alert deployment     │                                                           │
   │     │ bağını sessizce kaybediyor                                         │                                                           │
   ├─────┼────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────┤
   │ 115 │ created_at=ref.created_at or "Unknown" sonra sort(key=... or "") — │ monitoring/router.py:150,168                              │
   │     │ "Unknown" her ISO string'ten büyük karşılaştırılıyor, yani zaman   │                                                           │
   │     │ damgası çözülemeyen artefaktlar drift job seçicide en üste çıkıyor │                                                           │
   └─────┴────────────────────────────────────────────────────────────────────┴───────────────────────────────────────────────────────────┘
   ---

   Desen: ikinci turda da aynı iki kök neden

   A. "Sunucunun gerçekte gönderdiği ile renderer'ın gönderildiğini varsaydığı şey arasındaki sınır." psi_categorical'i DriftTable ve sortRows
    anlıyor ama useDriftHistory ve csvExport anlamıyor (#63, #93); null'ı asProbabilityMap anlıyor ama toNumericArray anlamıyor (#64);
   RFC-tırnaklı CSV'yi rowsToCsv yazıyor ama parseCsv okuyamıyor (#81); modern V8 hata mesajını Firefox dalı karşılıyor, Chrome dalı
   karşılamıyor (#96). 1129 geçen test bunları yakalamıyor, çünkü her test sınırın bir yarısını o yarıya benzeyen bir fixture ile çalıştırıyor
    — hiç karşı tarafın gerçek çıktısıyla değil.

   B. "Doğru yardımcı yazılmış ama tutarsız uygulanmış." parseIntSafe var, 12 panelde kullanılıyor, aynı dosyada bile atlanıyor (#66-68, #82,
   #110). _normalize_since_for_naive_column var, üç yerde doğru, iki yerde atlanmış ve birinde # ty: ignore ile susturulmuş (#72). FiniteFloat
    var, NumericStats'a uygulanmış, BoxPlotStats/GeospatialStats/vif'e uygulanmamış (#62, #77). redact_credentials var, middleware yolunda
   çağrılıyor, exception-handler yolunda çağrılmıyor (#60). update_drift_alert_disposition doğru JSON copy desenini gösteriyor, PATCH
   description göstermiyor (#59).

   Bu turun en yüksek sinyalli tek artefaktı #72'deki # ty: ignore[invalid-argument-type]: tip kontrolcüsü hatayı bulmuş, ve o satır
   error_events'ı sınırlı tutan retention job'ı — Postgres'te kalıcı olarak ölü.

   ---

   qwen polars/pandas test:

    ▪ İnceleme tamam. Scratch probe'ları tmp_polars_e2e/ altında bıraktım (untracked).

   Kısa cevap

   Polars ile başlamak structurally sağlam — 55 adımlık zincir boyunca engine hiç pandas'a düşmüyor, satır/kolon/NA sayıları pandas ile
   birebir aynı, numpy'a geçişte matrisler element bazında aynı. Ama tek bir yerde gerçek bir ayrışma var: temporal (Datetime/Date)
   kolonlar. Orada pandas yüksek sesle patlıyor, polars sessizce bozuk model eğitiyor.

   Çalıştırdıklarım
   ┌───────────────────────────┬────────────────────────────────┬────────────────────────────────────────────────────────────────────┐
   │ Probe                     │ Ne yaptı                       │ Sonuç                                                              │
   ├───────────────────────────┼────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
   │ Mevcut                    │ Her node için pandas↔polars    │ 229 passed                                                         │
   │ test_registry_contract.py │ parite                         │                                                                    │
   ├───────────────────────────┼────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
   │ chain_probe.py            │ 55 adımlık zincir, tüm         │ 52 adım iki engine'de de başarılı, 3 adım iki engine'de de aynı    │
   │                           │ preprocessing aileleri, polars │ hatayla düştü (poly NaN, over/undersample non-numeric). Sıfır      │
   │                           │ vs pandas                      │ polars-only hata, sıfır engine flip, sıfır row/col/NA farkı        │
   ├───────────────────────────┼────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
   │ bridge_probe.py A         │ Sayısal zincir → SklearnBridge │ X_train max sapma 3.55e-15, kolonlar aynı, lr_auc/rf_acc farkı 0.0 │
   │                           │ → numpy → LR/RF                │                                                                    │
   ├───────────────────────────┼────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
   │ bridge_probe.py B         │ feature_target_split ile y'yi  │ İki engine'de de aynı satırlar (181→180→137→137→112→112→89), y her │
   │                           │ ayırıp 7 satır-düşüren node    │ adımda X ile hizalı                                                │
   ├───────────────────────────┼────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
   │ bridge_probe.py C/D       │ Bridge dtype kenar durumları   │ Aşağıda                                                            │
   └───────────────────────────┴────────────────────────────────┴────────────────────────────────────────────────────────────────────┘
   Bulgu: SklearnBridge karışık-dtype frame'de engine'e göre farklı numpy üretiyor

   Mekanizma (skyulf-core/skyulf/engines/sklearn_bridge.py:74-76 → PolarsEngine.to_numpy): pandas.DataFrame.to_numpy() karışık dtype'lı
   frame'de object döner, pl.DataFrame.to_numpy() ise ortak supertype'a cast eder.

   Bunun çoğu zararsız — bool+float ve nullable Int64+float için dtype farklı ama değerler aynı ve sklearn iki tarafta da aynı davranıyor
   (ikisi de RF=fits, ikisi de LR=ValueError).

   Zararsız olmayan tek durum: Datetime. pipeline_repro.py ile SkyulfPipeline.fit() üzerinden (production entry point) doğruladım:
    Config: DateFeatures(drop_original=False)  ← node'un KENDİ default'u
      pandas → TypeError: float() argument must be a string or a real number, not 'Timestamp'
      polars → fitted
               test: balanced_accuracy=0.5  matthews_corrcoef=0.0  roc_auc=0.4475
               coef_: [1.8e-36, 3.6e-37, 9.1e-19, 5.4e-37, 2.9e-36]
   Aynı data, date kolonu düşürülünce (drop_original=True): iki engine de fit oluyor, worst metric diff 1.11e-16, worst coef diff 1.30e-15 —
    tam parite. Kontrol config'i (date hiç yok) de tam parite.

   Yani polars tarafında Datetime kolon köprüden epoch-nanosecond float64 olarak geçiyor (~1.67e18, diğer feature'larla oranı ~8.7e17).
   L2-regularized logistic regression bu ölçekte katsayıları 1e-36'ya çökertiyor ve model sabit sınıf tahmincisi haline geliyor: MCC=0,
   balanced_accuracy=0.5, recall=1.0, AUC 0.5'in altında. Hiç hata yok, hiç warning yok.

   İki ayrı kusur var burada: (1) engine paritesi kırılıyor — aynı config bir engine'de çalışıp diğerinde patlıyor; (2) polars tarafı
   temporal feature'ı sayısal modele sessizce sokuyor, pandas'ın TypeError'ı kaza eseri koruyucu davranıyor.

   Kapsam

   SkyulfPipeline.fit(pl.DataFrame) kodda açıkça desteklenen bir yol (pipeline.py:277 civarı yorum: "the advertised polars-native usage").
   Backend tarafını tam izlemedim ama artifact'lar joblib round-trip (artifacts/local.py:60) ve ingestion data_service._should_use_polars
   ile default polars; pandas'a dönüş sadece preview/SHAP/merge gibi dar noktalarda var. Yani product yolunda da polars frame'in core'a
   ulaşması olası — bunu kesinleştirmek için canlı bir job'da _get_input çıktısının tipine bakmak gerek.

   Bu davranışı sabitleyen hiçbir test yok (test_engines_sklearn_bridge.py içinde datetime/Timestamp geçmiyor), yani düzeltme mevcut bir
   kontratı kırmaz.

   qwen polars/pandas test 2. deneme:

   Tekrar test ettim ve kendi probe'umda bir hata buldum — düzeltince sonuç çok daha güçlü çıktı, ayrıca ikinci bir bulgu
   ortaya çıktı.

   1. Önceki sonuçlar birebir reproduce oldu

   chain_probe (55 adım), bridge_probe A/B/C/D, pipeline_repro — hepsi aynı sayılarla, hiç flaky değil. A: max sapma
   3.553e-15. B: 181→180→137→137→112→112→89, iki engine'de de hizalı. C/D: aynı 4 DIFF.

   2. Kendi hatam: ilk sweep'te flag tersti

   build_config(model, keep_date=True) içinde "drop_original": keep_date yazmışım. Yani "polars" diye etiketlenen sütun
   aslında date-dropped, "clean" diye etiketlenen date-kept idi — ve pandas'ın date-kept hücresi hiç çalışmamıştı. O sweep
   "0 divergence" raporladı, bu yanlıştı.

   Düzeltilmiş 2×2 sweep (8 model × 4 seed, dört hücre de koşuldu):
    date-KEPT: pandas raised  : 32/32
    date-KEPT: polars raised  :  0/32
    date-KEPT: divergent      : 32/32
   Model ailesine göre polars'ın "başarısı":
   ┌──────────┬─────────────────────┬───────────────────────────────────────────────────────────────────────────────┐
   │ Aile     │ Modeller            │ date-KEPT polars sonucu                                                       │
   ├──────────┼─────────────────────┼───────────────────────────────────────────────────────────────────────────────┤
   │ linear   │ logistic, svc, sgd, │ 16/16 dejenere — mcc tam +0.000, auc ≈0.5 (logistic: 0.457/0.526/0.513/0.486) │
   │          │ gaussian_nb         │                                                                               │
   ├──────────┼─────────────────────┼───────────────────────────────────────────────────────────────────────────────┤
   │ distance │ k_neighbors         │ 4/4 bozuldu — auc 0.376/0.372/0.408/0.596 vs clean 0.619/0.622/0.474/0.618    │
   ├──────────┼─────────────────────┼───────────────────────────────────────────────────────────────────────────────┤
   │ tree     │ RF, GB, DT          │ dejenere değil; epoch kolonunu normal monotonik feature olarak kullanıyor,    │
   │          │                     │ clean baseline ile karşılaştırılabilir                                        │
   └──────────┴─────────────────────┴───────────────────────────────────────────────────────────────────────────────┘
   Yani linear/distance'ta 20/20 hücre hasarlı, tree'lerde değil (ağaçlar eşikle böldüğü için 1.67e18 ölçeğinden
   etkilenmiyor). pandas ise 32/32 TypeError veriyor.

   3. Yeni bulgu: sgd_classifier — datetime ile ilgisi yok

   Düzeltilmiş sweep, date düşürülmüş config'de bile sgd'nin engine'ler arası farklı olduğunu gösterdi (seed 3: pandas mcc
   +0.184, polars −0.177; 4 seed'de de). Bunu 4 probe ile kök nedene kadar izledim ve ilk iki hipotezim yanlış çıktı:

   - ❌ dtype (object vs float64): izole sklearn testinde object ile float64 bit-bit aynı coef verdi.
   - ❌ X/y hizalama: SGDClassifier.fit'i enstrümante ettim — y tam eşit, params tam eşit, tek fit çağrısı.

   Gerçek neden:
    fit#0: X bitwise-equal=False  max|d|=4.441e-16  y equal=True  params equal=True
           132/600 hücre farklı; kolon başına: [100, 32, 0, 0]
           n_iter_: pandas 15  vs  polars 49     (offline refit ikisinde de 50)
   Fark sadece scale edilmiş kolonlarda (month/dayofweek bit-bit aynı). İzole ettim:
   ┌────────────────────────────────┬──────────────────────────────────┐
   │ Adım                           │ pandas vs polars                 │
   ├────────────────────────────────┼──────────────────────────────────┤
   │ SimpleImputer (mean)           │ bitwise eşit değil, max 1.39e-17 │
   ├────────────────────────────────┼──────────────────────────────────┤
   │ SimpleImputer + StandardScaler │ max 2.22e-16                     │
   └────────────────────────────────┴──────────────────────────────────┘
   Yani kök neden float non-associativity: pandas ve polars kolon ortalamasını farklı toplama sırasıyla hesaplıyor. Bu
   1e-17 gürültü SGD'nin tol-bazlı durma kriterini diskret olarak kaydırıyor (epoch 15 vs 49) ve model tamamen
   farklılaşıyor. 1e-16 pertürbasyon testi tek başına bunu göstermedi çünkü orada n_iter aynı kalmıştı — asıl mesele durma
   epoch'unun atlaması.

   Stabil learner'larda etkisi yok: logistic coef farkı 1.3e-15, metric'ler identical; tree'ler identical. Yani bu P1
   değil, ama "aynı canvas + aynı data + aynı seed → engine'e göre farklı model" demek.

   Özet

   - Sağlam: 55 adımlık polars zinciri engine flip yapmıyor, row/col/NA paritesi tam, X/y hizalama tüm satır-düşüren
   node'larda doğru, temiz sayısal veride numpy matrisi element bazında aynı.
   - P1 — datetime köprüsü: 32/32 divergence, linear/distance'ta 20/20 dejenere model, tamamen sessiz.
   - P2 — son-bit float gürültüsü: SimpleImputer mean → StandardScaler, tol-bazlı iterative learner'larda farklı model.

   Bir uyarı

   Bu oturum sırasında working tree benim dışımda değişiyor: 29 modified dosya, mtime 15:33–15:37 (benim probe'larım
   15:18+), hepsi benim konumla ilgisiz (data_ingestion, monitoring, EDA frontend). Test ettiğim yollar altında iki dosya
   var — skyulf/pipeline/seal.py ve preprocessing/vectorization/sentence_embedder.py — ikisi de bu bulguları etkilemiyor.
   Ama commit planlıyorsanız paralel bir oturum açık olabilir.

   Probe'lar tmp_polars_e2e/ altında (9 dosya, untracked). Bir sonraki adım için önceki üç seçenek duruyor; ek olarak P2
   için dördüncü bir seçenek var: SimpleImputer/StandardScaler'da polars yolunu pandas ile bit-bit aynı toplamaya zorlamak
   — ama bu performans maliyeti getirir ve float non-associativity tamamen çözülebilir bir şey değil, o yüzden orada
   "düzelt" yerine "belgele + tol-bazlı learner'lar için uyar" daha gerçekçi olabilir.


qwen ucuncu test turu:

Üçüncü tur tamam. En önemli şey: kod tabanı altımda değişmiş, o yüzden her şeyi yeni duruma karşı baştan koştum.

   HEAD değişmiş — önceki sonuçlar farklı bir kod durumuna aitti
    oturum başı : branch 0822, HEAD 7966d0e9
    şimdi       : branch 0823, HEAD d2959e0f, dirty=0
   Test ettiğim yollarda değişenler: engines/sklearn_bridge.py (+22), preprocessing/scaling/_common.py (+26), scaling/minmax.py, scaling/robust.py,
   pipeline/seal.py (+48), ayrıca tuning/optuna/pruning. Scaling değişiklikleri sadece validasyon (validate_scaling_range), aritmetik değil;
   imputation/simple.py hiç değişmemiş → P2'nin kök nedeni etkilenmemiş.

   Yeni HEAD'de hepsi birebir reproduce oldu
   ┌──────────────────────┬─────────────────────────────────────────────────────────────────────────────────────────────────┬───────────┐
   │ Probe                │ Sonuç                                                                                           │ Öncekiyle │
   ├──────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────┼───────────┤
   │ pipeline_repro       │ pandas TypeError / polars fitted; config 2-3 parite 1.110e-16 / 1.303e-15                       │ aynı      │
   ├──────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────┼───────────┤
   │ robustness_probe 2×2 │ 32/32 divergent, pandas raised 32/32, polars fitted 32/32, dejenere 23/32                       │ aynı      │
   ├──────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────┼───────────┤
   │ sgd_fit_probe        │ 132 hücre, kolon başına [100, 32, 0, 0], max|d| 4.441e-16, y equal, params equal                │ aynı      │
   ├──────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────┼───────────┤
   │ bridge_probe A       │ max_abs_dev 3.553e-15, lr/rf diff 0.0                                                           │ aynı      │
   ├──────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────┼───────────┤
   │ bridge_probe B       │ 181→180→137→137→112→112→89, iki engine'de de hizalı                                             │ aynı      │
   ├──────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────┼───────────┤
   │ bridge_probe C/D     │ aynı 4 DIFF, aynı D satırları                                                                   │ aynı      │
   ├──────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────┼───────────┤
   │ Core test suite      │ 395 passed (registry contract, sklearn bridge, polars engine, wrapped polars, xy row alignment) │ yeni      │
   └──────────────────────┴─────────────────────────────────────────────────────────────────────────────────────────────────┴───────────┘
   395 passed önemli: mevcut güvenlik ağı iki bulguyu da yakalamıyor.

   Yeni açı: pl.from_pandas artifact'i değil

   Native kurulmuş polars frame'lerle test ettim (dönüşümden geçmemiş):
    input                       bridge on date_col      pandas fit   this-engine fit
    pandas datetime64[ns]       datetime64[ns]          raised       RAISED TypeError
    polars Datetime(ns) native  datetime64[ns]          raised       fit auc=0.457 mcc=+0.000
    polars Datetime(us) native  datetime64[us]          raised       fit auc=0.457 mcc=+0.000
    polars Datetime(ms) native  datetime64[ms]          raised       fit auc=0.457 mcc=+0.000
    polars Date native          datetime64[D]           raised       fit auc=0.653 mcc=+0.101
    polars via from_pandas      datetime64[ns]          raised       fit auc=0.457 mcc=+0.000
   Üç şey netleşti:

   1. Native frame'ler from_pandas ile aynı davranıyor → bulgu dönüşüm artifact'i değil, bridge'in kendisinde.
   2. Birim fark etmiyor: ns (1.67e18), us (1.67e15), ms (1.67e12) → üçü de aynı dejenere sonuç (auc 0.457, mcc +0.000). Yani magnitude değil, yüksek
   kardinaliteli temporal kolonun kendisi.
   3. pl.Date dejenere olmuyor: auc 0.653 / mcc +0.101 — date-dropped clean baseline ile birebir aynı. Nedenini izole edemedim: kardinalite/magnitude
   ayrımı için yazdığım script hatalı bir pl.Series kurulumunda polars ComputeError verdi, o yüzden bu doğrulanmamış bir gözlem olarak kalsın.

   Ayrıca trigger'ı daraltan bir detay: bridge sadece temporal bir frame'de iki engine'de de datetime64 koruyor (float'a çevirmiyor). Supertype cast
   yalnızca temporal sayısal kolonlarla karışık olduğunda oluyor. Yani hata "datetime varsa" değil, "datetime + sayısal karışık frame modele gidiyorsa".

   Değişmeyenler

   - P1 — temporal karışık frame: 32/32 divergence, linear/distance'ta 20/20 dejenere model, tamamen sessiz; pandas TypeError veriyor.
   - P2 — son-bit float gürültüsü: SimpleImputer mean 1.39e-17 → StandardScaler 2.22e-16 → SGD'nin tol-bazlı durması epoch 15 vs 49'a kayıyor → farklı
   model. Stabil learner'larda görünmez (LR 1.3e-15).
   - Sağlam: 55 adımlık zincir, engine korunumu, X/y hizalama, temiz sayısal numpy paritesi.

   Probe'lar tmp_polars_e2e/ altında 10 dosya (untracked). İsterseniz sonuçları bir commit'e pinleyip bulguları
   initiatives/analysis/opus_core_analysis-open_queue.md'ye OC kaydı olarak yazabilirim — ama ağaç paralel bir oturum tarafından aktif değiştirildiği
   için, düzeltmeye başlamadan önce o oturumun bitmesini beklemek daha güvenli olur.


qwen dorduncu test turu:

Çelişki tamamen çözüldü ve P2'nin kök nedeni artık nedensel olarak kanıtlı. Bu arada kendi önceki iki çıkarımımın yanlış olduğunu da gördüm.

   1. Capture sadıkmış — probe 8'in hatası bende
    skyulf fitted [pandas]   coef=[17.47038, 3.242125, -4.348287, 11.216]  n_iter_=15
    offline refit [pandas]   coef=[17.47038, 3.242125, -4.348287, 11.216]  n_iter_=15   bit-identical: True
    skyulf fitted [polars]   coef=[5.926827, -0.031015, -2.412544, 5.174093] n_iter_=49
    offline refit [polars]   coef=[5.926827, -0.031015, -2.412544, 5.174093] n_iter_=49   bit-identical: True
   Probe 8'de SGDClassifier(random_state=42, shuffle=True) ile refit etmiştim — yani default parametrelerle, oysa skyulf loss='log_loss' kullanıyor (probe
    8'in kendi param dökümünde görünüyordu, ben kullanmadım). O yüzden 9.11/n_iter 50 çıkmıştı. Gerçek parametrelerle (SGDClassifier(**captured_params))
   refit, skyulf'ün modelini iki engine'de de bit-bit üretiyor. Yani "skyulf farklı bir yoldan fit ediyor" şüphesi boştu.

   2. 1e-16 gürültü tek başına modeli değiştiriyor
    X max|d| between engines : 4.441e-16
    coef max|d|              : 1.154e+01     n_iter_ 15 vs 49
   3. Nedensel kanıt: girdiyi bit-bit eşitleyince fark sıfırlanıyor
    differing cells per column BEFORE: [100, 32, 0, 0]
    differing cells per column AFTER : [0, 0, 0, 0]     (polars X'in 0-1 kolonları pandas'ınkiyle değiştirildi)

    polars params on bit-exact-pandas X : coef=[17.47038, ...] n_iter_=15
    pandas refit (reference)            : coef=[17.47038, ...] n_iter_=15   --> MATCHES: True
    polars refit (reference)            : coef=[5.926827, ...] n_iter_=49   --> matches: False
   Ve ters yön:
    pandas params on polars X (float64) : coef=[5.926827, ...] n_iter_=49  --> matches polars refit: True
   Yani sonucu belirleyen tek şey dizinin bitleri. Parametreler, dtype (object vs float64), engine kimliği — hepsi irrelevant. İki yönde de çapraz test
   bunu doğruluyor.

   Kendi önceki iki çıkarımımı düzeltiyorum

   1. "SGD 1e-16'ya duyarsız" demiştim (pertürbasyon testinde coef değişimi 5e-14 çıkmıştı). Bu veri için yanlış. Sentetik testim iyi koşullu rasgele
   veriydi ve iki taraf da aynı epoch'ta durmuştu. Gerçek veri loss='log_loss' + tol=1e-3 ile bir durma-kriteri uçurumunun kenarında: 1e-16 fark epoch'u
   15'ten 49'a atlatıyor. Yani bu genel bir amplifikasyon değil, veriye bağlı bir cliff — ama cliff'te olduğunda sonuç felaket.
   2. Probe 8'in "offline refit aynı sonucu verdi" çıkarımı geçersizdi; default parametrelerle koşmuştu.

   Konsolide nihai durum (HEAD d2959e0f, branch 0823)
   ┌─────────────────────────┬─────────────────┬───────────────────────────────────────────────────────────────────────────────────────────────────┐
   │                         │ Durum           │ Kanıt                                                                                             │
   ├─────────────────────────┼─────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────┤
   │ 55 adımlık polars       │ Sağlam          │ engine flip yok, row/col/NA paritesi tam, 52/55 adım iki engine'de de başarılı, 3'ü ikisinde de   │
   │ zinciri                 │                 │ aynı hatayla                                                                                      │
   ├─────────────────────────┼─────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────┤
   │ X/y hizalama (satır     │ Sağlam          │ 181→180→137→137→112→112→89, iki engine'de de birebir                                              │
   │ düşüren node'lar)       │                 │                                                                                                   │
   ├─────────────────────────┼─────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────┤
   │ Temiz sayısal → numpy → │ Sağlam          │ X max dev 3.553e-15, lr_auc/rf_acc farkı 0.0                                                      │
   │ model                   │                 │                                                                                                   │
   ├─────────────────────────┼─────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────┤
   │ P1: temporal + sayısal  │ Kırık           │ 32/32 divergent; pandas TypeError, polars sessizce fit; linear/distance 20/20 dejenere (mcc tam   │
   │ karışık frame           │                 │ 0.000, auc≈0.5); ns/us/ms fark etmiyor; native polars'ta da aynı                                  │
   ├─────────────────────────┼─────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────┤
   │ P2: son-bit float       │ Kırık (kanıtlı) │ SimpleImputer mean 1.39e-17 → StandardScaler 2.22e-16 → 132/600 hücre → SGD n_iter 15 vs 49 → mcc │
   │ gürültüsü               │                 │ +0.184 vs −0.177                                                                                  │
   ├─────────────────────────┼─────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────┤
   │ Mevcut test ağı         │ İkisini de      │ 395 passed                                                                                        │
   │                         │ yakalamıyor     │                                                                                                   │
   └─────────────────────────┴─────────────────┴───────────────────────────────────────────────────────────────────────────────────────────────────┘
   P2 için düzeltme yönü artık net: sorun bridge'de ya da dtype'ta değil, imputer/scaler'ın iki engine'de farklı toplama sırası kullanmasında. Bit-bit
   eşitlik isteniyorsa polars yolunda da pandas ile aynı indirgeme yapılmalı (maliyetli), ya da bu "tol-bazlı iterative learner'larda engine seçimi modeli
    değiştirir" diye belgelenmeli.

   Probe dosyaları tmp_polars_e2e/ altında 11 dosya (untracked). İsterseniz P1 ve P2'yi initiatives/analysis/opus_core_analysis-open_queue.md'ye
   reproduction kanıtlarıyla OC kaydı olarak yazabilirim — repo'nun mevcut formatına (tek cümlelik status + ## Log girişi archive'da) uygun şekilde.
