   A. ML yetenek boşlukları (skyulf-core)

   1. Konformal tahmin / belirsizlik aralıkları — conformal|prediction_interval|uncertainty için core+backend'de sıfır eşleşme var. Oysa ürünün sloganı "Stop trusting
   your pipeline. Verify it." — eksik parça tam olarak "ne zaman güvenmeyeceğini de söyle". Split-conformal sklearn-native (yeni ağır bağımlılık yok), mevcut
   calibrated_classifier + thresholds.py hikâyesinin doğal devamı. Çıktı: regresyonda aralık, sınıflandırmada risk kontrollü tahmin kümesi; /deployment/predict yanıtına
   lower/upper alanı, Experiments'a "Uncertainty" sekmesi (kapsam eğrisi). Efor: M. Değer: çok yüksek, farklılaştırıcı.

   2. Adillik / bias denetimi — fairlearn|disparate|demographic_parity|equalized_odds: sıfır eşleşme. Korunan bir kolon üzerinde y_true/y_pred/proba'dan saf numpy ile
   hesaplanır: demografik parite, equalized odds, grup-bazlı kalibrasyon, disparate impact. WOEEncoder + calibrated_classifier + threshold tuning zaten var → kredi
   skorlama kitlesi açıkça hedeflenmiş, o kitlede bu metrikler yasal zorunluluk. Efor: S–M. Değer: yüksek (regüle sektör satışı).

   3. Anomaly detection'ı modelleme düğümü yapmak — IsolationForest şu an sadece EDA profiler içinde (_analyzer/multivariate.py), EllipticEnvelope sadece outlier silme
   preprocessing'i. Yani tespit var ama servis yolu yok. metrics.py:462 zaten -1 noise etiketlerini yönetiyor; DBSCAN/HDBSCAN/OneClassSVM/LOF düğümleri + deployment'ta
   skorlama → dolandırıcılık, kestirimci bakım, izleme gibi baştan aşağı yeni bir kullanım alanı mevcut canvas ile açılır. Efor: M. Değer: yüksek.

   4. Boyut indirgeme düğümleri (PCA / TruncatedSVD / UMAP) — preprocessing/ altında hiç yok; PCA yalnızca EDA'da. Pratik sonuç: klasik TF-IDF → TruncatedSVD →
   sınıflandırıcı metin hattı bugün kurulamıyor. Ayrıca yüksek boyutta tuning maliyetini düşürür. Mevcut _artifacts.py TypedDict kalıbına birebir oturur. Efor: S. Değer:
   orta-yüksek, çok ucuz.

   5. SHAP'sız açıklanabilirlik: permutation importance + PDP/ICE — permutation_importance|partial_dependence: sıfır eşleşme, tek explainer SHAP ve o opsiyonel extra
   (skyulf-core/setup.py:78). Yani shap kurulu değilse kullanıcı hiçbir açıklama göremiyor. SklearnBridge.to_sklearn + StatefulEstimator zaten var, sklearn.inspection
   doğrudan çalışır. PDP/ICE ayrıca "bu müşteri için ne değişmeli" sorusunu cevaplar — threshold tuning'in doğal eşi. Efor: S–M. Değer: yüksek.

   6. Hiperparametre önem analizi (mevcut trial verisinden) — En ucuz/yüksek getirili fikir bu olabilir: her tuning job'ı tüm trial'ları zaten saklıyor (strategies.py:255
    → job.results, /jobs/{id}/trials, 128 job × 2000 trial LRU buffer). Üzerinde analiz yapılmıyor, sadece çiziliyor. Optuna'nın kendi fANOVA/importance API'si veya
   trial'lara uydurulmuş bir surrogate ile "hangi düğüm gerçekten önemliydi, hangisine hiç dokunmasan da olurdu" paneli. Sıfır yeni altyapı, mevcut veri → içgörü. Efor:
   S. Değer: yüksek.

   7. Quantile / Tweedie / Poisson / Huber regresörler + pinball loss — regression.py'de 14 model var ama hiçbiri bunlar değil. Sigorta fiyatlama, talep tahmini, çarpık
   hedefler için standart aileler; hepsi sklearn estimator'ü, sadece hyperparameters/_registry.py'ye search space + _evaluation/metrics.py'ye pinball loss eklemek
   gerekiyor. Efor: S. Değer: orta.

   8. Multi-label / multi-output desteği — hiç yok (LabelBinarizer/MultiOutput yok). Ürün segmentasyonu, çoklu churn, etiketleme gibi gerçek bir problem sınıfı kapalı.
   task_type çıkarsaması (_analyzer/target.py), metrikler ve canvas'ı birlikte etkiler. Efor: L. Değer: yüksek ama geniş yüzey.

   B. Üretim / platform boşlukları (backend)

   9. Sunucu tarafı prediction log — şu an hiç yok. grep prediction_log|log_prediction → sıfır. POST /api/deployment/predict session alıyor ama sadece active deployment'ı
    okuyor; ne tahmin edildiğine dair sunucuda hiçbir kayıt yok. Tahmin geçmişi tarayıcının localStorage'ında duruyor. Bu tek başına bir özellik değil, üç özelliğin ön
   koşulu: inference denetimi, prediction drift, canlı performans ölçümü. Bir prediction_logs tablosu (deployment_id, model_version, hash'lenmiş feature vektörü, skor,
   eşik, latency) bunu açar. Efor: M. Değer: kritik — bu olmadan "production MLOps" iddiası eksik.

   10. Gerçek sonuç (actuals) geri beslemesi → canlı performans & konsept drift — #9'un üzerine: POST /deployment/feedback ile gözlenen gerçekleri loglanmış tahminlere
   bağla. Böylece feature drift değil, model performansı drift'i ölçülür (bugün core'da yalnızca KS/PSI/Wasserstein feature+schema drift var; roadmap'teki DRIFT-01 de
   hâlâ feature temsilini tartışıyor, bu farklı bir eksen). Çıktı: "modelin gerçek dünyadaki AUC'u son 30 günde 0.84 → 0.79" + otomatik yeniden eğitim tetikleyicisi.
   Efor: M. Değer: çok yüksek.

   11. Toplu (batch) skorlama job'ı — predict 10.000 satırla sınırlı ve interaktif; security.py dokümanı bulk scoring'i açıkça "pipeline execution'a bırakılmış" diye
   geçiştiriyor. Yani platform eğitip deploy edebiliyor ama ölçekli skorlayamıyor. Celery task: DataSource oku → deployed artifact'i uygula → sonucu yeni DataSource
   olarak yaz. Mevcut ArtifactStore, SmartCatalog ve job altyapısı yeter. Efor: M. Değer: çok yüksek — en sık gerçek üretim senaryosu.

   12. Alembic migration altyapısı — alembic.ini yok, migrations/ yok; şema açılışta create_tables() (metadata create-all) ile kuruluyor. Sonuç: self-hosted bir kurulumu
   sürüm yükseltirken şema değişimi veri kaybı veya hata demek. Roadmap R10 hedefi "v1.0 self-hostable & authenticated" — migration olmadan bu hedef tutmaz. 12 tablo,
   bilinen ve sınırlı bir iş. Efor: M. Değer: yüksek (görünmez ama bloklayıcı).

   13. Uyarı teslim kanalları (webhook → Slack → e-posta) — slack|webhook|smtp|send_email|notification grep'i backend'de sadece UI-toast yorumlarını buluyor. Drift
   alert'lerinin tam bir yaşam döngüsü var (severity/acknowledge/resolve/owner/disposition_history) ama kimseye haber vermiyor — yani birinin sürekli ekrana bakması
   gerekiyor. Küçük bir Notifier soyutlaması + önce webhook (SMTP karmaşası yok), hem drift hem job-failure için. Celery beat zaten kurulu. Efor: S. Değer: yüksek, çok
   düşük maliyet.

   14. Prometheus /metrics + OpenTelemetry — prometheus|/metrics sıfır eşleşme; sadece opsiyonel Sentry var. Platform yavaş düğümleri ve hataları kendi içinde takip
   ediyor (slow-nodes, error_events) ama gerçek bir altyapıya dışarıdan bakılamıyor. prometheus-fastapi-instrumentator + Celery exporter ile başlanır; mevcut
   execution_time ve peak_memory_bytes metrikleri doğrudan expose edilir. Efor: S–M. Değer: orta-yüksek (kurumsal benimseme ön koşulu).

   15. Kaynak ve enerji muhasebesi — peak_memory_bytes zaten toplanıyor (preprocessing/base.py:221, _pipeline.py:92), execution_time da job metrics'inde. Eksik olan
   sadece birleştirip sunmak: job başına CPU-saniye, tepe bellek, tahmini enerji/CO₂ ve bunların Dashboard + SlowNodes'ta gösterimi. "Polars-first performance"
   pazarlaması yapan bir ürünün kendi maliyetini ölçebilmesi hem FinOps hem yeşil-BT raporlaması için somut bir fark. Efor: S. Değer: orta, veri %80 hazır.

   C. Canvas / UX

   16. Graph JSON içe/dışa aktarma — export sadece PNG/SVG/notebook. Canvas grafiği /pipeline/save ile zaten JSON olarak gidip geliyor, yani dosyaya yazmak/okumak çok
   ucuz. Kazanımı büyük: kurulumlar arası taşınabilirlik, PR/issue'ya pipeline ekleyebilmek, yedekleme ve topluluk şablonu paylaşımı (mevcut 5 şablonun ötesine geçmenin
   ön koşulu). Not: roadmap'te değil. Efor: S. Değer: yüksek/oran çok iyi.

   17. Canvas minimap + grup/çerçeve/annotation — MiniMap grep'i frontend'de sıfır render; React Flow'da tek satır. Gruplama/alt-akış/çerçeve/yorum da hiç yok. Büyük
   pipeline'larda gezinme ve pipeline'ı başkasına anlatma (çerçeve + not) bugün mümkün değil; review kültürü olan ekipler için doğrudan engel. Efor: minimap S,
   grup/çerçeve M. Değer: orta-yüksek.

   18. Versiyon yüklemeden önce diff göster — graphDiff.ts ve PipelineDiffView.tsx zaten yazılmış ama yalnızca iki tamamlanmış job arasında kullanılıyor. Bir
   PipelineVersion'ı geri yüklerken kullanıcı sadece bir onay diyaloğu görüyor → mevcut iş sessizce ezilebilir. Var olan kodu ikinci bir yere bağlamak. Efor: S. Değer:
   orta (veri kaybı önleme).

   19. Global arama (Ctrl+K her sayfada) — CommandPalette yalnızca canvas düğümü arıyor. Job/dataset/model/drift-alert/error adına göre atlama yok; 11 sayfalık bir
   uygulamada bu hissedilir bir sürtünme. Efor: S–M. Değer: orta.

   20. Düğümden dokümantasyona derin bağ + inline parametre yardımı — docs/user_guide/ 21 sayfa, ama node ayarlarında sadece tek satır açıklama ve 2 sekmelik
   HelpGuideModal var. Doküman zaten yazılmış; eksik olan ihtiyaç anına bağlamak (parametre alanının yanındaki ? → ilgili kılavuz bölümü). Efor: S. Değer: orta,
   activation'a doğrudan katkı.

   D. Mühendislik kalitesi

   21. Engine-parity otomatik fark denetleyicisi — initiatives/dual-engine-correctness/ içinde 49 bulgu ve 0 uygulanmış düzeltme var. Tek tek düzeltmek yerine:
   registry'deki her düğümü üretilmiş (Hypothesis) frame'ler üzerinde iki motorda da çalıştırıp sonuçları toleransla karşılaştıran bir harness. 49 bulguyu tek tek bulmak
   yerine sınıfın tamamını ortaya çıkarır ve OC-64/OC-65 gibi kalanlara da doğrudan cevap verir. Mevcut test_registry_contract.py bunun iskeleti. Efor: M. Değer: çok
   yüksek (bu repo'nun gerçek ağrı noktası).

   22. Metamorfik düğüm sözleşmeleri — Repo'nun kendi kaydı: OC-12/163/165/166'nın hepsi aynı desenden çıktı (X satır kaybetti, y korudu, model yanlış etiketle sessizce
   eğitildi). Bu bir tesadüf değil, test stratejisi boşluğu. Property tabanlı invariant'lar: "satır-filtreleyen düğüm değilse len(apply(X)) == len(X)", "y her zaman
   select_rows_by_position'dan geçti", "stateless transform'ta fit→apply idempotent". F841'in ortaya çıkardığı "3 test hiç assertion'sız geçmiş" vakası da aynı ailenin
   üyesi. Efor: M. Değer: çok yüksek — tekrarlayan hata sınıfını kapatır.

   23. Entry-point tabanlı node plugin API'si — NodeRegistry.register import yan etkisiyle çalışıyor, yani üçüncü taraf bir düğüm paketi fork etmeden eklenemiyor.
   importlib.metadata.entry_points("skyulf.nodes") keşfi + Apache-2.0 core üzerinden topluluk düğüm paketleri. Açık-çekirdek iş modeline ve landing page'de yıldız/indirme
    sayacı gösteren bir büyüme stratejisine doğrudan hizmet eder. (Roadmap R5'teki "schema-driven renderer" bunun frontend ön koşulu — çakışma değil, tamamlayıcı.) Efor:
   M. Değer: orta-yüksek, ekosistem bahsi.

   E. Ürün / pozisyonlama

   24. Pipeline Verification Report — Parçaların hepsi var ama hiçbir yerde birleşmiyor: leakage doğrulaması, fingerprint() (semantik mühür), export_model_card(),
   per-fold preprocessing refit, şema doğrulama, node summary'leri, mermaid diyagram. Bunları tek bir pass/fail skor kartı (HTML/PDF) halinde toplayan bir çıktı, markanın
    somut teslim edilebilir nesnesi olur: bir data scientist bunu reviewer/denetçiye götürebilir. Yeni yetenek değil, montaj. (Not: frontend-consumer-design'daki "report
   export" maddesiyle kısmi çakışma — orada EDA raporu kastediliyor, burada doğrulama/denetim raporu.) Efor: S–M. Değer: yüksek, konumlandırma açısından en güçlüsü.

   25. Kredi riski / skor kartı dikey paketi — WOE + Information Value (encoding/woe.py:180), kalibrasyon, threshold tuning, calibration/gains eğrileri zaten mevcut.
   Eksik olanlar az ve ucuz: ayrıştırma için KS istatistiği, Gini (2·AUC−1), skor bantlarında SOM/PSI, skor kartı tarzı katsayı tablosu, ve #2'deki adillik kontrolü.
   Bunlar bir araya gelince "görsel bir araçla regüle skor kartı üret" diye satın alınan bir dikey oluşur. Efor: M. Değer: yüksek (gelir yolu).
