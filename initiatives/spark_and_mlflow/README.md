# Spark ve MLflow çalışma kayıtları

Güncelleme: 11 Ekim 2026.

Bu klasör iç planları, uygulama kararlarını ve doğrulama kanıtlarını tutar.
Yeni çalışma için [219–222 uygulama sırası ve takip](223-delivery-order-tracker.md)
belgesinden başla. Bu kapsamda **0/8 adım tamamlandı**; ilk adım **S01 — mevcut
TRIGGERED yayınlama ile online model/endpoint kurulumunu Bundle'a bağlamak**.

Projenin genel görev durumu ve diğer bağımlılıkları için
[OPEN_QUEUE_updated.md](OPEN_QUEUE_updated.md) esas alınır. 223 yalnız seçilen
219–222 kapsamının sırasını takip eder; genel kuyruktaki işleri kapatmaz.

## Güncel belgeler

| Belge | Rolü |
| --- | --- |
| [223 — Uygulama sırası ve takip](223-delivery-order-tracker.md) | S01–S08 sırası, ön koşullar, kabul ölçütleri ve ilerleme. |
| [219 — Online kurulum ve yayın modları](219-online-serving-setup-continuous-publication.md) | Online lookup ayarlarını model/endpoint kurulumuna bağlama ve CONTINUOUS publication tasarımı. |
| [220 — Template yetenek karşılaştırması](220-dbml-template-capability-comparison.md) | Var olan, kısmi ve eksik yeteneklerin envanteri; tek bir uygulama görevi değildir. |
| [221 — Entegrasyon ve kabul](221-integration-completion-and-acceptance-plan.md) | Feature güncellemeleri, rollout/champion geçişi ve MLflow dataset Inputs için ortak kabul planı. |
| [222 — Monitoring ayrıntıları](222-monitoring-detail-views-gap-plan.md) | Mevcut kanıtların tablo ve ayrıntı görünümlerinde sunulması. |
| [224 — Klasör temizliği](224-folder-cleanup.md) | Kaldırılan geçici çıktılar ve eski planlar; korunan kanıtlar ve temizlik kontrolü. |

223'teki isteğe bağlı işler ve ertelenen kapsamlar ana sekiz adımdan ayrıdır.
Dağıtık model eğitimi ile sürekli tahmin akışı bu sıraya dahil değildir.

## Önceki çalışmaların kanıtları

Numaralı teslimat, kabul ve native doğrulama raporları kendi çalışma anlarını
belgeler. Kaynak commit, wheel, test komutu ve Databricks run bilgisi ilgili
rapordan okunmalıdır. Eski bir planın gelecek zamanlı anlatımı, bugünkü eksik
listesi olarak kullanılmamalıdır.

- [HANDOFF](HANDOFF.md), [ARCHITECTURE](ARCHITECTURE.md) ve
  [VALIDATION](VALIDATION.md): önceki devir, mimari karar ve doğrulama kayıtları.
- [İlk hazırlık incelemesi](reports/2026-09-21-spark-databricks-readiness.md) ve
  [ilk kaynak envanteri](reports/2026-09-21-spark-core-inventory.md): tarihli
  araştırma kayıtları; güncel durum için kuyruk ve ilgili teslimat raporu okunur.
- `rehearsals/`: korunan doğrulama betikleri, loglar, paketler ve sonuç kayıtları.
  Yeniden üretilebilen pytest geçici çıktıları tarihsel kanıtların yerine geçmez.

Kullanıcı dokümantasyonu [docs](../../docs/index.md) altındadır. İç görev
numaraları ve kişisel makine yolları public rehberlere taşınmaz.

## Çalışma ve tamamlama kuralları

- Uygulamadan önce güncel kodu ve görev bağımlılıklarını kontrol et.
- Yazılmış plan, yerel test, Databricks kabulü ve üretim ortamı kabulünü ayrı
  durumlar olarak kaydet. İlgili kod değiştiğinde eski kanıtı güncel sayma.
- Kapanışa komut, sonuç ve doğrulanan revision/wheel bilgisi ekle; native çalışma
  varsa run/resource bağlantısını da belirt.
- Temizlikte aktif görevleri, başvurulan belgeleri ve benzersiz sonuç kanıtlarını
  koru. Dosyanın eski veya görevin bitmiş olması tek başına silme gerekçesi değildir.
- Commit ve push ayrı işlemlerdir; kullanıcının güncel talimatını izle.
