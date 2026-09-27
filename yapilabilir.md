# Yapılabilir

Bu projede bundan sonra yapılabilecek işlerin listesi. Her madde, `v2-real-eval`
dalındaki ölçümlerden çıktı (bkz. README "v2" bölümü ve `reports/*_v2*.json`).
Süreler bir laptop için kaba tahmindir.

Durum (2026-09-28):

- İnsan-sesi dedektörü (v2): gerçek inleme/çığlıkların %70'ini yakalıyor, insan dışı
  seslerde %18 yanlış alarm veriyor (eşik 0.45).
- Acil durum sınıflandırıcısı (v2): sentetik fısıltı/inlemede %97, **gerçek inlemede
  0/89**. Yani fısıltı/inleme sınıfları henüz gerçek seste çalışmıyor.

---

## 1. Model: yüksek etki

| # | İş | Neden | Tahmini süre |
|---|---|---|---|
| 1.1 | **Önceden eğitilmiş ses temsilleri** (YAMNet / PANNs / BEATs embedding'leri) ile insan-sesi dedektörü | MFCC + Random Forest, kedi miyavlaması ile insan inlemesini ayırmada tavana dayandı. Veri çoğaltmak bu tavanı aşmadı. | 1–2 gün |
| 1.2 | **Gerçek inleme verisiyle eğitim**: VIVAE'yi konuşmacıya göre böl (ör. 7 eğitim / 4 test) ya da ek gerçek kaynak bul | Sentetik inleme (alçak geçiren filtre + perde kaydırma) gerçek inlemeye benzemiyor. Model "boğuk ses = inleme" kuralını öğrendi. | 0.5 gün |
| 1.3 | **Gerçek fısıltı verisi**: CHAINS (ücretsiz, sunucu erişilemiyordu), wTIMIT (yazarlardan izin) ya da gönüllülerden Türkçe kayıt | Fısıltı başarısı şu an yalnızca sentetik veride biliniyor. Protokol: `docs/REAL_DATA.md` | kayıt: 1–2 gün |
| 1.4 | **Klip uzunluğu uyumsuzluğu**: acil durum modeli 2–3 sn'lik tüm kliplerle eğitiliyor, canlı sistem 1 sn'lik pencereyle çalışıyor | Eğitim ve kullanım koşulu farklı; gerçek performans raporlanandan düşük olabilir | 0.5 gün |
| 1.5 | **Vurma (tapping) tespiti**: onset analizi ile | Enkaz altından en gerçekçi sinyal vurma sesidir; betonda havadan çok daha iyi taşınır | 1 gün |
| 1.6 | Panik / stres karışıklığı | Panik recall değeri 0.39; çoğu stres sanılıyor. Oyunculu veride bu iki duygu zaten birbirine yakın | araştırma |
| 1.7 | Maliyet ağırlıkları yerine `class_weight="balanced"` | 8x/6x ağırlıklar recall değerini artırmadı, yanlış alarmı 0.17'den 0.23'e çıkardı | 10 dk |
| 1.8 | Daha fazla insan dışı veri (lisansı açık hayvan sesi setleri, şantiye/jeneratör kayıtları) | Yanlış alarmların çoğu hayvan ve su sesi | 0.5 gün |

## 2. Veri ve değerlendirme

| # | İş | Not |
|---|---|---|
| 2.1 | Nonspeech7k train arşivinin tamamı (2.3 GB) | Zenodo çok yavaştı. Yarım dosya `data/_downloads/` altında, `curl -C -` ile devam ettirilebilir |
| 2.2 | Tam enkaz testi (hafif/orta/ağır) | Son insan-sesi turu `--quick` modundaydı (sadece temiz + orta) |
| 2.3 | `rubble_acoustics.py` kalibrasyonu | Katsayılar uydurma. Gerçek bir boşlukta `sweep_rir.py` ile ölçülen birkaç IR, modeli gerçeğe bağlar |
| 2.4 | Saha testi | `docs/EVALUATION.md` bölüm 3. AFAD/AKUT eğitim enkazında kayıt |

## 3. Yazılım

| # | İş | Not |
|---|---|---|
| 3.1 | **Öznitelik önbelleği** (dosya + dönüşüm → öznitelik) | Her çalıştırma her şeyi baştan hesaplıyor (~25–30 dk). Önbellekle yeni veri seti eklemek dakikalar sürer |
| 3.2 | Fısıltı sentezini hızlandırma | LPC her karede Python döngüsüyle hesaplanıyor; en yavaş adım bu |
| 3.3 | v1/v2 kod tekrarını birleştirme | `features`/`features_v2`, `pipeline`/`pipeline_v2`, `train`/`train_v2` paralel kopyalar |
| 3.4 | GitHub Actions ile CI | 95 test var ama otomatik çalışmıyor |
| 3.5 | Temizlik | Eski kök scriptler, `data_emergency/`, `DEPREM_PROJESI_PROBLEMLER.md`, `data/_downloads/` (silme işlemini elle yapın) |
| 3.6 | `v2-real-eval` dalını `main`'e birleştirme (PR) | Model `.pkl` dosyaları commit'lenmedi (54 + 6 MB, LFS), scriptlerle yeniden üretilebilir |

## 4. Gömülü sistem (ESP32-S3)

Donanımda test edilmeyecek; amaç derlenebilen, testli ve dürüstçe belgelenmiş
bir prototip. İş fazlara bölündü:

| Faz | İş | Durum |
|---|---|---|
| ESP-1 | **C dışa aktarıcı**: CLI, model adına göre önekler (iki model aynı projede), eşiklerin float32'ye kayıpsız yazılması, sınıf sırası kontrolü, karar eşiği ve örnekleme hızının `.h`'ye yazılması, C'yi gerçekten derleyip sklearn ile karşılaştıran test | Bitti |
| ESP-2 | **Küçük model** (`scripts/train_esp.py`, `reports/esp_model_sweep.json`): ağaç sayısı x derinlik x sınıf ağırlığı taraması, flash boyutu gerçekten derlenerek ölçüldü (ARM Thumb `-Os`, Xtensa değil; yaklaşık). Seçilen: insan 10 ağaç / derinlik 12, 230 KB (dengeli doğruluk 0.836, tam v2 0.845; ama VIVAE recall 0.70 -> 0.50). Acil durum 10 ağaç / derinlik 12, ağırlıksız, 421 KB (macro-F1 temiz 0.77 / ağır enkaz 0.50, tam v2 0.80 / 0.45; yanlış alarm 0.14, tam v2 0.23). v2 maliyet ağırlıkları küçük modellerde normal konuşmanın %59'unu acil durum yaptı, bu yüzden kullanılmadı. C çıktısı sklearn ile birebir (fark < 2e-7) | Bitti |
| ESP-3 | **Firmware iskeleti** (`firmware/`, bkz. `firmware/README.md`): PlatformIO, INMP441 I2S, kayıt ve çıkarım görevleri, iki aşamalı karar ve alarm takibi (saf C, `pipeline_v2` ve `events.AlarmTracker` ile birebir test edildi), seri porttan JSON, panele köprü (`scripts/esp_serial_bridge.py`, `--replay` ile donanımsız). `pio run` henüz denenmedi (ESP-5 CI) | Bitti |
| ESP-4 | **C'de öznitelik çıkarımı**: librosa ile birebir MFCC (reflect padding, Slaney mel, `top_db=80`, ortho DCT), düzlük, HNR (sadece 44–441 gecikmeleri, tam korelasyon ESP'de saniyeler sürer), altın vektör testleri. İki örnekleme hızı (22.05 / 16 kHz) sorunu | |
| ESP-5 | **CI ve belgeler**: GitHub Actions (`pytest` + `pio run`), README'de "prototip, donanımda test edilmedi" bölümü, bağlantı tablosu, malzeme listesi | |

Açık notlar:

- 1.1 seçilirse embedding modelleri ESP32 için ağır; küçültülmüş (quantized)
  sürüm ya da hesabı operatör bilgisayarında yapmak gerekebilir.
- INMP441'in öz gürültüsü fısıltı için yüksek olabilir. Vurma sesi için
  jeofon/piezo kontak sensör kanalı eklenebilir.
- Sessizlik kapısı (`RMS_SILENCE_THRESHOLD`) mutlak seviyeye bağlı; mikrofon
  ölçeklemesi ve kalibrasyonu tanımlanmalı.
- C testleri derleyici yoksa atlanır. Windows'ta: `pip install ziglang` ve
  `CC="python -m ziglang cc"`.

---

## Önerilen sıra

1. **1.7 + 3.4 + 3.6:** kısa, düşük riskli işler.
2. **1.2:** gerçek inleme ile eğitim. Acil durum modelinin en büyük açığını kapatır.
3. **3.1:** önbellek. Sonraki bütün deneyleri hızlandırır.
4. **1.1:** önceden eğitilmiş ses temsilleri. En büyük potansiyel kazanç, ama gömülü tasarımı etkiler.
