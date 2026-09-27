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

| # | İş | Not |
|---|---|---|
| 4.1 | C dışa aktarıcıdaki eşik hassasiyeti | `export_c_model.py` eşikleri `%.6f` ile yazıyor; küçük ölçekli özniteliklerde bölme yönü değişebilir. `%.9g` kullanılmalı |
| 4.2 | C kodunu gerçekten derleyen test | Mevcut "birebir eşleşme" testi C'yi derlemiyor, sklearn'ü kendisiyle karşılaştırıyor |
| 4.3 | Model boyutu | 350 ağaç × derinlik 22 flash'a sığmayabilir. Küçük model ile doğruluk kaybını ölçmek gerekiyor |
| 4.4 | Öznitelik çıkarımının C'ye taşınması | Asıl zorluk ağaçlar değil, MFCC/HNR hesabının librosa ile birebir eşleşmesi |
| 4.5 | 1.1 seçilirse | Embedding modelleri ESP32 için ağır. Küçültülmüş (quantized) sürüm veya hesabı operatör bilgisayarında yapmak gerekebilir |
| 4.6 | Donanım | INMP441'in öz gürültüsü fısıltı için yüksek olabilir. Vurma sesi için jeofon/piezo kontak sensör kanalı eklenebilir |

---

## Önerilen sıra

1. **1.7 + 3.4 + 3.6:** kısa, düşük riskli işler.
2. **1.2:** gerçek inleme ile eğitim. Acil durum modelinin en büyük açığını kapatır.
3. **3.1:** önbellek. Sonraki bütün deneyleri hızlandırır.
4. **1.1:** önceden eğitilmiş ses temsilleri. En büyük potansiyel kazanç, ama gömülü tasarımı etkiler.
