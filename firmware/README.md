# ESP32-S3 firmware (prototip)

> **Durum: prototip, donanımda test edilmedi.** Karar mantığı, alarm takibi ve
> modeller bilgisayarda derlenip Python ile birebir karşılaştırılarak test edildi.
> Mikrofon ve FreeRTOS kısmı (`src/main.cpp`) hiç çalıştırılmadı. Öznitelik
> çıkarımı henüz yok: firmware konuşma içeren pencereleri `unclassified` diye
> raporlar, modelleri çağırmaz (bkz. [Eksikler](#eksikler)).

Enkaz altına bırakılan bir mikrofonun sesi cihaz üzerinde sınıflandırıp yalnızca
olayları (tespit, alarm) göndermesi hedefleniyor. Python tarafındaki
`pipeline_v2` + `events.AlarmTracker` hattının C karşılığıdır.

```
INMP441 ──I2S──> capture_task (çekirdek 0)
                   │ float örnekler (1 sn'lik stream buffer)
                   v
                 inference_task (çekirdek 1), 1 sn'lik pencere başına:
                   RMS < 0.0015            -> silence
                   öznitelik (33)          -> human_detector_predict  (10 ağaç, 230 KB)
                   P(insan) < 0.45         -> no_human
                   öznitelik (19)          -> emergency_classifier_predict (10 ağaç, 421 KB)
                   sınıf != normal ve 1 - P(normal) >= 0.45 -> acil durum penceresi
                   rubble_tracker: art arda 3 pencere -> tek alarm (5 sn bekleme)
                   │
                   v
                 seri port, satır başına JSON ──> scripts/esp_serial_bridge.py ──> app.py paneli
```

## Dosyalar

| Yol | İçerik |
|---|---|
| `src/main.cpp` | I2S kurulumu, kayıt ve çıkarım görevleri, JSON çıktısı |
| `include/config.h` | Pinler, örnekleme hızı, eşikler |
| `include/human_detector_model.h`, `include/emergency_classifier_model.h` | `scripts/export_c_model.py` ile `models/*_esp.pkl`'den üretildi, elle düzenlemeyin |
| `lib/rubble_core/rubble_decision.*` | İki aşamalı karar (`pipeline_v2` ile aynı) |
| `lib/rubble_core/rubble_events.*` | Bölüm/alarm takibi (`events.AlarmTracker` ile aynı) ve JSON |
| `lib/rubble_core/rubble_features.*` | RMS; öznitelik fonksiyonları şimdilik boş (ESP-4) |
| `test/sample_serial.txt` | Köprüyü donanımsız denemek için örnek seri çıktı (C çekirdeğine senaryo çalıştırılarak üretildi) |

`lib/rubble_core` donanıma bağlı değil, saf C99. Testler
(`tests/test_firmware_core.py`) onu bilgisayarda derleyip Python koduyla
karşılaştırır.

## Donanım

| Parça | Not |
|---|---|
| ESP32-S3 DevKitC-1 | PSRAM'li modül önerilir (ör. N16R8); öznitelik çıkarımı eklenince STFT tamponları büyüyecek |
| INMP441 I2S MEMS mikrofon | 3.3 V |

| INMP441 | ESP32-S3 |
|---|---|
| VDD | 3V3 |
| GND | GND |
| L/R | GND (sol kanal) |
| SCK | GPIO 4 |
| WS | GPIO 5 |
| SD | GPIO 6 |

Pinler `include/config.h` içinden değiştirilebilir.

## Derleme

[PlatformIO](https://platformio.org/) gerekir.

```bash
cd firmware
pio run                        # derle
pio run -t upload              # yükle
pio device monitor             # seri çıktıyı izle
```

Modeller değişirse başlık dosyalarını yeniden üretin (repo kökünden):

```bash
python scripts/train_esp.py            # küçük modelleri yeniden eğit (isteğe bağlı, ~10 dk)
python scripts/export_c_model.py models/human_detector_esp.pkl --name human_detector
python scripts/export_c_model.py models/emergency_classifier_esp.pkl --name emergency_classifier
```

`tests/test_firmware_core.py`, repodaki başlık dosyalarının `.pkl` dosyalarıyla
uyuşmadığı durumda başarısız olur.

## Seri protokol

115200 baud, satır başına bir JSON nesnesi. Zaman (`t_ms`) cihaz açıldığından
beri geçen milisaniyedir; cihazda gerçek saat yok.

| `type` | Ne zaman | Alanlar |
|---|---|---|
| `boot` | Açılışta | `sample_rate`, ağaç sayıları, eşikler, `features` |
| `status` | Saniyede bir | `windows`, `last_rms`, `last_status`, `inference_ms`, `overruns`, `open_episode`, `free_heap` |
| `detection` | Acil durum penceresi | `episode`, `window`, `state`, `state_confidence`, `human_prob`, `emergency_prob` |
| `alarm` | Bölümde 3. pencere | `episode`, `episode_start_ms`, `windows`, `states`, `peak_emergency_prob` |
| `episode_end` | Normal konuşma / insan sesi yok | `episode`, `windows`, `alarmed` |
| `error` | Başlatma hatası | `message` |

## Panele bağlama

`scripts/esp_serial_bridge.py`, olayları `logs/events.jsonl` dosyasına,
durumu `logs/listener_status.json` dosyasına yazar; panel (`streamlit run
app.py`) ESP'yi canlı dinleyici gibi gösterir.

```bash
pip install pyserial
python scripts/esp_serial_bridge.py --port COM3          # Linux: /dev/ttyACM0

# Donanım olmadan: örnek çıktıyı yarım saniye arayla oynat
python scripts/esp_serial_bridge.py --replay firmware/test/sample_serial.txt --delay 0.5
```

## Eksikler

Planın geri kalanı `yapilabilir.md` bölüm 4'te.

- **Öznitelik çıkarımı (ESP-4).** Modeller librosa ile hesaplanan özniteliklerle
  eğitildi. C tarafı aynı sayıları üretmeden modeller kullanılamaz.
- **İki örnekleme hızı.** İnsan modeli 22050 Hz, acil durum modeli 16000 Hz
  bekliyor. Mikrofon 22050 Hz'de okunuyor; acil durum modeli için yeniden
  örnekleme gerekiyor.
- **Kalibrasyon.** Sessizlik eşiği (0.0015) mutlak seviyeye bağlı. INMP441'in
  24 bit verisi [-1, 1) aralığına ölçekleniyor, ama bu ölçek Python'da
  eğitimde kullanılan kayıtlarla karşılaştırılmadı.
- **RMS farkı.** Cihaz tüm pencerenin RMS'ini alıyor; `pipeline_v2` ise
  librosa karelerinin RMS ortalamasını. Sessizlik kapısı için fark küçük, ama
  birebir değil.
- **Doğruluk.** Küçük modeller tam v2 modellerinden zayıf; gerçek
  inleme/çığlık yakalama 0.70'ten 0.50'ye düşüyor
  (`reports/esp_model_sweep.json`). v2'nin gerçek veride yaşadığı sorunlar
  (ör. gerçek inlemeler) burada da geçerli.
- **İletişim.** Olaylar yalnızca USB seri porttan gidiyor. Sahada LoRa ya da
  ESP-NOW gerekir.
- **Güç.** Pil ve uyku modu yok.
- **Donanım testi yok.** I2S ayarları (kanal seçimi, bit hizalama) gerçek
  mikrofonla doğrulanmadı.
