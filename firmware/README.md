# ESP32-S3 firmware (prototip)

> **Durum: prototip. Gerçek donanımda hiç çalıştırılmadı ve çalıştırılması
> planlanmıyor.** Bunun yerine simülasyonla test edildi
> (bkz. [Nasıl test edildi](#nasıl-test-edildi)): C kodu bilgisayarda Python ile
> birebir karşılaştırıldı, firmware gerçek ses dosyalarıyla bilgisayarda
> simüle edildi ve ESP32-S3 için derlendi. Wokwi'de (simüle ESP32-S3)
> çalıştırmak için senaryo hazır.
> Cihaz üzerinde öznitelik çıkarımı henüz yok (bkz. [Eksikler](#eksikler)).

Enkaz altına bırakılan bir mikrofonun sesi cihaz üzerinde sınıflandırıp yalnızca
olayları (tespit, alarm) göndermesi hedefleniyor. Python tarafındaki
`pipeline_v2` + `events.AlarmTracker` hattının C karşılığıdır.

```
INMP441 ──I2S──> capture_task (çekirdek 0)
                   │ float örnekler (1 sn'lik stream buffer)
                   v
                 inference_task (çekirdek 1), 1 sn'lik pencere başına rubble_classify():
                   RMS < 0.0015            -> silence
                   öznitelik (33)          -> human_detector_predict  (10 ağaç, 230 KB)
                   P(insan) < 0.45         -> no_human
                   öznitelik (19)          -> emergency_classifier_predict (10 ağaç, 421 KB)
                   sınıf != normal ve 1 - P(normal) >= 0.45 -> acil durum penceresi
                   rubble_tracker: art arda 3 pencere -> tek alarm (5 sn bekleme), LED
                   │
                   v
                 seri port, satır başına JSON ──> scripts/esp_serial_bridge.py ──> app.py paneli
```

## Dosyalar

| Yol | İçerik |
|---|---|
| `src/main.cpp` | Mikrofon modu (I2S, kayıt ve çıkarım görevleri) ve Wokwi senaryo modu; JSON çıktısı, alarm LED'i |
| `include/rubble_pipeline.h` | `rubble_classify()`: bir pencerenin sınıflandırılması. Cihaz, bilgisayar simülatörü ve Wokwi aynı fonksiyonu çağırır |
| `include/config.h` | Pinler, örnekleme hızı, eşikler |
| `include/human_detector_model.h`, `include/emergency_classifier_model.h` | `scripts/export_c_model.py` ile `models/*_esp.pkl`'den üretildi |
| `include/sim_scenario.h` | Wokwi senaryosu: 19 pencerenin öznitelikleri ve beklenen sonuçları (`scripts/esp_simulate.py` üretir) |
| `lib/rubble_core/` | Saf C99, donanımdan bağımsız: karar (`pipeline_v2`), alarm takibi (`events.AlarmTracker`), JSON, RMS (librosa ile aynı); öznitelik fonksiyonları boş (ESP-4) |
| `sim/sim_main.c` | Bilgisayar simülatörünün C tarafı |
| `wokwi.toml`, `diagram.json` | Wokwi: ESP32-S3 DevKitC-1 + GPIO 7'de alarm LED'i |
| `test/sample_serial.txt` | Köprüyü donanımsız denemek için seri çıktı (simülatörün gerçek çıktısı) |

## Nasıl test edildi

Gerçek kart ve mikrofon kullanılmadı. Her katman bir öncekinin kapsamadığı şeyi sınar.

| Katman | Ne sınanıyor | Nasıl | Sonuç |
|---|---|---|---|
| 1. C birim testleri (`tests/test_firmware_core.py`, `tests/test_c_model_export.py`) | Karar mantığı, alarm takibi, RMS, JSON, dışa aktarılan modeller | `lib/rubble_core` ve modeller bilgisayarda derlenip gerçek Python koduyla karşılaştırılır: `PipelineV2` (400 durum), `AlarmTracker` (1.500 rastgele pencere), `librosa.feature.rms`, sklearn `predict_proba` (18.036 test satırı) | Birebir aynı (olasılık farkı < 2e-7) |
| 2. Bilgisayar simülatörü (`scripts/esp_simulate.py`) | Firmware hattının gerçek ses kayıtlarındaki davranışı | Ses dosyası -> 1 sn pencereler -> öznitelikler (Python) -> cihazın C kodu -> olaylar; her pencere Python hattıyla karşılaştırılır | Repodaki 4 örnekte 26/26 pencere aynı (`reports/esp_simulation.json`) |
| 3. Wokwi (simüle ESP32-S3) | Firmware'in gerçek Xtensa kodu olarak açılması, FreeRTOS görevleri, modellerin cihaz işlemcisinde çalışması, seri çıktı, alarm LED'i | `pio run -e wokwi`; 19 pencerelik senaryo saniyede bir oynatılır, cihaz her kararı bilgisayar simülatörünün sonucuyla karşılaştırır, sonunda `SIM_DONE OK` basar | Senaryo firmware'i CI'da derleniyor; Wokwi'de çalıştırma bir Wokwi hesabı gerektirir (VS Code ya da CI token'ı, aşağıda) |
| 4. CI derlemesi | Mikrofon firmware'inin ESP32-S3 için derlenmesi | GitHub Actions, `pio run` | Flash 1.08 MB / 3 MB, statik RAM 23 KB |

Simülasyonların **sınamadığı** şeyler: I2S mikrofon sürücüsü ve INMP441 ayarları
(Wokwi ESP32-S3'te I2S ve mikrofon simüle etmiyor), gerçek zamanlama ve bellek
baskısı, güç tüketimi, cihaz üzerinde öznitelik çıkarımı (yazılmadı).

### Bilgisayar simülatörü

C derleyicisi gerekir (gcc / clang; Windows'ta en kolayı `pip install ziglang`).

```bash
python scripts/esp_simulate.py samples/people_talk.mp3
python scripts/esp_simulate.py samples/*.mp3 --report reports/esp_simulation.json
python scripts/esp_simulate.py kayit.wav --dashboard --delay 1   # panelde canlı izle: streamlit run app.py
```

Repodaki örneklerle sonuç (`reports/esp_simulation.json`), modelin
zayıflıklarını da gösteriyor:

| Kayıt | Sonuç |
|---|---|
| `people_talk.mp3` | 5 pencerenin 4'ü normal konuşma, 1'i stres (tek pencere, alarm yok) |
| `woman_scream.mp3` | Çığlığın çoğu "insan sesi değil" (küçük insan dedektörü sözsüz sesleri kaçırıyor) |
| `bird.mp3` | 4 pencere yanlışlıkla fısıltı/stres; araya bir "insan değil" girdiği için alarm yok |
| `car_start.mp3` | 5/5 "insan sesi değil" |

### Wokwi

[Wokwi](https://wokwi.com) ESP32-S3'ü simüle eder, ama I2S ve mikrofonu
simüle etmez. Bu yüzden `env:wokwi` derlemesi (`RUBBLE_SIM_SCENARIO`) mikrofon
yerine `include/sim_scenario.h`'deki pencereleri oynatır. Senaryo:

1. `samples/people_talk.mp3`: normal konuşma
2. JL Corpus (CC0), test bölmesindeki bir konuşmacının öfkeli kayıtları: iki kez art arda 3 stres penceresi, **iki alarm, LED yanar**
3. `samples/car_start.mp3`: insan sesi değil

Öznitelikler Python'da çıkarıldığı için Wokwi, öznitelik adımı dışındaki
her şeyi cihaz işlemcisinde çalıştırır. Her pencere için bir `sim_check`
satırı basılır (`match` alanı bilgisayar simülatörüyle karşılaştırmadır).

VS Code ile:

1. [PlatformIO](https://platformio.org/install/ide?install=vscode) ve
   [Wokwi Simulator](https://marketplace.visualstudio.com/items?itemName=wokwi.wokwi-vscode)
   eklentilerini kurun, Wokwi lisansını etkinleştirin (ücretsiz hesap).
2. `firmware/` klasörünü açın, `pio run -e wokwi` ile derleyin.
3. `F1` -> **Wokwi: Start Simulator**. Seri monitörde JSON satırları, devrede alarm LED'i.

CI'da: repo ayarlarına `WOKWI_CLI_TOKEN` gizli değişkeni eklenirse
([Wokwi CI](https://wokwi.com/dashboard/ci)), her push'ta senaryo çalışır ve
`SIM_DONE OK` beklenir.

Senaryoyu yeniden üretmek için (`data/` içinde JL Corpus gerekir; bkz. ana README):

```bash
python scripts/esp_simulate.py samples/people_talk.mp3 \
    data/human/jl_corpus/female2_angry_10a_1.wav data/human/jl_corpus/female2_angry_10a_2.wav \
    data/human/jl_corpus/female2_angry_10b_1.wav data/human/jl_corpus/female2_angry_10b_2.wav \
    data/human/jl_corpus/female2_angry_11a_1.wav data/human/jl_corpus/female2_angry_11a_2.wav \
    samples/car_start.mp3 --gap 0 --export-scenario
```

`tests/test_esp_simulate.py`, senaryodaki beklenen sonuçların güncel
modellerle tutarlı olduğunu denetler.

## Donanım (hedef, denenmedi)

| Parça | Not |
|---|---|
| ESP32-S3 DevKitC-1 | PSRAM'li modül önerilir (ör. N16R8); öznitelik çıkarımı eklenince STFT tamponları büyüyecek |
| INMP441 I2S MEMS mikrofon | 3.3 V |
| LED + 220 Ω | Alarm göstergesi |

| Bağlantı | ESP32-S3 |
|---|---|
| INMP441 VDD | 3V3 |
| INMP441 GND, L/R | GND (L/R = GND: sol kanal) |
| INMP441 SCK | GPIO 4 |
| INMP441 WS | GPIO 5 |
| INMP441 SD | GPIO 6 |
| LED (+ direnç) | GPIO 7 |

Pinler `include/config.h` içinden değiştirilebilir.

## Derleme

[PlatformIO](https://platformio.org/) gerekir.

```bash
cd firmware
pio run                        # mikrofon firmware'i
pio run -e wokwi               # Wokwi senaryo firmware'i
pio run -t upload && pio device monitor    # gerçek kart olsaydı
```

Modeller değişirse başlık dosyalarını ve senaryoyu yeniden üretin (repo kökünden):

```bash
python scripts/train_esp.py            # küçük modelleri yeniden eğit (isteğe bağlı, ~10 dk)
python scripts/export_c_model.py models/human_detector_esp.pkl --name human_detector
python scripts/export_c_model.py models/emergency_classifier_esp.pkl --name emergency_classifier
```

Testler, repodaki başlık dosyaları `.pkl`'lerle ya da senaryo güncel
modellerle uyuşmazsa başarısız olur.

## Seri protokol

115200 baud, satır başına bir JSON nesnesi. Zaman (`t_ms`) cihaz açıldığından
beri geçen milisaniyedir; cihazda gerçek saat yok.

| `type` | Ne zaman | Alanlar |
|---|---|---|
| `boot` | Açılışta | `mode`, `sample_rate`, ağaç sayıları, eşikler, `features` |
| `status` | Saniyede bir | `windows`, `last_rms`, `last_status`, `inference_ms`, `overruns` (düşen I2S okuması), `dropped_windows` (içinde örnek düştüğü için atılan pencere), `i2s_errors`, `open_episode`, `free_heap` |
| `detection` | Acil durum penceresi | `episode`, `window`, `state`, `state_confidence`, `human_prob`, `emergency_prob` |
| `alarm` | Bölümde 3. pencere | `episode`, `episode_start_ms`, `windows`, `states`, `peak_emergency_prob` |
| `episode_end` | Normal konuşma / insan sesi yok | `episode`, `windows`, `alarmed` |
| `sim_check`, `sim_done` | Yalnızca Wokwi senaryosu | `window`, `status`, `state`, `inference_us`, `match` / `mismatches` |
| `error` | Başlatma hatası | `message` |

## Panele bağlama

`scripts/esp_serial_bridge.py`, olayları `logs/events.jsonl` dosyasına,
durumu `logs/listener_status.json` dosyasına yazar; panel (`streamlit run
app.py`) ESP'yi canlı dinleyici gibi gösterir.

```bash
pip install pyserial
python scripts/esp_serial_bridge.py --port COM3          # gerçek kart; Linux: /dev/ttyACM0

# Donanım olmadan: simülatör çıktısını yarım saniye arayla oynat
python scripts/esp_serial_bridge.py --replay firmware/test/sample_serial.txt --delay 0.5
```

## Eksikler

Planın geri kalanı `yapilabilir.md` bölüm 4'te.

- **Öznitelik çıkarımı (ESP-4).** Modeller librosa ile hesaplanan özniteliklerle
  eğitildi. Cihaz aynı sayıları üretmeden mikrofon modunda sınıflandırma
  yapılamaz; şimdilik konuşma içeren pencereler `unclassified` raporlanır.
  Simülasyonlarda bu adım Python'da yapılıyor.
- **İki örnekleme hızı.** İnsan modeli 22050 Hz, acil durum modeli 16000 Hz
  bekliyor. Mikrofon 22050 Hz'de okunuyor; acil durum modeli için yeniden
  örnekleme gerekiyor.
- **Kalibrasyon.** Sessizlik eşiği (0.0015) mutlak seviyeye bağlı. INMP441'in
  24 bit verisi [-1, 1) aralığına ölçekleniyor, ama bu ölçek eğitimde
  kullanılan kayıtlarla karşılaştırılmadı.
- **Doğruluk.** Küçük modeller tam v2 modellerinden zayıf; gerçek
  inleme/çığlık yakalama 0.70'ten 0.50'ye düşüyor
  (`reports/esp_model_sweep.json`), simülatörde de kuş sesi fısıltı sanılıyor.
- **İletişim.** Olaylar yalnızca USB seri porttan gidiyor. Sahada LoRa ya da
  ESP-NOW gerekir.
- **Güç.** Pil ve uyku modu yok.
- **Donanım testi yok.** I2S ayarları (kanal seçimi, bit hizalama) gerçek
  mikrofonla doğrulanmadı ve Wokwi bunları simüle etmiyor.
