/*
 * config.h - Donanım ve karar ayarları.
 *
 * Eşikler Python tarafıyla aynı tutulmalı:
 *   RMS_SILENCE_THRESHOLD      pipeline_v2.RMS_SILENCE_THRESHOLD
 *   HUMAN / EMERGENCY eşikleri model başlık dosyalarından gelir (export_c_model.py)
 *   ALARM_CONSECUTIVE, ALARM_COOLDOWN_MS  live_listener.py (3 pencere, 5 sn)
 */
#ifndef CONFIG_H
#define CONFIG_H

/* INMP441 bağlantısı (ESP32-S3 DevKitC-1). L/R pini GND'ye: sol kanal.
   VDD 3.3 V, GND ortak. Pinler serbestçe değiştirilebilir. */
#define I2S_PIN_SCK 4   /* BCLK */
#define I2S_PIN_WS 5    /* LRCLK */
#define I2S_PIN_SD 6    /* veri */

/* İnsan dedektörü 22050 Hz ile eğitildi. Acil durum modeli 16000 Hz bekler;
   yeniden örnekleme öznitelik çıkarımıyla birlikte ESP-4'te. */
#define SAMPLE_RATE 22050
#define WINDOW_SAMPLES SAMPLE_RATE      /* 1 sn pencere (pipeline_v2.WINDOW_SEC) */
#define I2S_READ_SAMPLES 512            /* DMA'dan bir okumada alınan örnek */

/* Kayıt ile çıkarım arasındaki tampon (88 KB). Çıkarım sırasında gelen
   örnekler burada birikir; çıkarım bu süreden uzun sürerse örnekler düşer
   ve "overruns" sayacı artar. */
#define AUDIO_BUFFER_SEC 1

#define RMS_SILENCE_THRESHOLD 0.0015f
#define ALARM_CONSECUTIVE 3
#define ALARM_COOLDOWN_MS 5000

/* Durum satırı (heartbeat) aralığı */
#define STATUS_INTERVAL_MS 1000

#define SERIAL_BAUD 115200

#endif /* CONFIG_H */
