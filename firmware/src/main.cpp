/*
 * main.cpp - Enkaz altı ses dinleyicisi (ESP32-S3 + INMP441). PROTOTİP:
 * donanımda test edilmedi.
 *
 * Görevler:
 *   capture_task (çekirdek 0)  I2S DMA -> float örnekler -> stream buffer
 *   inference_task (çekirdek 1) 1 sn pencere -> RMS kapısı -> insan dedektörü
 *                               -> acil durum sınıflandırıcısı -> alarm takibi
 *
 * Çıktı: seri porttan satır başına bir JSON (olaylar ve saniyede bir durum).
 * scripts/esp_serial_bridge.py bunları panelin (app.py) okuduğu dosyalara yazar.
 *
 * Öznitelik çıkarımı henüz yok (ESP-4): konuşma içeren pencereler
 * "unclassified" olarak raporlanır, modeller çağrılmaz.
 */
#include <Arduino.h>
#include <driver/i2s.h>
#include <esp_heap_caps.h>
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>
#include <freertos/stream_buffer.h>
#include <freertos/task.h>

#include "config.h"
#include "emergency_classifier_model.h"
#include "human_detector_model.h"
#include "rubble_decision.h"
#include "rubble_events.h"
#include "rubble_features.h"

#ifndef EMERGENCY_CLASSIFIER_THRESHOLD
#error "emergency_classifier_model.h eşiksiz üretilmiş; --threshold 0.45 ile yeniden üretin"
#endif
#ifndef HUMAN_DETECTOR_THRESHOLD
#error "human_detector_model.h eşiksiz üretilmiş; models/human_detector_esp.json kullanın"
#endif

static StreamBufferHandle_t audio_stream = nullptr;
static SemaphoreHandle_t serial_lock = nullptr;  // satırlar iki görevden yazılıyor
static volatile uint32_t overruns = 0;  // tampon dolduğu için düşen okuma sayısı

static float* window_buf = nullptr;
static float human_features[HUMAN_DETECTOR_NUM_FEATURES];
static float emergency_features[EMERGENCY_CLASSIFIER_NUM_FEATURES];

static const rubble_decision_config_t decision_cfg = {
    RMS_SILENCE_THRESHOLD,
    HUMAN_DETECTOR_THRESHOLD,
    EMERGENCY_CLASSIFIER_THRESHOLD,
    HUMAN_DETECTOR_CLASS_HUMAN,
    EMERGENCY_CLASSIFIER_CLASS_NORMAL,
    EMERGENCY_CLASSIFIER_NUM_CLASSES,
};

static rubble_tracker_t tracker;

// Durum satırı için son pencere bilgisi
static uint32_t windows_processed = 0;
static float last_rms = 0.0f;
static rubble_status_t last_status = RUBBLE_SILENCE;
static uint32_t last_inference_us = 0;

static void print_line(const char* line) {
    if (serial_lock) xSemaphoreTake(serial_lock, portMAX_DELAY);
    Serial.println(line);
    if (serial_lock) xSemaphoreGive(serial_lock);
}

static void fatal(const char* msg) {
    char buf[160];
    snprintf(buf, sizeof(buf), "{\"type\":\"error\",\"message\":\"%s\"}", msg);
    for (;;) {
        print_line(buf);
        delay(5000);
    }
}

static void* alloc_prefer_psram(size_t bytes) {
    void* p = heap_caps_malloc(bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    return p ? p : heap_caps_malloc(bytes, MALLOC_CAP_8BIT);
}

static bool i2s_setup() {
    // Alanlar tek tek atanıyor: ESP-IDF sürümleri arasında yapı sırası değişiyor.
    i2s_config_t cfg = {};
    cfg.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX);
    cfg.sample_rate = SAMPLE_RATE;
    cfg.bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT;  // INMP441: 32 bit çerçevede 24 bit veri
    cfg.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;   // L/R = GND
    cfg.communication_format = I2S_COMM_FORMAT_STAND_I2S;
    cfg.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
    cfg.dma_buf_count = 8;
    cfg.dma_buf_len = I2S_READ_SAMPLES;
    cfg.use_apll = true;  // 22050 Hz için daha doğru saat

    i2s_pin_config_t pins = {};
    pins.mck_io_num = I2S_PIN_NO_CHANGE;
    pins.bck_io_num = I2S_PIN_SCK;
    pins.ws_io_num = I2S_PIN_WS;
    pins.data_out_num = I2S_PIN_NO_CHANGE;
    pins.data_in_num = I2S_PIN_SD;

    if (i2s_driver_install(I2S_NUM_0, &cfg, 0, nullptr) != ESP_OK) return false;
    if (i2s_set_pin(I2S_NUM_0, &pins) != ESP_OK) return false;
    return true;
}

static void capture_task(void*) {
    static int32_t raw[I2S_READ_SAMPLES];
    static float samples[I2S_READ_SAMPLES];
    for (;;) {
        size_t bytes_read = 0;
        i2s_read(I2S_NUM_0, raw, sizeof(raw), &bytes_read, portMAX_DELAY);
        size_t n = bytes_read / sizeof(int32_t);
        for (size_t i = 0; i < n; i++) {
            // 24 bit işaretli veri üst bitlerde; [-1, 1) aralığına ölçekle.
            // Mutlak seviye RMS_SILENCE_THRESHOLD'u etkiler: kalibrasyon gerekir.
            samples[i] = (float)(raw[i] >> 8) / 8388608.0f;
        }
        size_t want = n * sizeof(float);
        if (xStreamBufferSend(audio_stream, samples, want, 0) != want) {
            overruns = overruns + 1;
        }
    }
}

static void emit(const rubble_event_t* ev) {
    char buf[320];
    if (rubble_event_to_json(ev, emergency_classifier_class_names,
                             EMERGENCY_CLASSIFIER_NUM_CLASSES, buf, sizeof(buf)) > 0) {
        print_line(buf);
    }
}

static void classify_window(rubble_result_t* r) {
    float rms = rubble_rms(window_buf, WINDOW_SAMPLES);
    if (rubble_decide_silence(&decision_cfg, rms, r)) return;

    if (rubble_human_features(window_buf, WINDOW_SAMPLES, SAMPLE_RATE, human_features)
        != RUBBLE_FEATURES_OK) {
        rubble_result_unclassified(rms, r);  // ESP-4'e kadar
        return;
    }
    float human_probs[HUMAN_DETECTOR_NUM_CLASSES];
    human_detector_predict(human_features, human_probs);
    if (rubble_decide_human(&decision_cfg, rms, human_probs, r)) return;

    if (rubble_emergency_features(window_buf, WINDOW_SAMPLES, SAMPLE_RATE, emergency_features)
        != RUBBLE_FEATURES_OK) {
        rubble_result_unclassified(rms, r);
        return;
    }
    float emergency_probs[EMERGENCY_CLASSIFIER_NUM_CLASSES];
    emergency_classifier_predict(emergency_features, emergency_probs);
    rubble_decide_emergency(&decision_cfg, rms, human_probs[HUMAN_DETECTOR_CLASS_HUMAN],
                            emergency_probs, r);
}

static void inference_task(void*) {
    for (;;) {
        // Tam bir pencere dolana kadar bekle
        size_t got = 0;
        const size_t need = WINDOW_SAMPLES * sizeof(float);
        while (got < need) {
            got += xStreamBufferReceive(audio_stream, (uint8_t*)window_buf + got, need - got,
                                        portMAX_DELAY);
        }

        uint32_t t0 = micros();
        rubble_result_t result;
        classify_window(&result);
        last_inference_us = micros() - t0;

        windows_processed++;
        last_rms = result.rms;
        last_status = result.status;

        rubble_event_t events[RUBBLE_MAX_EVENTS];
        int n = rubble_tracker_update(&tracker, &result, millis(), events);
        for (int i = 0; i < n; i++) emit(&events[i]);
    }
}

void setup() {
    Serial.begin(SERIAL_BAUD);
    serial_lock = xSemaphoreCreateMutex();
    delay(500);

    char buf[256];
    snprintf(buf, sizeof(buf),
             "{\"type\":\"boot\",\"sample_rate\":%d,\"human_trees\":%d,\"emergency_trees\":%d,"
             "\"human_threshold\":%.2f,\"emergency_threshold\":%.2f,\"features\":\"%s\"}",
             SAMPLE_RATE, HUMAN_DETECTOR_NUM_TREES, EMERGENCY_CLASSIFIER_NUM_TREES,
             (double)HUMAN_DETECTOR_THRESHOLD, (double)EMERGENCY_CLASSIFIER_THRESHOLD,
             "not_implemented");
    print_line(buf);

    rubble_tracker_init(&tracker, ALARM_CONSECUTIVE, ALARM_COOLDOWN_MS,
                        EMERGENCY_CLASSIFIER_NUM_CLASSES);

    window_buf = (float*)alloc_prefer_psram(WINDOW_SAMPLES * sizeof(float));
    audio_stream = xStreamBufferCreate(AUDIO_BUFFER_SEC * SAMPLE_RATE * sizeof(float),
                                       I2S_READ_SAMPLES * sizeof(float));
    if (!window_buf || !audio_stream) fatal("bellek ayrılamadı");
    if (!i2s_setup()) fatal("I2S başlatılamadı");

    xTaskCreatePinnedToCore(capture_task, "capture", 4096, nullptr, 5, nullptr, 0);
    xTaskCreatePinnedToCore(inference_task, "inference", 8192, nullptr, 3, nullptr, 1);
}

void loop() {
    static uint32_t last = 0;
    uint32_t now = millis();
    if (now - last >= STATUS_INTERVAL_MS) {
        last = now;
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "{\"type\":\"status\",\"t_ms\":%lu,\"windows\":%lu,\"last_rms\":%.5f,"
                 "\"last_status\":\"%s\",\"inference_ms\":%.1f,\"overruns\":%lu,"
                 "\"open_episode\":%lu,\"free_heap\":%lu}",
                 (unsigned long)now, (unsigned long)windows_processed, (double)last_rms,
                 rubble_status_name(last_status), last_inference_us / 1000.0,
                 (unsigned long)overruns, (unsigned long)tracker.episode,
                 (unsigned long)ESP.getFreeHeap());
        print_line(buf);
    }
    delay(20);
}
