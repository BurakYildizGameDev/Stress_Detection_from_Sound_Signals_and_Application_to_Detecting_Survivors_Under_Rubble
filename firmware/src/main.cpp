/*
 * main.cpp - Enkaz altı ses dinleyicisi (ESP32-S3 + INMP441). PROTOTİP:
 * gerçek donanımda çalıştırılmadı.
 *
 * İki derleme modu:
 *
 *   Mikrofon (varsayılan, env:esp32-s3-devkitc-1)
 *     capture_task (çekirdek 0)   I2S DMA -> float örnekler -> stream buffer
 *     inference_task (çekirdek 1) 1 sn pencere -> rubble_classify -> alarm takibi
 *     Öznitelik çıkarımı henüz yok (ESP-4): konuşma içeren pencereler
 *     "unclassified" olarak raporlanır.
 *
 *   Senaryo (RUBBLE_SIM_SCENARIO, env:wokwi)
 *     Wokwi'de I2S ve mikrofon simüle edilmediği için mikrofon yerine
 *     include/sim_scenario.h'deki pencereler (gerçek bir kayıttan Python ile
 *     çıkarılmış öznitelikler) saniyede bir oynatılır. Modeller, karar ve
 *     alarm takibi cihazdakiyle aynı koddur; her pencerenin sonucu bilgisayar
 *     simülatörünün beklenen sonucuyla karşılaştırılır.
 *
 * Çıktı: seri porttan satır başına bir JSON. Alarmda ALARM_LED_PIN yanar.
 */
#include <Arduino.h>
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>
#include <freertos/task.h>

#include "config.h"
#include "rubble_events.h"
#include "rubble_pipeline.h"

#ifdef RUBBLE_SIM_SCENARIO
#include "sim_scenario.h"
#else
#include <driver/i2s.h>
#include <esp_heap_caps.h>
#include <freertos/stream_buffer.h>
#endif

static SemaphoreHandle_t serial_lock = nullptr;  // satırlar iki görevden yazılıyor
static rubble_tracker_t tracker;
static volatile uint32_t led_off_ms = 0;

// Durum satırı için son pencere bilgisi
static volatile uint32_t windows_processed = 0;
static volatile float last_rms = 0.0f;
static volatile rubble_status_t last_status = RUBBLE_SILENCE;
static volatile uint32_t last_inference_us = 0;
static volatile uint32_t overruns = 0;  // tampon dolduğu için düşen okuma sayısı

static void print_line(const char* line) {
    if (serial_lock) xSemaphoreTake(serial_lock, portMAX_DELAY);
    Serial.println(line);
    if (serial_lock) xSemaphoreGive(serial_lock);
}

#ifndef RUBBLE_SIM_SCENARIO
static void fatal(const char* msg) {
    char buf[160];
    snprintf(buf, sizeof(buf), "{\"type\":\"error\",\"message\":\"%s\"}", msg);
    for (;;) {
        print_line(buf);
        delay(5000);
    }
}
#endif

// Pencere sonucunu alarm takibinden geçirir, olayları yazar, alarmda LED'i yakar.
static void handle_result(const rubble_result_t* r, uint32_t inference_us) {
    windows_processed = windows_processed + 1;
    last_rms = r->rms;
    last_status = r->status;
    last_inference_us = inference_us;

    rubble_event_t events[RUBBLE_MAX_EVENTS];
    uint32_t now = millis();
    int n = rubble_tracker_update(&tracker, r, now, events);
    for (int i = 0; i < n; i++) {
        char buf[320];
        if (rubble_event_to_json(&events[i], emergency_classifier_class_names,
                                 EMERGENCY_CLASSIFIER_NUM_CLASSES, buf, sizeof(buf)) > 0) {
            print_line(buf);
        }
        if (events[i].type == RUBBLE_EVENT_ALARM) led_off_ms = now + ALARM_LED_MS;
    }
}

#ifdef RUBBLE_SIM_SCENARIO
/* ------------------------------------------------------------ senaryo modu */

static int scenario_human(void* ctx, float* out) {
    const sim_window_t* w = (const sim_window_t*)ctx;
    if (!w->has_features) return RUBBLE_FEATURES_NOT_IMPLEMENTED;
    memcpy(out, w->human, sizeof(w->human));
    return RUBBLE_FEATURES_OK;
}

static int scenario_emergency(void* ctx, float* out) {
    const sim_window_t* w = (const sim_window_t*)ctx;
    if (!w->has_features) return RUBBLE_FEATURES_NOT_IMPLEMENTED;
    memcpy(out, w->emergency, sizeof(w->emergency));
    return RUBBLE_FEATURES_OK;
}

static void scenario_task(void*) {
    int mismatches = 0;
    for (int i = 0; i < SIM_SCENARIO_WINDOWS; i++) {
        uint32_t started = millis();
        const sim_window_t* w = &SIM_SCENARIO[i];

        uint32_t t0 = micros();
        rubble_result_t r;
        rubble_classify(w->rms, scenario_human, scenario_emergency, (void*)w, &r);
        uint32_t us = micros() - t0;

        int match = (int)r.status == w->expected_status && r.state == w->expected_state;
        if (!match) mismatches++;
        char buf[200];
        snprintf(buf, sizeof(buf),
                 "{\"type\":\"sim_check\",\"window\":%d,\"status\":\"%s\",\"state\":\"%s\","
                 "\"inference_us\":%lu,\"match\":%s}",
                 i + 1, rubble_status_name(r.status),
                 r.state >= 0 ? emergency_classifier_class_names[r.state] : "-",
                 (unsigned long)us, match ? "true" : "false");
        print_line(buf);
        handle_result(&r, us);

        uint32_t spent = millis() - started;
        if (spent < 1000) vTaskDelay(pdMS_TO_TICKS(1000 - spent));  // gerçek zaman: 1 pencere / sn
    }
    char buf[120];
    snprintf(buf, sizeof(buf), "{\"type\":\"sim_done\",\"windows\":%d,\"mismatches\":%d}",
             SIM_SCENARIO_WINDOWS, mismatches);
    print_line(buf);
    // Wokwi CI bu satırı bekler
    print_line(mismatches == 0 ? "SIM_DONE OK" : "SIM_DONE MISMATCH");
    vTaskDelete(nullptr);
}

static void start_audio() {
    xTaskCreatePinnedToCore(scenario_task, "scenario", 8192, nullptr, 3, nullptr, 1);
}

#else
/* ------------------------------------------------------------ mikrofon modu */

static StreamBufferHandle_t audio_stream = nullptr;
static float* window_buf = nullptr;

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

static int mic_human(void*, float* out) {
    return rubble_human_features(window_buf, WINDOW_SAMPLES, SAMPLE_RATE, out);
}

static int mic_emergency(void*, float* out) {
    return rubble_emergency_features(window_buf, WINDOW_SAMPLES, SAMPLE_RATE, out);
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
        rubble_result_t r;
        rubble_classify(rubble_rms(window_buf, WINDOW_SAMPLES), mic_human, mic_emergency,
                        nullptr, &r);
        handle_result(&r, micros() - t0);
    }
}

static void start_audio() {
    window_buf = (float*)alloc_prefer_psram(WINDOW_SAMPLES * sizeof(float));
    audio_stream = xStreamBufferCreate(AUDIO_BUFFER_SEC * SAMPLE_RATE * sizeof(float),
                                       I2S_READ_SAMPLES * sizeof(float));
    if (!window_buf || !audio_stream) fatal("bellek ayrılamadı");
    if (!i2s_setup()) fatal("I2S başlatılamadı");

    xTaskCreatePinnedToCore(capture_task, "capture", 4096, nullptr, 5, nullptr, 0);
    xTaskCreatePinnedToCore(inference_task, "inference", 8192, nullptr, 3, nullptr, 1);
}
#endif

void setup() {
    Serial.begin(SERIAL_BAUD);
    serial_lock = xSemaphoreCreateMutex();
    pinMode(ALARM_LED_PIN, OUTPUT);
    digitalWrite(ALARM_LED_PIN, LOW);
    delay(500);

#ifdef RUBBLE_SIM_SCENARIO
    const char* mode = "scenario";
    const char* features = "precomputed";
#else
    const char* mode = "microphone";
    const char* features = "not_implemented";
#endif
    char buf[256];
    snprintf(buf, sizeof(buf),
             "{\"type\":\"boot\",\"mode\":\"%s\",\"sample_rate\":%d,\"human_trees\":%d,"
             "\"emergency_trees\":%d,\"human_threshold\":%.2f,\"emergency_threshold\":%.2f,"
             "\"features\":\"%s\"}",
             mode, SAMPLE_RATE, HUMAN_DETECTOR_NUM_TREES, EMERGENCY_CLASSIFIER_NUM_TREES,
             (double)HUMAN_DETECTOR_THRESHOLD, (double)EMERGENCY_CLASSIFIER_THRESHOLD, features);
    print_line(buf);

    rubble_tracker_init(&tracker, ALARM_CONSECUTIVE, ALARM_COOLDOWN_MS,
                        EMERGENCY_CLASSIFIER_NUM_CLASSES);
    start_audio();
}

void loop() {
    static uint32_t last = 0;
    uint32_t now = millis();
    digitalWrite(ALARM_LED_PIN, (int32_t)(led_off_ms - now) > 0 ? HIGH : LOW);
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
