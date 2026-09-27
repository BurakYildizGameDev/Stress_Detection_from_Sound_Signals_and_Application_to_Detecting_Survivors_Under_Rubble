/*
 * sim_main.c - Firmware'in karar hattını bilgisayarda çalıştırır.
 * scripts/esp_simulate.py derler ve besler; elle çalıştırmak gerekmez.
 *
 * Cihazla aynı kod: rubble_pipeline.h (modeller + karar) ve rubble_events
 * (alarm takibi). Tek fark öznitelikler: ESP-4'e kadar Python (librosa)
 * hesaplar ve stdin'den verir.
 *
 * Girdi, pencere başına bir satır:
 *   W <audio_samples> <x1> ... <xn> <has_features> <h1..h33> <e1..e19>
 * RMS C tarafında (rubble_rms) cihazdaki gibi ham örneklerden hesaplanır.
 *
 * Çıktı, satır başına JSON: her pencere için "window", ardından varsa
 * detection / alarm / episode_end olayları (firmware ile aynı biçim).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rubble_events.h"
#include "rubble_pipeline.h"

#define WINDOW_MS 1000
#define ALARM_CONSECUTIVE 3
#define ALARM_COOLDOWN_MS 5000

typedef struct {
    int has_features;
    float human[HUMAN_DETECTOR_NUM_FEATURES];
    float emergency[EMERGENCY_CLASSIFIER_NUM_FEATURES];
} window_features_t;

static int sim_human(void* ctx, float* out) {
    const window_features_t* w = (const window_features_t*)ctx;
    if (!w->has_features) return RUBBLE_FEATURES_NOT_IMPLEMENTED;
    memcpy(out, w->human, sizeof(w->human));
    return RUBBLE_FEATURES_OK;
}

static int sim_emergency(void* ctx, float* out) {
    const window_features_t* w = (const window_features_t*)ctx;
    if (!w->has_features) return RUBBLE_FEATURES_NOT_IMPLEMENTED;
    memcpy(out, w->emergency, sizeof(w->emergency));
    return RUBBLE_FEATURES_OK;
}

static int read_floats(float* out, int n) {
    for (int i = 0; i < n; i++) {
        if (scanf("%f", &out[i]) != 1) return 0;
    }
    return 1;
}

int main(void) {
    rubble_tracker_t tracker;
    rubble_tracker_init(&tracker, ALARM_CONSECUTIVE, ALARM_COOLDOWN_MS,
                        EMERGENCY_CLASSIFIER_NUM_CLASSES);
    float* audio = NULL;
    int audio_cap = 0;
    window_features_t w;
    unsigned long index = 0;
    char cmd[4];

    while (scanf("%3s", cmd) == 1) {
        if (strcmp(cmd, "W") != 0) {
            fprintf(stderr, "bilinmeyen komut: %s\n", cmd);
            return 2;
        }
        int n;
        if (scanf("%d", &n) != 1 || n <= 0) return 2;
        if (n > audio_cap) {
            free(audio);
            audio = (float*)malloc((size_t)n * sizeof(float));
            if (!audio) return 3;
            audio_cap = n;
        }
        if (!read_floats(audio, n)) return 2;
        if (scanf("%d", &w.has_features) != 1) return 2;
        if (!read_floats(w.human, HUMAN_DETECTOR_NUM_FEATURES)) return 2;
        if (!read_floats(w.emergency, EMERGENCY_CLASSIFIER_NUM_FEATURES)) return 2;

        rubble_result_t r;
        rubble_classify(rubble_rms(audio, n), sim_human, sim_emergency, &w, &r);
        uint32_t t_ms = (uint32_t)(index * WINDOW_MS);
        index++;

        printf("{\"type\":\"window\",\"t_ms\":%lu,\"window\":%lu,\"status\":\"%s\",\"state\":\"%s\","
               "\"rms\":%.9g,\"human_prob\":%.9g,\"state_confidence\":%.9g,"
               "\"emergency_prob\":%.9g,\"is_emergency\":%s}\n",
               (unsigned long)t_ms, index, rubble_status_name(r.status),
               r.state >= 0 ? emergency_classifier_class_names[r.state] : "-",
               (double)r.rms, (double)r.human_prob, (double)r.state_confidence,
               (double)r.emergency_prob, r.is_emergency ? "true" : "false");

        rubble_event_t events[RUBBLE_MAX_EVENTS];
        int k = rubble_tracker_update(&tracker, &r, t_ms, events);
        for (int i = 0; i < k; i++) {
            char buf[320];
            if (rubble_event_to_json(&events[i], emergency_classifier_class_names,
                                     EMERGENCY_CLASSIFIER_NUM_CLASSES, buf, sizeof(buf)) > 0) {
                printf("%s\n", buf);
            }
        }
    }
    free(audio);
    return 0;
}
