/*
 * rubble_events.h - events.AlarmTracker'ın C karşılığı.
 *
 *   - Acil durum penceresi bir bölüm (episode) açar ya da açık bölümü sürdürür.
 *   - Normal konuşma / insan sesi yok penceresi bölümü kapatır.
 *   - Sessizlik bölümü kapatmaz (kişi nefes almak için susmuş olabilir).
 *   - Bölümde `consecutive` tespit birikince tek alarm; iki alarm arası >= cooldown_ms.
 *
 * Cihazda gerçek saat yok; zaman açılıştan beri geçen milisaniyedir (t_ms).
 * Olaylar satır başına bir JSON olarak yazılır; scripts/esp_serial_bridge.py
 * bunları panelin okuduğu logs/events.jsonl biçimine çevirir.
 */
#ifndef RUBBLE_EVENTS_H
#define RUBBLE_EVENTS_H

#include <stddef.h>
#include <stdint.h>

#include "rubble_decision.h"

#ifdef __cplusplus
extern "C" {
#endif

#define RUBBLE_MAX_CLASSES 8
#define RUBBLE_MAX_EVENTS 2  /* tek güncellemede en fazla: detection + alarm */

typedef enum {
    RUBBLE_EVENT_DETECTION = 0,
    RUBBLE_EVENT_ALARM = 1,
    RUBBLE_EVENT_EPISODE_END = 2
} rubble_event_type_t;

typedef struct {
    rubble_event_type_t type;
    uint32_t t_ms;
    uint32_t episode;           /* açılıştan beri bölüm numarası (1, 2, ...) */
    uint32_t episode_start_ms;  /* alarm */
    int window;                 /* detection: bölümdeki pencere sırası; alarm/end: toplam */
    int state;                  /* detection */
    float state_confidence;
    float human_prob;
    float emergency_prob;
    int state_counts[RUBBLE_MAX_CLASSES];  /* alarm */
    float peak_emergency_prob;             /* alarm */
    int alarmed;                           /* episode_end */
} rubble_event_t;

typedef struct {
    int consecutive;
    uint32_t cooldown_ms;
    int n_classes;
    /* iç durum */
    int has_last_alarm;
    uint32_t last_alarm_ms;
    uint32_t n_episodes;
    uint32_t episode;           /* 0 = açık bölüm yok */
    uint32_t episode_start_ms;
    int windows;
    int alarmed;
    float peak;
    int state_counts[RUBBLE_MAX_CLASSES];
} rubble_tracker_t;

void rubble_tracker_init(rubble_tracker_t* t, int consecutive, uint32_t cooldown_ms, int n_classes);

/* Pencere sonucunu işler; üretilen olayları out'a yazar, sayısını döndürür (0..2). */
int rubble_tracker_update(rubble_tracker_t* t, const rubble_result_t* r, uint32_t now_ms,
                          rubble_event_t out[RUBBLE_MAX_EVENTS]);

/* Olayı tek satır JSON'a çevirir (sonunda '\n' yok). Yazılan uzunluğu, yer
   yetmezse -1 döndürür. class_names: acil durum modelinin sınıf adları. */
int rubble_event_to_json(const rubble_event_t* e, const char* const* class_names,
                         int n_classes, char* buf, size_t len);

#ifdef __cplusplus
}
#endif

#endif /* RUBBLE_EVENTS_H */
