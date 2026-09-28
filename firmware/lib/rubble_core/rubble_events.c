#include "rubble_events.h"

#include <stdio.h>
#include <string.h>

static void reset_episode(rubble_tracker_t* t) {
    t->episode = 0;
    t->episode_start_ms = 0;
    t->windows = 0;
    t->alarmed = 0;
    t->peak = 0.0f;
    memset(t->state_counts, 0, sizeof(t->state_counts));
}

void rubble_tracker_init(rubble_tracker_t* t, int consecutive, uint32_t cooldown_ms, int n_classes) {
    memset(t, 0, sizeof(*t));
    t->consecutive = consecutive;
    t->cooldown_ms = cooldown_ms;
    t->n_classes = n_classes < RUBBLE_MAX_CLASSES ? n_classes : RUBBLE_MAX_CLASSES;
    reset_episode(t);
}

int rubble_tracker_update(rubble_tracker_t* t, const rubble_result_t* r, uint32_t now_ms,
                          rubble_event_t out[RUBBLE_MAX_EVENTS]) {
    if (r->status == RUBBLE_SILENCE) return 0;

    if (r->status == RUBBLE_DETECTED && r->is_emergency) {
        if (t->episode == 0) {
            t->episode = ++t->n_episodes;
            t->episode_start_ms = now_ms;
        }
        t->windows++;
        if (r->emergency_prob > t->peak) t->peak = r->emergency_prob;
        if (r->state >= 0 && r->state < t->n_classes) t->state_counts[r->state]++;

        rubble_event_t* d = &out[0];
        memset(d, 0, sizeof(*d));
        d->type = RUBBLE_EVENT_DETECTION;
        d->t_ms = now_ms;
        d->episode = t->episode;
        d->window = t->windows;
        d->state = r->state;
        d->state_confidence = r->state_confidence;
        d->human_prob = r->human_prob;
        d->emergency_prob = r->emergency_prob;

        /* işaretsiz çıkarma: millis() taşması (~49 gün) doğru çalışır */
        int cooled = !t->has_last_alarm || (uint32_t)(now_ms - t->last_alarm_ms) >= t->cooldown_ms;
        if (!t->alarmed && t->windows >= t->consecutive && cooled) {
            t->alarmed = 1;
            t->has_last_alarm = 1;
            t->last_alarm_ms = now_ms;
            rubble_event_t* a = &out[1];
            memset(a, 0, sizeof(*a));
            a->type = RUBBLE_EVENT_ALARM;
            a->t_ms = now_ms;
            a->episode = t->episode;
            a->episode_start_ms = t->episode_start_ms;
            a->window = t->windows;
            memcpy(a->state_counts, t->state_counts, sizeof(a->state_counts));
            a->peak_emergency_prob = t->peak;
            return 2;
        }
        return 1;
    }

    if (t->episode == 0) return 0;
    rubble_event_t* e = &out[0];
    memset(e, 0, sizeof(*e));
    e->type = RUBBLE_EVENT_EPISODE_END;
    e->t_ms = now_ms;
    e->episode = t->episode;
    e->window = t->windows;
    e->alarmed = t->alarmed;
    reset_episode(t);
    return 1;
}

/* snprintf sarmalayıcı: taşarsa -1 */
#define APPEND(...)                                                   \
    do {                                                              \
        int n_ = snprintf(buf + used, len - used, __VA_ARGS__);       \
        if (n_ < 0 || (size_t)n_ >= len - used) return -1;            \
        used += (size_t)n_;                                           \
    } while (0)

int rubble_event_to_json(const rubble_event_t* e, const char* const* class_names,
                         int n_classes, char* buf, size_t len) {
    size_t used = 0;
    if (len == 0) return -1;
    switch (e->type) {
        case RUBBLE_EVENT_DETECTION:
            APPEND("{\"type\":\"detection\",\"t_ms\":%lu,\"episode\":%lu,\"window\":%d,"
                   "\"state\":\"%s\",\"state_confidence\":%.4f,\"human_prob\":%.4f,"
                   "\"emergency_prob\":%.4f}",
                   (unsigned long)e->t_ms, (unsigned long)e->episode, e->window,
                   (e->state >= 0 && e->state < n_classes) ? class_names[e->state] : "?",
                   (double)e->state_confidence, (double)e->human_prob, (double)e->emergency_prob);
            break;
        case RUBBLE_EVENT_ALARM: {
            APPEND("{\"type\":\"alarm\",\"t_ms\":%lu,\"episode\":%lu,\"episode_start_ms\":%lu,"
                   "\"windows\":%d,\"states\":{",
                   (unsigned long)e->t_ms, (unsigned long)e->episode,
                   (unsigned long)e->episode_start_ms, e->window);
            int first = 1;
            for (int c = 0; c < n_classes && c < RUBBLE_MAX_CLASSES; c++) {
                if (e->state_counts[c] == 0) continue;
                APPEND("%s\"%s\":%d", first ? "" : ",", class_names[c], e->state_counts[c]);
                first = 0;
            }
            APPEND("},\"peak_emergency_prob\":%.4f}", (double)e->peak_emergency_prob);
            break;
        }
        case RUBBLE_EVENT_EPISODE_END:
            APPEND("{\"type\":\"episode_end\",\"t_ms\":%lu,\"episode\":%lu,\"windows\":%d,"
                   "\"alarmed\":%s}",
                   (unsigned long)e->t_ms, (unsigned long)e->episode, e->window,
                   e->alarmed ? "true" : "false");
            break;
        default:
            return -1;
    }
    return (int)used;
}
