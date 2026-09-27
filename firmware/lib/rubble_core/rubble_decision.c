#include "rubble_decision.h"

static void clear(float rms, rubble_status_t status, rubble_result_t* out) {
    out->status = status;
    out->rms = rms;
    out->human_prob = 0.0f;
    out->state = -1;
    out->state_confidence = 0.0f;
    out->emergency_prob = 0.0f;
    out->is_emergency = 0;
}

int rubble_decide_silence(const rubble_decision_config_t* cfg, float rms, rubble_result_t* out) {
    if (rms < cfg->silence_rms) {
        clear(rms, RUBBLE_SILENCE, out);
        return 1;
    }
    return 0;
}

int rubble_decide_human(const rubble_decision_config_t* cfg, float rms,
                        const float* human_probs, rubble_result_t* out) {
    float p = human_probs[cfg->human_class];
    if (p < cfg->human_threshold) {
        clear(rms, RUBBLE_NO_HUMAN, out);
        out->human_prob = p;
        return 1;
    }
    return 0;
}

void rubble_decide_emergency(const rubble_decision_config_t* cfg, float rms, float human_prob,
                             const float* emergency_probs, rubble_result_t* out) {
    /* np.argmax gibi: eşitlikte ilk sınıf */
    int best = 0;
    for (int c = 1; c < cfg->n_emergency_classes; c++) {
        if (emergency_probs[c] > emergency_probs[best]) best = c;
    }
    float p_normal = cfg->normal_class >= 0 ? emergency_probs[cfg->normal_class] : 0.0f;

    clear(rms, RUBBLE_DETECTED, out);
    out->human_prob = human_prob;
    out->state = best;
    out->state_confidence = emergency_probs[best];
    out->emergency_prob = 1.0f - p_normal;
    out->is_emergency = (best != cfg->normal_class) && (out->emergency_prob >= cfg->emergency_threshold);
}

void rubble_result_unclassified(float rms, rubble_result_t* out) {
    clear(rms, RUBBLE_UNCLASSIFIED, out);
}

const char* rubble_status_name(rubble_status_t status) {
    switch (status) {
        case RUBBLE_SILENCE: return "silence";
        case RUBBLE_NO_HUMAN: return "no_human";
        case RUBBLE_DETECTED: return "DETECTED";
        case RUBBLE_UNCLASSIFIED: return "unclassified";
    }
    return "unknown";
}
