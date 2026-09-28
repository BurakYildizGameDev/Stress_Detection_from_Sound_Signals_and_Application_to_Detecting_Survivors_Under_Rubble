/*
 * rubble_pipeline.h - Bir pencerenin sınıflandırılması (pipeline_v2 ile aynı sıra).
 *
 * Firmware (src/main.cpp), bilgisayar simülatörü (sim/sim_main.c) ve Wokwi
 * senaryo modu aynı fonksiyonu çağırır; yalnızca özniteliklerin nereden
 * geldiği değişir:
 *   firmware   rubble_human_features / rubble_emergency_features (mikrofon, ESP-4)
 *   simülatör  Python'da (librosa) hesaplanıp stdin'den gelir
 *   Wokwi      firmware'e gömülü senaryo penceresinden okunur
 *
 * Acil durum öznitelikleri yalnızca insan sesi varsa istenir (pipeline_v2 gibi).
 */
#ifndef RUBBLE_PIPELINE_H
#define RUBBLE_PIPELINE_H

#include "emergency_classifier_model.h"
#include "human_detector_model.h"
#include "rubble_decision.h"
#include "rubble_features.h"

#ifndef EMERGENCY_CLASSIFIER_THRESHOLD
#error "emergency_classifier_model.h eşiksiz üretilmiş; models/emergency_classifier_esp.json kullanın"
#endif
#ifndef HUMAN_DETECTOR_THRESHOLD
#error "human_detector_model.h eşiksiz üretilmiş; models/human_detector_esp.json kullanın"
#endif

#ifndef RMS_SILENCE_THRESHOLD
#define RMS_SILENCE_THRESHOLD 0.0015f  /* pipeline_v2.RMS_SILENCE_THRESHOLD */
#endif

/* Öznitelik kaynağı: out'u doldurur, RUBBLE_FEATURES_OK ya da hata döndürür. */
typedef int (*rubble_feature_fn)(void* ctx, float* out);

static const rubble_decision_config_t RUBBLE_DECISION_CONFIG = {
    RMS_SILENCE_THRESHOLD,
    HUMAN_DETECTOR_THRESHOLD,
    EMERGENCY_CLASSIFIER_THRESHOLD,
    HUMAN_DETECTOR_CLASS_HUMAN,
    EMERGENCY_CLASSIFIER_CLASS_NORMAL,
    EMERGENCY_CLASSIFIER_NUM_CLASSES,
};

static void rubble_classify(float rms, rubble_feature_fn human_fn, rubble_feature_fn emergency_fn,
                            void* ctx, rubble_result_t* r) {
    const rubble_decision_config_t* cfg = &RUBBLE_DECISION_CONFIG;
    if (rubble_decide_silence(cfg, rms, r)) return;

    float human_features[HUMAN_DETECTOR_NUM_FEATURES];
    if (human_fn(ctx, human_features) != RUBBLE_FEATURES_OK) {
        rubble_result_unclassified(rms, r);
        return;
    }
    float human_probs[HUMAN_DETECTOR_NUM_CLASSES];
    human_detector_predict(human_features, human_probs);
    if (rubble_decide_human(cfg, rms, human_probs, r)) return;

    float emergency_features[EMERGENCY_CLASSIFIER_NUM_FEATURES];
    if (emergency_fn(ctx, emergency_features) != RUBBLE_FEATURES_OK) {
        rubble_result_unclassified(rms, r);
        return;
    }
    float emergency_probs[EMERGENCY_CLASSIFIER_NUM_CLASSES];
    emergency_classifier_predict(emergency_features, emergency_probs);
    rubble_decide_emergency(cfg, rms, human_probs[HUMAN_DETECTOR_CLASS_HUMAN], emergency_probs, r);
}

#endif /* RUBBLE_PIPELINE_H */
