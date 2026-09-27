/*
 * rubble_decision.h - İki aşamalı karar (pipeline_v2.PipelineV2.analyze_audio_array ile aynı).
 *
 *   1. RMS sessizlik kapısı
 *   2. İnsan sesi olasılığı >= human_threshold değilse "no_human"
 *   3. Acil durum sınıfı = argmax; emergency_prob = 1 - p(normal);
 *      is_emergency = sınıf != normal ve emergency_prob >= emergency_threshold
 *
 * Model çağrıları burada değil: çağıran taraf modeli çalıştırıp olasılıkları verir.
 * Böylece bu dosya donanımsız derlenip Python ile karşılaştırılarak test edilir.
 */
#ifndef RUBBLE_DECISION_H
#define RUBBLE_DECISION_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    RUBBLE_SILENCE = 0,
    RUBBLE_NO_HUMAN = 1,
    RUBBLE_DETECTED = 2,
    RUBBLE_UNCLASSIFIED = 3  /* öznitelik ya da model yok (pipeline_v2: RAW_AUDIO) */
} rubble_status_t;

typedef struct {
    float silence_rms;          /* pipeline_v2.RMS_SILENCE_THRESHOLD */
    float human_threshold;      /* models/human_detector_esp.json: human_threshold */
    float emergency_threshold;  /* pipeline_v2.EMERGENCY_PROB_THRESHOLD */
    int human_class;            /* insan dedektöründe "human" sınıfının indeksi */
    int normal_class;           /* acil durum modelinde "normal" sınıfının indeksi */
    int n_emergency_classes;
} rubble_decision_config_t;

typedef struct {
    rubble_status_t status;
    float rms;
    float human_prob;           /* SILENCE'ta 0 */
    int state;                  /* acil durum sınıf indeksi, DETECTED değilse -1 */
    float state_confidence;
    float emergency_prob;
    int is_emergency;
} rubble_result_t;

/* 1. aşama: sessizse 1 döner ve out'u doldurur. */
int rubble_decide_silence(const rubble_decision_config_t* cfg, float rms, rubble_result_t* out);

/* 2. aşama: insan değilse 1 döner ve out'u doldurur (acil durum modeli çalıştırılmaz). */
int rubble_decide_human(const rubble_decision_config_t* cfg, float rms,
                        const float* human_probs, rubble_result_t* out);

/* 3. aşama: acil durum olasılıklarından son karar. */
void rubble_decide_emergency(const rubble_decision_config_t* cfg, float rms, float human_prob,
                             const float* emergency_probs, rubble_result_t* out);

void rubble_result_unclassified(float rms, rubble_result_t* out);

const char* rubble_status_name(rubble_status_t status);

#ifdef __cplusplus
}
#endif

#endif /* RUBBLE_DECISION_H */
