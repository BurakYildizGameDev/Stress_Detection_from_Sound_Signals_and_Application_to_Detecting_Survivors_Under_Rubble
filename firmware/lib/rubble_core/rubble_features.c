#include "rubble_features.h"

#include <math.h>

float rubble_rms(const float* x, int n) {
    if (n <= 0) return 0.0f;
    const int half = RUBBLE_RMS_FRAME / 2;
    /* sıfır dolgulu sinyal uzunluğu n + FRAME; çerçeve sayısı librosa gibi */
    const int n_frames = 1 + n / RUBBLE_RMS_HOP;
    double sum_rms = 0.0;
    for (int t = 0; t < n_frames; t++) {
        int start = t * RUBBLE_RMS_HOP - half;  /* orijinal sinyal koordinatında */
        int lo = start < 0 ? 0 : start;
        int hi = start + RUBBLE_RMS_FRAME;
        if (hi > n) hi = n;
        double acc = 0.0;
        for (int i = lo; i < hi; i++) acc += (double)x[i] * (double)x[i];
        sum_rms += sqrt(acc / (double)RUBBLE_RMS_FRAME);
    }
    return (float)(sum_rms / (double)n_frames);
}

/* ESP-4: features_v2.human_features_v2 karşılığı */
int rubble_human_features(const float* audio, int n, int sample_rate, float* out) {
    (void)audio; (void)n; (void)sample_rate; (void)out;
    return RUBBLE_FEATURES_NOT_IMPLEMENTED;
}

/* ESP-4: features_v2.emergency_features_v2 karşılığı */
int rubble_emergency_features(const float* audio, int n, int sample_rate, float* out) {
    (void)audio; (void)n; (void)sample_rate; (void)out;
    return RUBBLE_FEATURES_NOT_IMPLEMENTED;
}
