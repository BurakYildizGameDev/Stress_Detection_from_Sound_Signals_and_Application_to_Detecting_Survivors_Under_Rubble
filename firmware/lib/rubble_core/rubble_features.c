#include "rubble_features.h"

#include <math.h>

float rubble_rms(const float* x, int n) {
    if (n <= 0) return 0.0f;
    double acc = 0.0;
    for (int i = 0; i < n; i++) acc += (double)x[i] * (double)x[i];
    return (float)sqrt(acc / (double)n);
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
