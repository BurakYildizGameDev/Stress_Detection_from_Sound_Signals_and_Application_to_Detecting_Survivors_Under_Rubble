/*
 * rubble_features.h - Cihaz üzerinde öznitelik çıkarımı arayüzü.
 *
 * DURUM: ESP-4'te yazılacak. Şimdilik öznitelik fonksiyonları
 * RUBBLE_FEATURES_NOT_IMPLEMENTED döndürür ve firmware pencereleri
 * "unclassified" olarak raporlar.
 *
 * Modeller features_v2.py (librosa) ile eğitildi. C tarafı aynı sayıları
 * üretmeli: MFCC (n_fft 2048, hop 512, Hann, center=True + reflect padding,
 * 128 Slaney mel, power_to_db top_db=80, ortho DCT-II), spectral flatness,
 * rolloff, flux, entropy, HNR (yalnızca 44-441 gecikmeleri), 0-500 Hz oranı;
 * sinyal önce RMS 0.05'e ölçeklenir. İnsan modeli 22050 Hz, acil durum modeli
 * 16000 Hz bekler.
 */
#ifndef RUBBLE_FEATURES_H
#define RUBBLE_FEATURES_H

#ifdef __cplusplus
extern "C" {
#endif

#define RUBBLE_FEATURES_OK 0
#define RUBBLE_FEATURES_NOT_IMPLEMENTED (-1)

#define RUBBLE_HUMAN_N_FEATURES 33
#define RUBBLE_EMERGENCY_N_FEATURES 19

/* Pencerenin tamamı üzerinden RMS. Not: pipeline_v2 librosa.feature.rms
   karelerinin ortalamasını kullanır; sessiz/sesli ayrımı için fark küçüktür,
   birebir eşleşme ESP-4'te. */
float rubble_rms(const float* x, int n);

/* out: RUBBLE_HUMAN_N_FEATURES float (features_v2.human_features_v2) */
int rubble_human_features(const float* audio, int n, int sample_rate, float* out);

/* out: RUBBLE_EMERGENCY_N_FEATURES float (features_v2.emergency_features_v2) */
int rubble_emergency_features(const float* audio, int n, int sample_rate, float* out);

#ifdef __cplusplus
}
#endif

#endif /* RUBBLE_FEATURES_H */
