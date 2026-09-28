"""
features_v2.py - İnleme, Fısıltı ve Zayıf Vokal İmzaları Destekleyen Genişletilmiş Öznitelik Çıkarıcı.

Bu modül Aşama 1 için 33 boyutlu ve Aşama 2 için 19 boyutlu öznitelik vektörleri üretir.

Ses şiddetinden bağımsızlık: sinyal öznitelik çıkarımından önce sabit RMS'e
ölçeklenir ve RMS vektöre konmaz. Aksi hâlde model "sessizse fısıltıdır" gibi
bir kısayol öğrenir; enkaz arkasında her ses zaten sessizdir. Ham RMS yalnızca
canlı hattaki sessizlik kapısında (pipeline_v2) kullanılır.

V2 İyileştirmeleri:
1. Harmonik-Gürültü Oranı (HNR): İnleme seslerindeki zayıf glottal titreşimi yakalar.
2. Spektral Düzlük (Spectral Flatness): Fısıltıdaki aperiyodik türbülansı doğrular.
3. 0-500 Hz Alt-Bant Enerji Oranı (Sub-Band Energy Ratio): Boğazdan gelen derin inlemeleri izole eder.
4. Spektral Akı (Spectral Flux) ve Entropi (Spectral Entropy): Fısıltı ve çevre gürültüsünü ayrıştırır.
"""
import numpy as np
import librosa

from features import HUMAN_SR, EMERGENCY_SR, N_MFCC, _resample

HUMAN_N_FEATURES_V2 = 33
EMERGENCY_N_FEATURES_V2 = 19
TARGET_RMS = 0.05

FEATURE_SPECS_V2 = {
    "human": {
        "version": 3,
        "sr": HUMAN_SR,
        "n_features": HUMAN_N_FEATURES_V2,
        "clip_sec": 2.0,
        "loudness_normalized": True,
        "desc": "RMS-normalized: 13 MFCC mean + 13 MFCC std + ZCR + Flatness(mean/std) + Rolloff + Flux + HNR + SubBandRatio"
    },
    "emergency": {
        "version": 3,
        "sr": EMERGENCY_SR,
        "n_features": EMERGENCY_N_FEATURES_V2,
        "clip_sec": None,
        "loudness_normalized": True,
        "desc": "RMS-normalized: 13 MFCC mean + Centroid + HNR + Flatness + SubBandRatio + Entropy + Flux"
    }
}


def extract_hnr_robust(y, sr):
    """
    Zaman alanında otokorelasyon ile gürültüye dayanıklı Harmonik-Gürültü Oranı (HNR) çıkarır.
    İnleme (moaning) durumlarında vokal kordların zayıf periyodikliğini yakalar.
    Dönüş: dB cinsinden [-20.0, 40.0] aralığında kırpılmış değer.
    """
    if len(y) == 0 or np.all(y == 0):
        return 0.0

    # Otokorelasyon hesabı
    corr = np.correlate(y, y, mode="full")[len(y) - 1:]
    if len(corr) == 0 or corr[0] <= 1e-12:
        return 0.0

    min_lag = max(1, int(sr / 500))  # 500 Hz üst sınır (F0 tepe)
    max_lag = min(len(corr) - 1, int(sr / 50))   # 50 Hz alt sınır (F0 taban)

    if min_lag >= max_lag:
        return 0.0

    peak_slice = corr[min_lag:max_lag]
    if len(peak_slice) == 0:
        return 0.0

    peak_lag = min_lag + int(np.argmax(peak_slice))
    r_peak = corr[peak_lag]
    r_zero = corr[0]

    if r_zero <= r_peak or r_peak <= 0:
        return 0.0

    ratio = r_peak / max(r_zero - r_peak, 1e-10)
    hnr_db = 10.0 * np.log10(ratio)
    return float(np.clip(hnr_db, -20.0, 40.0))


def normalize_loudness(y, target_rms=TARGET_RMS):
    """Sinyali sabit RMS'e ölçekler (tamamen sessiz sinyal olduğu gibi kalır)."""
    rms = float(np.sqrt(np.mean(y ** 2))) if len(y) else 0.0
    if rms < 1e-8:
        return y
    return (y * (target_rms / rms)).astype(np.float32)


def human_features_v2(y, sr=HUMAN_SR):
    """
    İnleme ve fısıltı tespitine duyarlı 33 boyutlu insan varlık öznitelik vektörü
    (sinyal önce TARGET_RMS'e ölçeklenir):
    - 0..12:   13 MFCC ortalama
    - 13..25:  13 MFCC standart sapma
    - 26:      1 ZCR ortalama
    - 27:      1 Spektral Düzlük (Spectral Flatness) ortalama [Fısıltı göstergesi]
    - 28:      1 Spektral Düzlük standart sapma
    - 29:      1 Spektral Düşüş (Spectral Rolloff %85) / sr
    - 30:      1 Spektral Akı (Spectral Flux) ortalama
    - 31:      1 HNR / 40.0 (Normalize Harmonik Oran) [İnleme göstergesi]
    - 32:      1 0-500 Hz Alt-Bant Enerji Oranı (Sub-Band Energy Ratio) [İnleme göstergesi]
    """
    y = normalize_loudness(_resample(y, sr, HUMAN_SR))

    # 1. Klasik 28 Öznitelik
    mfcc = librosa.feature.mfcc(y=y, sr=HUMAN_SR, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    # 2. Spektral Düzlük (Fısıltıda 0.40 - 0.70 aralığına fırlar)
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    flatness_mean = float(np.mean(flatness)) if len(flatness) > 0 else 0.0
    flatness_std = float(np.std(flatness)) if len(flatness) > 0 else 0.0

    # 3. Spektral Düşüş Frekansı (Rolloff)
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=HUMAN_SR, roll_percent=0.85))) / float(HUMAN_SR)

    # 4. Spektral STFT, Akı (Flux) ve 0-500 Hz Alt-Bant Oranı
    stft = np.abs(librosa.stft(y))
    if stft.shape[1] > 1:
        flux = float(np.mean(np.sqrt(np.sum(np.diff(stft, axis=1)**2, axis=0))))
    else:
        flux = 0.0

    # 5. HNR (Harmonik-Gürültü Oranı)
    hnr_val = extract_hnr_robust(y, HUMAN_SR) / 40.0

    # 6. Alt Bant Enerji Oranı (0-500 Hz) - İnlemede %70-85'e ulaşır
    freqs = librosa.fft_frequencies(sr=HUMAN_SR, n_fft=stft.shape[0] * 2 - 2 if stft.shape[0] > 1 else 2048)
    # Eşleşen frekans indeksi
    n_freq_bins = stft.shape[0]
    freqs = np.linspace(0, HUMAN_SR / 2.0, n_freq_bins)
    low_idx = np.where(freqs <= 500.0)[0]

    tot_energy = float(np.sum(stft**2)) + 1e-10
    low_energy = float(np.sum(stft[low_idx, :]**2)) if len(low_idx) > 0 else 0.0
    low_ratio = low_energy / tot_energy

    feats = np.hstack([
        mfcc_mean,
        mfcc_std,
        np.float32(zcr),
        np.float32(flatness_mean),
        np.float32(flatness_std),
        np.float32(rolloff),
        np.float32(flux),
        np.float32(hnr_val),
        np.float32(low_ratio),
    ]).astype(np.float32)

    # NaN / Inf koruması
    return np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=-1.0)


def emergency_features_v2(y, sr=EMERGENCY_SR):
    """
    Acil durum ve travma sınıflandırması için 19 boyutlu vektör
    (sinyal önce TARGET_RMS'e ölçeklenir):
    - 0..12:   13 MFCC ortalama
    - 13:      1 Spectral Centroid / (sr / 2) [Normalize Kütle Merkezi]
    - 14:      1 HNR / 40.0 [İnleme ayrımı]
    - 15:      1 Spectral Flatness [Fısıltı ayrımı]
    - 16:      1 0-500 Hz Alt-Bant Enerji Oranı [İnleme ayrımı]
    - 17:      1 Spectral Entropy [Kaos ve gürültü ayrımı]
    - 18:      1 Spectral Flux [Zaman-frekans akısı]
    """
    y = normalize_loudness(_resample(y, sr, EMERGENCY_SR))

    # 1. 13 MFCC
    mfcc_mean = np.mean(librosa.feature.mfcc(y=y, sr=EMERGENCY_SR, n_mfcc=N_MFCC), axis=1)

    # 2. Spektral Ağırlık Merkezi (Centroid)
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=EMERGENCY_SR))) / (EMERGENCY_SR / 2.0)

    # 3. HNR
    hnr_val = extract_hnr_robust(y, EMERGENCY_SR) / 40.0

    # 4. Spektral Düzlük
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))

    # 5. STFT, Alt Bant ve Entropi
    stft = np.abs(librosa.stft(y))
    n_freq_bins = stft.shape[0]
    freqs = np.linspace(0, EMERGENCY_SR / 2.0, n_freq_bins)
    low_idx = np.where(freqs <= 500.0)[0]

    tot_energy = float(np.sum(stft**2)) + 1e-10
    low_energy = float(np.sum(stft[low_idx, :]**2)) if len(low_idx) > 0 else 0.0
    low_ratio = low_energy / tot_energy

    # Spektral Entropi
    col_sums = np.sum(stft, axis=0, keepdims=True) + 1e-10
    prob_dist = stft / col_sums
    entropy_cols = -np.sum(prob_dist * np.log2(prob_dist + 1e-10), axis=0)
    entropy_norm = float(np.mean(entropy_cols)) / 10.0  # normalize

    # Spektral Akı
    if stft.shape[1] > 1:
        flux = float(np.mean(np.sqrt(np.sum(np.diff(stft, axis=1)**2, axis=0))))
    else:
        flux = 0.0

    feats = np.hstack([
        mfcc_mean,
        np.float32(centroid),
        np.float32(hnr_val),
        np.float32(flatness),
        np.float32(low_ratio),
        np.float32(entropy_norm),
        np.float32(flux),
    ]).astype(np.float32)

    return np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=-1.0)


FEATURE_FUNCS_V2 = {
    "human": human_features_v2,
    "emergency": emergency_features_v2
}
