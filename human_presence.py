import numpy as np
import librosa
import sys

def detect_human_voice(y, sr):
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_mean = np.mean(zcr)

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_mean = np.mean(centroid)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_var = np.mean(np.var(mfcc, axis=1))

    score = 0

    # 🔴 CANLILIK
    if rms_mean > 0.015:
        score += 1
    if rms_std > 0.015:
        score += 1

    # 🔴 İNSAN SESİ BÖLGESİ
    if 300 < centroid_mean < 3500:
        score += 1

    # 🔴 MAKİNE ELEME
    if zcr_mean < 0.2:
        score += 1

    # 🔴 KAOTİK AMA CANLI SES (ÇIĞLIK / İNİLTİ)
    if mfcc_var > 50:
        score += 1

    confidence = score / 5
    is_human = confidence >= 0.6

    return is_human, confidence, {
        "rms_mean": float(rms_mean),
        "rms_std": float(rms_std),
        "zcr": float(zcr_mean),
        "centroid": float(centroid_mean),
        "mfcc_var": float(mfcc_var)
    }


if __name__ == "__main__":
    audio_path = sys.argv[1]
    y, sr = librosa.load(audio_path, sr=22050, duration=3)

    is_human, conf, details = detect_human_voice(y, sr)

    print("\nSONUÇ:")
    print(f"İnsan sesi: {'EVET' if is_human else 'HAYIR'}")
    print(f"Güven: %{conf*100:.1f}")
    print("Detaylar:", details)
