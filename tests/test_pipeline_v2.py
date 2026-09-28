import os
import sys
import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline_v2 import PipelineV2, RMS_SILENCE_THRESHOLD, ALERT_PRIORITY


class DummyClassifier:
    def __init__(self, classes, fixed_probs):
        self.classes_ = np.array(classes)
        self.fixed_probs = np.array(fixed_probs, dtype=np.float32)

    def predict_proba(self, X):
        return np.tile(self.fixed_probs, (len(X), 1))


def tone(sr=22050, sec=1.0, freq=440.0, amp=0.3):
    t = np.arange(int(sr * sec)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_pipeline_v2_silence_detection():
    pipeline = PipelineV2()
    y_silence = np.zeros(22050, dtype=np.float32)
    res = pipeline.analyze_audio_array(y_silence)

    assert res["status"] == "silence"
    assert res["is_emergency"] is False
    assert res["priority"] == "NONE"


def test_pipeline_v2_whisper_sensitivity_gate():
    pipeline = PipelineV2()
    # 0.003 genlikli zayıf fısıltı sinyali (V1'in 0.01 eşiğinde elenirdi, V2'de geçmeli)
    y_whisper = tone(amp=0.004)
    res = pipeline.analyze_audio_array(y_whisper)

    assert res["rms"] > RMS_SILENCE_THRESHOLD
    # Model henüz eğitilmediği için RAW_AUDIO dönmeli ama silence dönmemeli
    assert res["status"] in ("RAW_AUDIO", "DETECTED", "no_human")


def test_pipeline_v2_whisper_critical_priority_mock():
    # Mock modeller ile fısıltı tespitini test et
    mock_human = DummyClassifier(classes=["non_human", "human"], fixed_probs=[0.1, 0.9])
    # 5 sınıf: normal, stress, panic, moan, whisper
    mock_emerg = DummyClassifier(
        classes=["normal", "stress", "panic", "moan", "whisper"],
        fixed_probs=[0.05, 0.05, 0.1, 0.1, 0.7]  # whisper baskın
    )

    pipeline = PipelineV2(human_model=mock_human, emergency_model=mock_emerg)
    y = tone(amp=0.05)
    res = pipeline.analyze_audio_array(y)

    assert res["status"] == "DETECTED"
    assert res["state"] == "whisper"
    assert res["is_emergency"] is True
    assert res["priority"] == "CRITICAL_SURVIVOR"
    assert res["all_probs"]["whisper"] == pytest.approx(0.7)


def test_pipeline_v2_moan_critical_priority_mock():
    mock_human = DummyClassifier(classes=["non_human", "human"], fixed_probs=[0.1, 0.9])
    mock_emerg = DummyClassifier(
        classes=["normal", "stress", "panic", "moan", "whisper"],
        fixed_probs=[0.1, 0.1, 0.1, 0.6, 0.1]  # moan baskın
    )

    pipeline = PipelineV2(human_model=mock_human, emergency_model=mock_emerg)
    res = pipeline.analyze_audio_array(tone(amp=0.1))

    assert res["state"] == "moan"
    assert res["is_emergency"] is True
    assert res["priority"] == "CRITICAL_SURVIVOR"


def test_pipeline_v2_panic_high_emergency_priority_mock():
    mock_human = DummyClassifier(classes=["non_human", "human"], fixed_probs=[0.05, 0.95])
    mock_emerg = DummyClassifier(
        classes=["normal", "stress", "panic", "moan", "whisper"],
        fixed_probs=[0.1, 0.1, 0.7, 0.05, 0.05]  # panic baskın
    )

    pipeline = PipelineV2(human_model=mock_human, emergency_model=mock_emerg)
    res = pipeline.analyze_audio_array(tone(amp=0.2))

    assert res["state"] == "panic"
    assert res["is_emergency"] is True
    assert res["priority"] == "HIGH_EMERGENCY"


def test_pipeline_v2_normal_speech_no_alarm_mock():
    mock_human = DummyClassifier(classes=["non_human", "human"], fixed_probs=[0.05, 0.95])
    mock_emerg = DummyClassifier(
        classes=["normal", "stress", "panic", "moan", "whisper"],
        fixed_probs=[0.85, 0.05, 0.05, 0.02, 0.03]  # normal baskın
    )

    pipeline = PipelineV2(human_model=mock_human, emergency_model=mock_emerg)
    res = pipeline.analyze_audio_array(tone(amp=0.1))

    assert res["state"] == "normal"
    assert res["is_emergency"] is False
    assert res["priority"] == "NORMAL_SPEECH"


def test_pipeline_v2_non_human_rejected_mock():
    mock_human = DummyClassifier(classes=["non_human", "human"], fixed_probs=[0.95, 0.05])
    mock_emerg = DummyClassifier(
        classes=["normal", "stress", "panic", "moan", "whisper"],
        fixed_probs=[0.2, 0.2, 0.2, 0.2, 0.2]
    )

    pipeline = PipelineV2(human_model=mock_human, emergency_model=mock_emerg)
    res = pipeline.analyze_audio_array(tone(amp=0.2))

    assert res["status"] == "no_human"
    assert res["is_emergency"] is False
    assert res["priority"] == "NONE"


def test_pipeline_v2_file_mode(tmp_path):
    wav_path = str(tmp_path / "test_whisper.wav")
    y = tone(amp=0.05)
    sf.write(wav_path, y, 22050)

    mock_human = DummyClassifier(classes=["non_human", "human"], fixed_probs=[0.1, 0.9])
    mock_emerg = DummyClassifier(
        classes=["normal", "stress", "panic", "moan", "whisper"],
        fixed_probs=[0.1, 0.1, 0.1, 0.1, 0.6]
    )
    pipeline = PipelineV2(human_model=mock_human, emergency_model=mock_emerg)
    res = pipeline.analyze_file(wav_path)

    assert res["status"] == "DETECTED"
    assert res["state"] == "whisper"


def test_pipeline_v2_human_threshold_is_respected():
    mock_human = DummyClassifier(["human", "non_human"], [0.4, 0.6])
    mock_emerg = DummyClassifier(["moan", "normal", "panic", "stress", "whisper"],
                                 [0.8, 0.05, 0.05, 0.05, 0.05])
    strict = PipelineV2(human_model=mock_human, emergency_model=mock_emerg, human_threshold=0.5)
    loose = PipelineV2(human_model=mock_human, emergency_model=mock_emerg, human_threshold=0.3)
    assert strict.analyze_audio_array(tone())["status"] == "no_human"
    assert loose.analyze_audio_array(tone())["state"] == "moan"


def test_pipeline_v2_reads_threshold_from_metadata(tmp_path):
    import json
    import joblib
    from pipeline_v2 import MODEL_FILES_V2
    joblib.dump(DummyClassifier(["human", "non_human"], [0.5, 0.5]),
                tmp_path / MODEL_FILES_V2["human"])
    (tmp_path / MODEL_FILES_V2["human"].replace(".pkl", ".json")).write_text(
        json.dumps({"human_threshold": 0.73}), encoding="utf-8")
    assert PipelineV2(models_dir=str(tmp_path)).human_threshold == 0.73
