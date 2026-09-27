import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pipeline


def test_models_expect_extractor_feature_counts():
    from features import EMERGENCY_N_FEATURES, HUMAN_N_FEATURES
    assert pipeline.human_model.n_features_in_ == HUMAN_N_FEATURES
    assert pipeline.emergency_model.n_features_in_ == EMERGENCY_N_FEATURES


def test_silence_is_not_analyzed():
    y = np.zeros(pipeline.SR, dtype=np.float32)
    assert pipeline.analyze_audio_array(y, pipeline.SR)["status"] == "silence"


def test_analyze_file_returns_known_status():
    path = os.path.join(os.path.dirname(pipeline.__file__), "deneme sesleri", "carstart.mp3")
    res = pipeline.analyze_file(path)
    assert res["status"] in {"silence", "no_human", "DETECTED"}
