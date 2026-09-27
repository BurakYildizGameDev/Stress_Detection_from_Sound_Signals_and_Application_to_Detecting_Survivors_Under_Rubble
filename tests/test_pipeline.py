import json
import os
import sys

import joblib
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pipeline
from features import FEATURE_SPECS
from pipeline import MODEL_FILES, ModelLoadError, load_model


def test_models_expect_extractor_feature_counts():
    for task in ("human", "emergency"):
        loaded = pipeline.get_model(task)
        assert loaded.model.n_features_in_ == FEATURE_SPECS[task]["n_features"]
        assert "normal" in pipeline.get_model("emergency").labels
        assert "human" in pipeline.get_model("human").labels


def test_silence_is_not_analyzed():
    y = np.zeros(pipeline.SR, dtype=np.float32)
    assert pipeline.analyze_audio_array(y, pipeline.SR)["status"] == "silence"


def test_analyze_file_returns_known_status():
    path = os.path.join(os.path.dirname(pipeline.__file__), "deneme sesleri", "carstart.mp3")
    res = pipeline.analyze_file(path)
    assert res["status"] in {"silence", "no_human", "DETECTED"}


def dummy_model(n_features, labels):
    X = np.zeros((len(labels), n_features))
    return DummyClassifier().fit(X, labels)


def test_missing_model_gives_clear_error(tmp_path):
    with pytest.raises(ModelLoadError, match="bulunamadı"):
        load_model("emergency", str(tmp_path))


def test_lfs_pointer_gives_clear_error(tmp_path):
    (tmp_path / MODEL_FILES["emergency"]).write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n")
    with pytest.raises(ModelLoadError, match="git lfs pull"):
        load_model("emergency", str(tmp_path))


def test_feature_count_mismatch_is_rejected(tmp_path):
    joblib.dump(dummy_model(10, ["normal", "panic"]), tmp_path / MODEL_FILES["emergency"])
    with pytest.raises(ModelLoadError, match="öznitelik bekliyor"):
        load_model("emergency", str(tmp_path))


def test_feature_spec_mismatch_is_rejected(tmp_path):
    n = FEATURE_SPECS["emergency"]["n_features"]
    path = tmp_path / MODEL_FILES["emergency"]
    joblib.dump(dummy_model(n, ["normal", "panic"]), path)
    spec = {**FEATURE_SPECS["emergency"], "version": 0}
    path.with_suffix(".json").write_text(json.dumps({"feature_spec": spec}))
    with pytest.raises(ModelLoadError, match="farklı bir öznitelik"):
        load_model("emergency", str(tmp_path))


def test_new_style_model_uses_string_labels(tmp_path):
    n = FEATURE_SPECS["emergency"]["n_features"]
    path = tmp_path / MODEL_FILES["emergency"]
    joblib.dump(dummy_model(n, ["normal", "panic", "stress"]), path)
    import sklearn
    path.with_suffix(".json").write_text(json.dumps(
        {"feature_spec": FEATURE_SPECS["emergency"], "sklearn_version": sklearn.__version__}))
    assert load_model("emergency", str(tmp_path)).labels == ["normal", "panic", "stress"]
