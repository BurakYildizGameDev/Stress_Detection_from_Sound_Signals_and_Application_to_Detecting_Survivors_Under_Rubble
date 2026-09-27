import os
import sys
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.export_c_model import (
    tree_to_c_code,
    export_random_forest_to_c,
    simulate_c_model_predict
)


def test_export_c_model_syntax_and_structure():
    # 3 sınıflı, 5 öznitelikli mini veri
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 5)).astype(np.float32)
    y = rng.integers(0, 3, size=100)
    classes = ["normal", "stress", "panic"]

    clf = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
    clf.fit(X, y)

    c_code = export_random_forest_to_c(clf, classes, model_name="rubble_test_model")

    # Temel C tanımlarını denetle
    assert "#ifndef TINYML_MODEL_H" in c_code
    assert "#define RUBBLE_TEST_MODEL_NUM_FEATURES 5" in c_code
    assert "#define RUBBLE_TEST_MODEL_NUM_CLASSES 3" in c_code
    assert "#define RUBBLE_TEST_MODEL_NUM_TREES 5" in c_code
    assert 'static const char* rubble_test_model_class_names[3] = {' in c_code
    assert '"normal",' in c_code
    assert '"stress",' in c_code
    assert '"panic",' in c_code
    assert "static void evaluate_tree_0" in c_code
    assert "static void evaluate_tree_4" in c_code
    assert "static int rubble_test_model_predict" in c_code


def test_simulate_c_model_matches_sklearn_probabilities():
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((80, 4)).astype(np.float32)
    y_train = rng.integers(0, 2, size=80)
    X_test = rng.standard_normal((20, 4)).astype(np.float32)

    clf = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=42)
    clf.fit(X_train, y_train)

    sklearn_probs = clf.predict_proba(X_test)
    pred_classes, c_sim_probs = simulate_c_model_predict(clf, X_test)

    # Scikit-learn ile C simülasyonunun olasılıkları tam olarak uyuşmalıdır
    np.testing.assert_allclose(sklearn_probs, c_sim_probs, atol=1e-5)
    np.testing.assert_array_equal(clf.predict(X_test), pred_classes)


def test_single_tree_export():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 3)).astype(np.float32)
    y = np.where(X[:, 0] > 0, 1, 0)
    classes = ["non_human", "human"]

    clf = RandomForestClassifier(n_estimators=1, max_depth=2, random_state=0)
    clf.fit(X, y)

    c_code = export_random_forest_to_c(clf, classes, model_name="single_tree_model")
    assert "#define SINGLE_TREE_MODEL_NUM_TREES 1" in c_code
    assert "evaluate_tree_0" in c_code
    assert "evaluate_tree_1" not in c_code
