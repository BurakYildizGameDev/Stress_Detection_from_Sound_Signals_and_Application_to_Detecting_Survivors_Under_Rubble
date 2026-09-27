"""
export_c_model.py - Scikit-Learn Modellerini ESP32-S3 İçin C/C++ Başlık Dosyasına (.h) Dönüştürücü.

ESP32-S3 mikrodenetleyicisi üzerinde harici hiçbir kütüphane (Python, librosa vs.)
olmadan saf C/C++ ile mikrosaniye mertebesinde (sub-millisecond) yapay zeka
çıkarımı yapılmasını sağlar.

Desteklenenler:
- RandomForestClassifier ağaçlarının kompakt C dizilerine veya if-else dallanmalarına çevrilmesi
- Olasılık dağılımı (softmax/voting) ve sınıf tahmini
- ESP-IDF, Arduino IDE ve FreeRTOS ile %100 uyumluluk
"""
import os
import sys
import numpy as np


def tree_to_c_code(tree, tree_id, n_classes):
    """Tek bir DecisionTree'yi C if-else fonksiyonuna dönüştürür."""
    tree_ = tree.tree_
    feature = tree_.feature
    threshold = tree_.threshold
    value = tree_.value

    lines = []
    lines.append(f"static void evaluate_tree_{tree_id}(const float* f, float* votes) {{")

    def recurse(node, depth):
        indent = "    " * depth
        if feature[node] != -2:  # İç Düğüm (Internal Node)
            feat_idx = feature[node]
            thresh_val = threshold[node]
            lines.append(f"{indent}if (f[{feat_idx}] <= {thresh_val:.6f}f) {{")
            recurse(tree_.children_left[node], depth + 1)
            lines.append(f"{indent}}} else {{")
            recurse(tree_.children_right[node], depth + 1)
            lines.append(f"{indent}}}")
        else:  # Yaprak Düğüm (Leaf Node)
            # Yapraktaki sınıf dağılımı
            val = value[node][0]
            val_norm = val / np.sum(val)
            for c in range(n_classes):
                if val_norm[c] > 0.0001:
                    lines.append(f"{indent}votes[{c}] += {val_norm[c]:.6f}f;")

    recurse(0, 1)
    lines.append("}\n")
    return "\n".join(lines)


def export_random_forest_to_c(rf_model, class_names, model_name="tinyml_model"):
    """
    RandomForestClassifier nesnesini eksiksiz ve bağımsız bir C/C++ başlık stringine dönüştürür.
    """
    n_estimators = len(rf_model.estimators_)
    n_classes = len(class_names)
    n_features = rf_model.n_features_in_

    c_code = []
    c_code.append("/*")
    c_code.append(f" * Otomatik Üretilen ESP32-S3 TinyML Modeli: {model_name}")
    c_code.append(f" * Ağaç Sayısı: {n_estimators}, Öznitelik Sayısı: {n_features}, Sınıflar: {n_classes}")
    c_code.append(" * Bağımlılık: YOK (Pure C99 / C++11)")
    c_code.append(" */")
    c_code.append("#ifndef TINYML_MODEL_H")
    c_code.append("#define TINYML_MODEL_H")
    c_code.append("\n#include <stdint.h>\n#include <string.h>\n")
    c_code.append(f"#define {model_name.upper()}_NUM_FEATURES {n_features}")
    c_code.append(f"#define {model_name.upper()}_NUM_CLASSES {n_classes}")
    c_code.append(f"#define {model_name.upper()}_NUM_TREES {n_estimators}\n")

    # Sınıf isimleri
    c_code.append(f"static const char* {model_name}_class_names[{n_classes}] = {{")
    for name in class_names:
        c_code.append(f'    "{name}",')
    c_code.append("};\n")

    # Her bir ağacın C fonksiyonu
    for i, est in enumerate(rf_model.estimators_):
        c_code.append(tree_to_c_code(est, i, n_classes))

    # Ana tahmin fonksiyonu
    c_code.append(f"""
static int {model_name}_predict(const float* features, float* out_probabilities) {{
    float votes[{n_classes}];
    memset(votes, 0, sizeof(votes));

    // Tüm ağaçların kararlarını topla
""")
    for i in range(n_estimators):
        c_code.append(f"    evaluate_tree_{i}(features, votes);")

    c_code.append(f"""
    // Olasılıkları normalize et
    int best_class = 0;
    float max_prob = -1.0f;
    for (int c = 0; c < {n_classes}; c++) {{
        float p = votes[c] / (float){n_estimators};
        if (out_probabilities != 0) {{
            out_probabilities[c] = p;
        }}
        if (p > max_prob) {{
            max_prob = p;
            best_class = c;
        }}
    }}
    return best_class;
}}

#endif // TINYML_MODEL_H
""")

    return "\n".join(c_code)


def simulate_c_model_predict(rf_model, X):
    """
    Python'da C kodunun yapacağı hesaplamanın aynısını simüle eder (doğrulama için).
    """
    n_estimators = len(rf_model.estimators_)
    all_votes = np.zeros((len(X), len(rf_model.classes_)), dtype=np.float32)

    for est in rf_model.estimators_:
        preds = est.predict_proba(X)
        all_votes += preds

    probs = all_votes / float(n_estimators)
    pred_classes = np.argmax(probs, axis=1)
    return pred_classes, probs
