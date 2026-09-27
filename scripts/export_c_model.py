"""
export_c_model.py - RandomForestClassifier modelini bağımlılıksız bir C başlık
dosyasına (.h) dönüştürür (ESP32-S3 hedefi, C99 / C++11).

Her ağaç iç içe if-else bloklarından oluşan bir fonksiyona çevrilir. Üretilen
kod sklearn'ün `predict_proba` sonucunu float hassasiyetinde yeniden üretir:

- Eşikler float32'ye AŞAĞI yuvarlanır. sklearn öznitelikleri float32'ye çevirip
  float64 eşikle karşılaştırır (x <= t). t'den büyük olmayan en büyük float32
  değeri t32 ise her float32 x için (x <= t) ile (x <= t32) aynı sonucu verir.
  En yakın float32'ye yuvarlamak bu garantiyi bozar.
- Sayılar `%.9g` ile yazılır; float32 değerleri kayıpsız geri okunur.
- Sınıf sırası modelin `classes_` sırasıdır; farklı sıra verilirse hata verilir.
- Tüm tanımlar model adıyla öneklenir; iki model aynı derleme biriminde
  (ör. insan dedektörü + acil durum sınıflandırıcısı) birlikte kullanılabilir.

Uyarı: v2 modelleri (250-350 ağaç, derinlik 20+) yüz MB'larca C kodu üretir ve
ESP32-S3 flash'ına sığmaz. Gömülü kullanım için küçük bir model gerekir.

Kullanım:
    python scripts/export_c_model.py models/human_detector_v2.pkl --name human_detector
    python scripts/export_c_model.py models/emergency_classifier_v2.pkl \\
        --name emergency_classifier --threshold 0.45 --out firmware/include
"""
import argparse
import json
import os
import re
import sys

import joblib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT_DIR = os.path.join(ROOT, "firmware", "include")

# Bu düğüm sayısının üstündeki modeller ESP32-S3 için gerçekçi değil
# (if-else kodu kabaca düğüm başına 15-20 bayt flash tutar).
EMBEDDED_NODE_WARN = 100_000
# Bundan büyük başlık dosyası --allow-large olmadan yazılmaz (GitHub dosya sınırı 100 MB).
MAX_HEADER_MB = 50

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def c_float(x):
    """Python sayısını geçerli bir C float sabitine çevirir (ör. 3 -> 3.0f)."""
    if not np.isfinite(x):
        raise ValueError(f"C sabitine çevrilemeyen değer: {x}")
    s = f"{float(x):.9g}"
    if not any(ch in s for ch in ".e"):
        s += ".0"
    return s + "f"


def float32_threshold(t):
    """t'den büyük olmayan en büyük float32 değeri (bkz. modül açıklaması)."""
    t32 = np.float32(t)
    if float(t32) > t:
        t32 = np.nextafter(t32, np.float32(-np.inf))
    return t32


def _label_macro(label):
    return re.sub(r"[^A-Za-z0-9]", "_", str(label)).upper()


def tree_to_c_code(tree, func_name):
    """Tek bir DecisionTree'yi if-else gövdeli bir C fonksiyonuna dönüştürür."""
    tree_ = tree.tree_
    feature = tree_.feature
    threshold = tree_.threshold
    value = tree_.value
    left = tree_.children_left
    right = tree_.children_right

    lines = [f"static void {func_name}(const float* f, float* votes) {{"]

    def recurse(node, depth):
        indent = "    " * depth
        if left[node] != right[node]:  # iç düğüm
            t32 = float32_threshold(threshold[node])
            lines.append(f"{indent}if (f[{feature[node]}] <= {c_float(t32)}) {{")
            recurse(left[node], depth + 1)
            lines.append(f"{indent}}} else {{")
            recurse(right[node], depth + 1)
            lines.append(f"{indent}}}")
        else:  # yaprak: sınıf olasılıkları
            val = value[node][0]
            val = val / np.sum(val)
            for c, p in enumerate(val):
                if p > 0.0:
                    lines.append(f"{indent}votes[{c}] += {c_float(p)};")

    recurse(0, 1)
    lines.append("}\n")
    return "\n".join(lines)


def export_random_forest_to_c(rf_model, class_names=None, model_name="tinyml_model",
                              threshold=None, feature_spec=None):
    """
    RandomForestClassifier'ı bağımsız bir C başlık dosyası metnine dönüştürür.

    class_names  None ise rf_model.classes_ kullanılır; verilirse aynı sırada olmalı.
    threshold    Karar eşiği (ör. insan dedektörü için human_threshold);
                 verilirse <MODEL>_THRESHOLD olarak yazılır.
    feature_spec Model .json'undaki feature_spec; örnekleme hızı ve öznitelik
                 tanımı başlığa yazılır.
    """
    if not _IDENT_RE.match(model_name):
        raise ValueError(f"model_name geçerli bir C tanımlayıcısı değil: {model_name!r}")

    model_classes = [str(c) for c in rf_model.classes_]
    if class_names is None:
        class_names = model_classes
    elif [str(c) for c in class_names] != model_classes:
        raise ValueError(f"class_names {list(class_names)} modelin sınıf sırasıyla "
                         f"{model_classes} aynı değil")

    n_estimators = len(rf_model.estimators_)
    n_classes = len(class_names)
    n_features = rf_model.n_features_in_
    n_nodes = sum(e.tree_.node_count for e in rf_model.estimators_)
    prefix = model_name.upper()
    guard = f"{prefix}_MODEL_H"

    c = []
    c.append("/*")
    c.append(f" * Otomatik üretildi: scripts/export_c_model.py ({model_name})")
    c.append(f" * Ağaç: {n_estimators}, düğüm: {n_nodes}, öznitelik: {n_features}, sınıf: {n_classes}")
    if feature_spec:
        c.append(f" * Öznitelikler: {feature_spec.get('desc', '')}")
    c.append(" * Bağımlılık yok (C99 / C++11, #include bile yok). Elle düzenlemeyin.")
    c.append(" */")
    c.append(f"#ifndef {guard}")
    c.append(f"#define {guard}")
    c.append("")
    c.append(f"#define {prefix}_NUM_FEATURES {n_features}")
    c.append(f"#define {prefix}_NUM_CLASSES {n_classes}")
    c.append(f"#define {prefix}_NUM_TREES {n_estimators}")
    for i, name in enumerate(class_names):
        c.append(f"#define {prefix}_CLASS_{_label_macro(name)} {i}")
    if threshold is not None:
        c.append(f"#define {prefix}_THRESHOLD {c_float(threshold)}")
    if feature_spec:
        if feature_spec.get("sr"):
            c.append(f"#define {prefix}_SAMPLE_RATE {int(feature_spec['sr'])}")
        if feature_spec.get("version") is not None:
            c.append(f"#define {prefix}_FEATURE_VERSION {int(feature_spec['version'])}")
    c.append("")

    names = ", ".join(json.dumps(n, ensure_ascii=False) for n in class_names)
    c.append(f"static const char* const {model_name}_class_names[{n_classes}] = {{{names}}};\n")

    for i, est in enumerate(rf_model.estimators_):
        c.append(tree_to_c_code(est, f"{model_name}_tree_{i}"))

    c.append(f"/* features: {n_features} float; out_probabilities: {n_classes} float veya NULL.")
    c.append("   Dönüş: en olası sınıfın indeksi. */")
    c.append(f"static int {model_name}_predict(const float* features, float* out_probabilities) {{")
    c.append(f"    float votes[{n_classes}] = {{0}};")
    for i in range(n_estimators):
        c.append(f"    {model_name}_tree_{i}(features, votes);")
    c.append(f"""
    int best_class = 0;
    float max_prob = -1.0f;
    for (int c = 0; c < {n_classes}; c++) {{
        float p = votes[c] / (float){n_estimators};
        if (out_probabilities) out_probabilities[c] = p;
        if (p > max_prob) {{
            max_prob = p;
            best_class = c;
        }}
    }}
    return best_class;
}}

#endif /* {guard} */
""")
    return "\n".join(c)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help=".pkl model dosyası (RandomForestClassifier)")
    ap.add_argument("--name", required=True, help="C önekleri için model adı (ör. human_detector)")
    ap.add_argument("--out", default=DEFAULT_OUT_DIR, help="çıktı klasörü (varsayılan: firmware/include)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="karar eşiği; verilmezse model .json'undaki human_threshold "
                         "ya da emergency_threshold kullanılır")
    ap.add_argument("--allow-large", action="store_true",
                    help=f"{MAX_HEADER_MB} MB'tan büyük başlık dosyasını da yaz")
    args = ap.parse_args()

    rf = joblib.load(args.model)
    if not hasattr(rf, "estimators_"):
        sys.exit(f"Desteklenmeyen model türü: {type(rf).__name__} (RandomForestClassifier bekleniyor)")

    meta = {}
    meta_path = os.path.splitext(args.model)[0] + ".json"
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    threshold = args.threshold
    if threshold is None:
        threshold = next((meta[k] for k in ("human_threshold", "emergency_threshold")
                          if meta.get(k) is not None), None)

    n_nodes = sum(e.tree_.node_count for e in rf.estimators_)
    if n_nodes > EMBEDDED_NODE_WARN:
        print(f"UYARI: {n_nodes} düğüm. Üretilen kod ESP32-S3 flash'ına sığmaz; "
              "gömülü kullanım için küçük bir model eğitin.")

    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10_000))
    code = export_random_forest_to_c(rf, model_name=args.name, threshold=threshold,
                                     feature_spec=meta.get("feature_spec"))

    size_mb = len(code.encode("utf-8")) / 1e6
    if size_mb > MAX_HEADER_MB and not args.allow_large:
        sys.exit(f"Başlık dosyası {size_mb:.0f} MB; yazılmadı (sınır {MAX_HEADER_MB} MB, "
                 "yine de yazmak için --allow-large).")
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"{args.name}_model.h")
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(code)
    print(f"{out_path}: {len(code.encode('utf-8')) / 1e6:.1f} MB, {n_nodes} düğüm, "
          f"eşik={threshold}")


if __name__ == "__main__":
    main()
