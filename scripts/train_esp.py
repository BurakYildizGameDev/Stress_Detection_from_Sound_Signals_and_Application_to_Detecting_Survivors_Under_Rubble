"""
train_esp.py - ESP32-S3 flash'ına sığacak küçük v2 modellerini arar, boyut ile
doğruluk arasındaki dengeyi raporlar ve bütçeye sığan en iyi modeli kaydeder.

v2 modelleri (250/350 ağaç, derinlik 20+) 940 bin / 2,3 milyon düğümdür; C'ye
çevrilince flash'a sığmaz. Bu betik ağaç sayısı x derinlik ızgarasını aynı
öznitelikler (features/*_v2.npz) ve aynı değerlendirme ile dener.

Flash boyutu:
  --flash-cc verilirse her model export_c_model.py ile C'ye çevrilip gerçekten
  derlenir ve nesne dosyasındaki kod + sabit veri boyutu okunur. Xtensa derleyicisi
  yoksa yakın bir 32 bit hedef kullanılabilir (ARM Thumb, -Os). Verilmezse düğüm
  sayısı x BYTES_PER_NODE ile tahmin edilir. İkisi de ESP32-S3 üzerindeki gerçek
  boyut değil, yaklaşık değerdir.

Seçim (test verisine bakılmaz):
  Eğitim gruplarının %15'i doğrulamaya ayrılır (train_v2 eşik seçimiyle aynı
  gruplar). Her aday kalan %85 ile eğitilip doğrulamada puanlanır:
  human      doğrulamada seçilen eşikte (yanlış alarm <= hedef) dengeli doğruluk
             ((insan recall + (1 - yanlış alarm)) / 2), orijinal kayıtlar
  emergency  doğrulamada temiz (original) ve enkaz (rubble_random) kayıtlarında
             ortalama macro-F1
  Bütçeye sığanlar içinde en iyisi tüm eğitim verisiyle yeniden eğitilir ve test
  bölmesinde yalnızca bir kez ölçülür (v2 referansıyla birlikte).

Sınıf ağırlıkları: v2'nin maliyet ağırlıkları (fısıltı x8, inleme x6) sığ
ağaçlarda normal konuşmanın %59-93'ünü acil durum yaptı (tam modelde %23).
Bu yüzden acil durum için "balanced" ve ağırlıksız denenir.

Çıktılar:
  reports/esp_model_sweep.json
  models/human_detector_esp.pkl, models/emergency_classifier_esp.pkl (+ .json)

Kullanım:
    python scripts/train_esp.py
    python scripts/train_esp.py --task human --flash-cc "python -m ziglang cc -target thumb-freestanding-eabi -mcpu=cortex_m4 -Os"
"""
import argparse
import json
import os
import shlex
import struct
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import EVAL_ONLY_DATASETS, MODELS_DIR, ROOT
from evaluation import summarize
from features_v2 import FEATURE_SPECS_V2
from pipeline_v2 import EMERGENCY_PROB_THRESHOLD, MODEL_FILES_V2
from scripts.export_c_model import export_random_forest_to_c
from scripts.train_v2 import (LABELS, TARGET_FALSE_ALARM, load_features_v2,
                              pick_human_threshold, validation_split)

TASKS = ("human", "emergency")
REPORT_PATH = os.path.join(ROOT, "reports", "esp_model_sweep.json")
ESP_MODEL_FILES = {"human": "human_detector_esp.pkl", "emergency": "emergency_classifier_esp.pkl"}
C_NAMES = {"human": "human_detector", "emergency": "emergency_classifier"}

N_TREES = (10, 25, 50)
MAX_DEPTHS = (6, 8, 10, 12)
MIN_SAMPLES_LEAF = 5
CLASS_WEIGHTS = {"human": ("balanced",), "emergency": ("balanced", None)}

# İki model + firmware (ESP-IDF, I2S, FFT) varsayılan 3 MB'lık "huge_app"
# bölümüne rahat sığsın diye model başına bütçe.
FLASH_BUDGET_KB = {"human": 512, "emergency": 768}

# ARM Thumb -Os ile ölçüldü (4260 düğüm -> 77,5 KB); --flash-cc yoksa kullanılır.
BYTES_PER_NODE = 18.2


def make_esp_model(n_trees, max_depth, class_weight):
    return RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth,
                                  min_samples_leaf=MIN_SAMPLES_LEAF, class_weight=class_weight,
                                  random_state=42, n_jobs=-1)


def n_nodes(model):
    return int(sum(e.tree_.node_count for e in model.estimators_))


def alloc_section_sizes(obj_bytes):
    """ELF nesne dosyasında belleğe yüklenen (SHF_ALLOC) bölümlerin boyutları."""
    b = obj_bytes
    if b[:4] != b"\x7fELF":
        raise ValueError("ELF dosyası değil")
    is64, e = b[4] == 2, "<" if b[5] == 1 else ">"
    if is64:
        shoff = struct.unpack_from(e + "Q", b, 0x28)[0]
        shentsize, shnum, shstrndx = struct.unpack_from(e + "HHH", b, 0x3A)
    else:
        shoff = struct.unpack_from(e + "I", b, 0x20)[0]
        shentsize, shnum, shstrndx = struct.unpack_from(e + "HHH", b, 0x2E)

    def header(i):
        o = shoff + i * shentsize
        if is64:
            name, _ = struct.unpack_from(e + "II", b, o)
            flags, _, offset, size = struct.unpack_from(e + "QQQQ", b, o + 8)
        else:
            name, _, flags, _, offset, size = struct.unpack_from(e + "IIIIII", b, o)
        return name, flags, offset, size

    strtab = header(shstrndx)[2]
    sizes = {}
    for i in range(shnum):
        name, flags, _, size = header(i)
        if flags & 0x2:  # SHF_ALLOC
            start = strtab + name
            sizes[b[start:b.index(b"\0", start)].decode()] = size
    return sizes


def measure_flash_bytes(model, cc_cmd, name="m"):
    """Modeli C'ye çevirip derler, nesne dosyasının yüklenen bölümlerini toplar.
    Hedef ve optimizasyon bayrakları cc_cmd'den gelir (raporda flash_compiler);
    raporlanan sayılar -Os ile ARM Thumb derlemesidir. Sonuç model nesnesinin
    boyutudur, firmware'in toplam flash kullanımı değil (o, pio run çıktısında)."""
    code = export_random_forest_to_c(model, model_name=name)
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "m.h"), "w", encoding="utf-8") as f:
            f.write(code)
        src = os.path.join(tmp, "m.c")
        with open(src, "w", encoding="utf-8") as f:
            f.write(f'#include "m.h"\nint entry(const float* x, float* p) {{ return {name}_predict(x, p); }}\n')
        obj = os.path.join(tmp, "m.o")
        subprocess.run(cc_cmd + ["-c", src, "-o", obj], check=True, capture_output=True, text=True)
        with open(obj, "rb") as f:
            return int(sum(alloc_section_sizes(f.read()).values()))


def human_metrics(d, model, threshold):
    """Test bölmesi, orijinal kayıtlar: koşul başına yanlış alarm, insan recall ve
    VIVAE (gerçek sözsüz vokalizasyon) recall."""
    p_human = model.predict_proba(d["X"])[:, list(model.classes_).index("human")]
    is_human = p_human >= threshold
    test = (d["split"] == "test") & (d["synth"] == "none")
    out = {}
    for cond in dict.fromkeys(d["condition"][test].tolist()):
        at = test & (d["condition"] == cond)
        pos, neg = at & (d["y"] == "human"), at & (d["y"] == "non_human")
        vivae = pos & (d["dataset"] == "vivae")
        recall, fa = float(is_human[pos].mean()), float(is_human[neg].mean())
        out[cond] = {"human_recall": recall, "false_alarm": fa,
                     "vivae_recall": float(is_human[vivae].mean()) if vivae.any() else None,
                     "balanced_accuracy": (recall + 1.0 - fa) / 2.0}
    return out


def emergency_metrics(d, model):
    """Test bölmesi, korpus (dış veri hariç): koşul başına macro-F1 ve sınıf recall'ları."""
    pred = model.predict(d["X"])
    test = (d["split"] == "test") & ~np.isin(d["dataset"], sorted(EVAL_ONLY_DATASETS))
    out = {}
    for cond in dict.fromkeys(d["condition"][test].tolist()):
        at = test & (d["condition"] == cond)
        s = summarize("emergency", d["y"][at], pred[at], LABELS["emergency"])
        out[cond] = {"macro_f1": s["macro_f1"], "false_alarm": s["false_alarm_rate"],
                     "miss": s["miss_rate"],
                     "recall": {k: v["recall"] for k, v in s["per_class"].items()}}
    return out


def validation_metrics(task, d, is_val, model, target_fa):
    """Aday modeli doğrulama gruplarında puanlar. Dönüş: (metrikler, skor, eşik)."""
    if task == "human":
        val = is_val & (d["transform"] == "original") & (d["synth"] == "none")
        p_human = model.predict_proba(d["X"][val])[:, list(model.classes_).index("human")]
        y = d["y"][val]
        threshold = pick_human_threshold(p_human, y, target_fa)
        is_human = p_human >= threshold
        recall = float(is_human[y == "human"].mean())
        fa = float(is_human[y == "non_human"].mean())
        m = {"human_recall": recall, "false_alarm": fa, "balanced_accuracy": (recall + 1.0 - fa) / 2.0,
             "n": int(val.sum())}
        return m, m["balanced_accuracy"], threshold
    m = {}
    for name, transform in (("clean", "original"), ("rubble", "rubble_random")):
        at = is_val & (d["transform"] == transform)
        s = summarize("emergency", d["y"][at], model.predict(d["X"][at]), LABELS["emergency"])
        m[name] = {"macro_f1": s["macro_f1"], "false_alarm": s["false_alarm_rate"],
                   "miss": s["miss_rate"], "n": s["n"]}
    return m, (m["clean"]["macro_f1"] + m["rubble"]["macro_f1"]) / 2.0, None


def report_score(task, metrics):
    """Yalnızca raporlama için (seçimde kullanılmaz)."""
    if task == "human":
        return metrics["clean"]["balanced_accuracy"]
    return (metrics["clean"]["macro_f1"] + metrics["rubble_physical_severe"]["macro_f1"]) / 2.0


def select_best(rows, budget_bytes):
    """Bütçeye sığanlar içinde en yüksek skor; eşitlikte daha küçük model."""
    fits = [r for r in rows if r["flash_bytes"] <= budget_bytes]
    if not fits:
        return None
    return max(fits, key=lambda r: (round(r["score"], 4), -r["flash_bytes"]))


def flash_bytes(model, cc_cmd):
    return measure_flash_bytes(model, cc_cmd) if cc_cmd else int(n_nodes(model) * BYTES_PER_NODE)


def evaluate_config(task, d, fit, is_val, n_trees, max_depth, class_weight, target_fa, cc_cmd):
    """Adayı %85 ile eğitir, doğrulamada puanlar. Test verisi kullanılmaz."""
    t0 = time.time()
    model = make_esp_model(n_trees, max_depth, class_weight).fit(d["X"][fit], d["y"][fit])
    metrics, val_score, threshold = validation_metrics(task, d, is_val, model, target_fa)
    return {"n_trees": n_trees, "max_depth": max_depth, "min_samples_leaf": MIN_SAMPLES_LEAF,
            "class_weight": class_weight or "none",
            "nodes": n_nodes(model), "flash_bytes": flash_bytes(model, cc_cmd),
            "flash_measured": cc_cmd is not None, "threshold": threshold,
            "val_metrics": metrics, "score": val_score, "seconds": round(time.time() - t0, 1)}


def reference_row(task, d):
    """Kaydedilmiş tam v2 modeli, aynı ölçütlerle (karşılaştırma için)."""
    path = os.path.join(MODELS_DIR, MODEL_FILES_V2[task])
    if not os.path.exists(path):
        return None
    model = joblib.load(path)
    threshold = None
    if task == "human":
        with open(os.path.splitext(path)[0] + ".json", encoding="utf-8") as f:
            threshold = json.load(f).get("human_threshold") or 0.5
    metrics = human_metrics(d, model, threshold) if task == "human" else emergency_metrics(d, model)
    nodes = n_nodes(model)
    return {"model": os.path.basename(path), "n_trees": len(model.estimators_),
            "max_depth": int(max(e.tree_.max_depth for e in model.estimators_)),
            "nodes": nodes, "flash_bytes_estimated": int(nodes * BYTES_PER_NODE),
            "threshold": threshold, "metrics": metrics, "test_score": report_score(task, metrics)}


def print_val_row(task, r, mark=""):
    m = r["val_metrics"]
    if task == "human":
        detail = f"recall {m['human_recall']:.3f}  FA {m['false_alarm']:.3f}  eşik {r['threshold']:.2f}"
    else:
        detail = f"F1 temiz {m['clean']['macro_f1']:.3f}  enkaz {m['rubble']['macro_f1']:.3f}"
    print(f"  {r['class_weight']:8s} {r['n_trees']:4d} ağaç  d={r['max_depth']:2d}  {r['nodes']:8d} düğüm  "
          f"{r['flash_bytes'] / 1024:9.0f} KB  doğrulama {r['score']:.3f} | {detail} {mark}")


def print_test_row(task, r):
    m = r["metrics"]
    if task == "human":
        c = m["clean"]
        detail = (f"recall {c['human_recall']:.3f}  FA {c['false_alarm']:.3f}  "
                  f"VIVAE {c['vivae_recall'] if c['vivae_recall'] is not None else float('nan'):.3f}  "
                  f"eşik {r['threshold']:.2f}")
    else:
        detail = (f"F1 temiz {m['clean']['macro_f1']:.3f}  ağır {m['rubble_physical_severe']['macro_f1']:.3f}  "
                  f"FA {m['clean']['false_alarm']:.3f}  kaçırma {m['clean']['miss']:.3f}")
    flash = r.get("flash_bytes", r.get("flash_bytes_estimated"))
    print(f"  {r.get('class_weight', 'v2'):8s} {r['n_trees']:4d} ağaç  d={r['max_depth']:2d}  {r['nodes']:8d} düğüm  "
          f"{flash / 1024:9.0f} KB  test {r['test_score']:.3f} | {detail}")


def run_task(task, target_fa, cc_cmd, save):
    d = load_features_v2(task)
    train = d["split"] == "train"
    budget = FLASH_BUDGET_KB[task] * 1024
    print(f"\n=== {task} (bütçe {FLASH_BUDGET_KB[task]} KB, "
          f"flash {'ölçülüyor' if cc_cmd else 'tahmin'}) ===")

    ref = reference_row(task, d)
    if ref:
        print("Referans (tam v2 modeli, test):")
        print_test_row(task, ref)

    fit, is_val = validation_split(d, train)
    print(f"Aday seçimi doğrulamada: {len(set(d['group'][is_val]))} grup "
          f"({is_val.sum()} satır), eğitim {fit.sum()} satır")
    rows = []
    for class_weight in CLASS_WEIGHTS[task]:
        for n_trees in N_TREES:
            for max_depth in MAX_DEPTHS:
                row = evaluate_config(task, d, fit, is_val, n_trees, max_depth, class_weight,
                                      target_fa, cc_cmd)
                rows.append(row)
                print_val_row(task, row, "" if row["flash_bytes"] <= budget else "(bütçe aşıldı)")

    best = select_best(rows, budget)
    result = {"budget_kb": FLASH_BUDGET_KB[task], "selection": "validation groups (15% of train)",
              "reference_v2": ref, "sweep": rows,
              "selected": best and {k: best[k] for k in ("class_weight", "n_trees", "max_depth")}}
    if best is None:
        print("Bütçeye sığan model yok.")
        return result

    # Seçilen ayar tüm eğitim verisiyle; eşik doğrulamada seçilen (train_v2 ile aynı yöntem)
    weight = None if best["class_weight"] == "none" else best["class_weight"]
    model = make_esp_model(best["n_trees"], best["max_depth"], weight).fit(d["X"][train], d["y"][train])
    threshold = best["threshold"]
    metrics = human_metrics(d, model, threshold) if task == "human" else emergency_metrics(d, model)
    final = {"class_weight": best["class_weight"], "n_trees": best["n_trees"],
             "max_depth": best["max_depth"], "nodes": n_nodes(model),
             "flash_bytes": flash_bytes(model, cc_cmd), "flash_measured": cc_cmd is not None,
             "threshold": threshold, "metrics": metrics, "test_score": report_score(task, metrics)}
    result["final"] = final
    print("Seçilen (tüm eğitim verisiyle, test bölmesinde bir kez):")
    print_test_row(task, final)

    if save:
        path = os.path.join(MODELS_DIR, ESP_MODEL_FILES[task])
        joblib.dump(model, path, compress=3)
        meta = {
            "task": f"{task}_esp",
            "labels": [str(c) for c in model.classes_],
            "feature_spec": FEATURE_SPECS_V2[task],
            "sklearn_version": sklearn.__version__,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "params": {"n_estimators": best["n_trees"], "max_depth": best["max_depth"],
                       "min_samples_leaf": MIN_SAMPLES_LEAF, "class_weight": best["class_weight"]},
            "nodes": final["nodes"],
            "flash_bytes": final["flash_bytes"],
            "flash_measured": final["flash_measured"],
            "human_threshold": threshold,
            "emergency_threshold": EMERGENCY_PROB_THRESHOLD if task == "emergency" else None,
            "selection": "validation groups (15% of train); test evaluated once",
            "validation_metrics": best["val_metrics"],
            "metrics": metrics,
            "reference_v2_metrics": ref and ref["metrics"],
            "c_name": C_NAMES[task],
            "report": os.path.relpath(REPORT_PATH, ROOT).replace(os.sep, "/"),
        }
        with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"-> {os.path.relpath(path, ROOT)} (+ .json)")
    return result


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=TASKS + ("all",), default="all")
    p.add_argument("--flash-cc", default=None,
                   help="flash boyutunu ölçmek için çapraz derleyici komutu (ör. zig cc -target ... -Os)")
    p.add_argument("--target-false-alarm", type=float, default=TARGET_FALSE_ALARM)
    p.add_argument("--no-save", action="store_true", help="modeli kaydetme, yalnızca raporla")
    args = p.parse_args()

    cc_cmd = shlex.split(args.flash_cc, posix=os.name != "nt") if args.flash_cc else None
    report = {}
    if os.path.exists(REPORT_PATH):
        with open(REPORT_PATH, encoding="utf-8") as f:
            report = json.load(f)
    report.update({
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "flash_compiler": args.flash_cc,
        "flash_note": "ölçüm verilen derleyicinin nesne dosyasıdır, ESP32-S3 (Xtensa) değil; yaklaşık"
                      if args.flash_cc else f"düğüm x {BYTES_PER_NODE} bayt tahmini",
    })
    for t in TASKS if args.task == "all" else (args.task,):
        report[t] = run_task(t, args.target_false_alarm, cc_cmd, save=not args.no_save)

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n-> {os.path.relpath(REPORT_PATH, ROOT)}")


if __name__ == "__main__":
    main()
