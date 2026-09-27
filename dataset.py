"""
Veri setleri, etiketler ve bölmeler için tek kaynak.

İndirici (scripts/download_data.py), manifest oluşturucu, öznitelik çıkarımı ve
eğitim scriptleri yolları ve etiket eşlemesini buradan alır.

Etiketler dosya adındaki anahtar kelimelerden değil, her veri setinin kendi
isimlendirme şemasından okunur. Her kayıt en fazla bir acil durum sınıfına düşer.
"""
import csv
import hashlib
import os
import random
import re
from dataclasses import asdict, dataclass, fields

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
FEATURES_DIR = os.path.join(ROOT, "features")
MODELS_DIR = os.path.join(ROOT, "models")
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.csv")

TEST_SPEAKER_FRACTION = 0.2
SPLIT_SEED = 42

# =========================
# SINIFLAR
# =========================
EMERGENCY_CLASSES = ("normal", "stress", "panic")

# Veri setlerinin duygu etiketi -> acil durum sınıfı. Burada olmayan duygular
# (happy, sad, disgust, surprise, JL'nin ikincil duyguları...) acil durum
# modeline girmez; insan-sesi modelinde "human" olarak kullanılır.
EMOTION_TO_CLASS = {
    "neutral": "normal",
    "calm": "normal",
    "angry": "stress",
    "fear": "panic",
}

# Aşama 1: Fısıltı ve İnleme destekli 5 sınıflı acil durum yapısı
EMERGENCY_CLASSES_V2 = ("normal", "stress", "panic", "moan", "whisper")
EMOTION_TO_CLASS_V2 = {
    **EMOTION_TO_CLASS,
    "moan": "moan",
    "whisper": "whisper",
    # VIVAE: yalnızca düşük/orta şiddetli ağrı vokalizasyonu inlemeye karşılık
    # gelir; güçlü/zirve ağrı çığlığa dönüşür ve hiçbir sınıfa atanmaz.
    "pain_low": "moan",
    "pain_moderate": "moan",
}

# ESC-50 içindeki insan kaynaklı sesli sınıflar "non_human" diye etiketlenemez.
ESC50_HUMAN_VOCAL = {
    20: "crying_baby", 21: "sneezing", 23: "breathing",
    24: "coughing", 26: "laughing", 28: "snoring",
}


class Skip(Exception):
    """Dosya manifest'e alınmaz; mesaj atlanma nedenidir."""


# =========================
# VERİ SETLERİ
# =========================
def _ravdess(name, _):
    # 03-02-06-01-02-01-12.wav: modalite-kanal-duygu-yoğunluk-cümle-tekrar-aktör
    p = name[:-4].split("-")
    if len(p) != 7:
        raise Skip("RAVDESS şemasına uymuyor")
    emotions = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad",
                "05": "angry", "06": "fear", "07": "disgust", "08": "surprise"}
    return f"actor{p[6]}", emotions[p[2]], name[:-4]


def _berlin(name, _):
    # 03a01Fa.wav: konuşmacı(2) metin(3) duygu(1) sürüm(1)
    m = re.fullmatch(r"(\d\d)([a-z]\d\d)([WLEAFTN])([a-z])\.wav", name)
    if not m:
        raise Skip("EMO-DB şemasına uymuyor")
    emotions = {"W": "angry", "L": "boredom", "E": "disgust", "A": "fear",
                "F": "happy", "T": "sad", "N": "neutral"}
    return m.group(1), emotions[m.group(3)], name[:-4]


def _tess(name, _):
    # OAF_back_angry.wav: konuşmacı_kelime_duygu. Hedef kelime ("shout" dahil)
    # etiket değildir.
    p = name[:-4].split("_")
    if len(p) != 3:
        raise Skip("TESS şemasına uymuyor")
    speaker = {"OA": "OAF"}.get(p[0], p[0])     # orijinal veride bir yazım hatası
    emotion = {"ps": "surprise"}.get(p[2].lower(), p[2].lower())
    return speaker, emotion, f"{speaker}_{p[1]}_{emotion}"


def _subesco(name, _):
    # F_01_OISHI_S_10_ANGRY_1.wav: cinsiyet_no_isim_S_cümle_DUYGU_tekrar
    p = name[:-4].split("_")
    if len(p) != 7:
        raise Skip("SUBESCO şemasına uymuyor")
    return f"{p[0]}_{p[1]}", p[5].lower(), name[:-4]


def _savee(name, _):
    # DC_a01.wav: konuşmacı_duyguNo
    m = re.fullmatch(r"([A-Z]{2})_(a|d|f|h|n|sa|su)(\d+)\.wav", name)
    if not m:
        raise Skip("SAVEE şemasına uymuyor")
    emotions = {"a": "angry", "d": "disgust", "f": "fear", "h": "happy",
                "n": "neutral", "sa": "sad", "su": "surprise"}
    return m.group(1), emotions[m.group(2)], name[:-4]


def _jl_corpus(name, _):
    # female1_angry_10a_1.wav: konuşmacı_duygu_cümle_tekrar
    p = name[:-4].split("_")
    if len(p) != 4:
        raise Skip("JL-Corpus şemasına uymuyor")
    return p[0], p[1], name[:-4]


def _esc50(name, _):
    # 1-100032-A-0.wav: fold-freesoundID-parça-sınıf. Aynı freesound kaydından
    # kesilen parçalar aynı gruba düşer (konuşmacı karşılığı).
    m = re.fullmatch(r"(\d)-(\d+)-([A-Z])-(\d+)\.wav", name)
    if not m:
        raise Skip("ESC-50 şemasına uymuyor")
    target = int(m.group(4))
    if target in ESC50_HUMAN_VOCAL:
        raise Skip(f"insan sesi içeren ESC-50 sınıfı ({ESC50_HUMAN_VOCAL[target]})")
    return f"clip{m.group(2)}", f"esc{target}", name[:-4]


def _vivae(name, _):
    # S04_pain_moderate_10.wav: konuşmacı_duygu_şiddet_öğe. Ağrı için şiddet
    # etikete katılır (pain_low ... pain_peak), diğer duygular şiddetsiz.
    m = re.fullmatch(r"(S\d\d)_([a-z]+)_(low|moderate|strong|peak)_(\d+)\.wav", name)
    if not m:
        raise Skip("VIVAE şemasına uymuyor")
    speaker, emotion, intensity = m.group(1), m.group(2), m.group(3)
    if emotion == "pain":
        emotion = f"pain_{intensity}"
    return speaker, emotion, name[:-4]


def _field(name, path):
    # Elle eklenen gerçek kayıtlar: field/<etiket>/<konuşmacı>/<dosya>.wav
    # (ör. CHAINS / wTIMIT fısıltıları veya gönüllülerden alınan kayıtlar).
    parts = os.path.normpath(path).split(os.sep)
    label, speaker = parts[-3], parts[-2]
    if label not in EMERGENCY_CLASSES_V2:
        raise Skip(f"field/ altında bilinmeyen etiket klasörü ({label})")
    return speaker, label, f"{label}/{speaker}/{name[:-4]}"


@dataclass(frozen=True)
class DatasetSpec:
    dir: str        # DATA_DIR'e göre
    role: str       # "human" | "non_human"
    license: str
    parse: object   # (dosya_adı, yol) -> (konuşmacı, duygu, kayıt_kimliği)
    # True: bütün kayıtlar test bölmesine düşer; eğitimde hiç kullanılmaz.
    # Gerçek fısıltı/inleme verisi az olduğu için yalnızca dış doğrulama içindir.
    eval_only: bool = False


DATASETS = {
    "ravdess":   DatasetSpec("human/ravdess",   "human",     "CC BY-NC-SA 4.0",           _ravdess),
    "berlin":    DatasetSpec("human/berlin",    "human",     "free use with attribution", _berlin),
    "tess":      DatasetSpec("human/tess",      "human",     "CC BY-NC 4.0",              _tess),
    "subesco":   DatasetSpec("human/subesco",   "human",     "CC BY 4.0",                 _subesco),
    "savee":     DatasetSpec("human/savee",     "human",     "research only",             _savee),
    "jl_corpus": DatasetSpec("human/jl_corpus", "human",     "CC0",                       _jl_corpus),
    "esc50":     DatasetSpec("non-human/esc50", "non_human", "CC BY-NC 3.0",              _esc50),
    "vivae":     DatasetSpec("human/vivae",     "human",     "CC BY-NC 4.0",              _vivae, eval_only=True),
    "field":     DatasetSpec("human/field",     "human",     "kayda göre değişir",        _field, eval_only=True),
}

EVAL_ONLY_DATASETS = {k for k, v in DATASETS.items() if v.eval_only}


def dataset_dir(name):
    return os.path.join(DATA_DIR, *DATASETS[name].dir.split("/"))


# =========================
# MANIFEST
# =========================
@dataclass
class Record:
    path: str               # ROOT'a göre, "/" ayraçlı
    dataset: str
    license: str
    role: str               # human | non_human
    speaker: str            # veri seti içinde; bölme bu gruba göre yapılır
    recording_id: str       # veri seti içinde tekil
    emotion: str
    emergency_class: str    # EMERGENCY_CLASSES'tan biri veya ""
    split: str = ""         # train | test
    transform: str = "original"
    emergency_class_v2: str = ""   # EMERGENCY_CLASSES_V2'den biri veya ""

    @property
    def group(self):
        return f"{self.dataset}/{self.speaker}"


MANIFEST_FIELDS = [f.name for f in fields(Record)]


def scan_dataset(name):
    """Bir veri setinin wav dosyalarını Record listesine çevirir.
    Dönüş: (kayıtlar, {atlanma nedeni: sayı})"""
    spec = DATASETS[name]
    base = dataset_dir(name)
    records, skipped, seen = [], {}, set()
    for dirpath, _, files in sorted(os.walk(base)):
        for f in sorted(files):
            if not f.lower().endswith(".wav"):
                continue
            path = os.path.join(dirpath, f)
            try:
                speaker, emotion, rec_id = spec.parse(f, path)
            except Skip as e:
                skipped[str(e)] = skipped.get(str(e), 0) + 1
                continue
            if rec_id in seen:
                skipped["aynı kaydın kopyası"] = skipped.get("aynı kaydın kopyası", 0) + 1
                continue
            seen.add(rec_id)
            records.append(Record(
                path=os.path.relpath(path, ROOT).replace(os.sep, "/"),
                dataset=name,
                license=spec.license,
                role=spec.role,
                speaker=speaker,
                recording_id=rec_id,
                emotion=emotion,
                # v1 modelleri eval_only veri setlerini hiç görmez
                emergency_class=EMOTION_TO_CLASS.get(emotion, "")
                if spec.role == "human" and not spec.eval_only else "",
                emergency_class_v2=EMOTION_TO_CLASS_V2.get(emotion, "") if spec.role == "human" else "",
            ))
    return records, skipped


def assign_splits(records, test_fraction=TEST_SPEAKER_FRACTION, seed=SPLIT_SEED):
    """Konuşmacı bazlı bölme: bir konuşmacının bütün kayıtları aynı tarafa düşer.
    Her veri setinden konuşmacıların ~%20'si teste ayrılır (en az 1, en az 1 de
    eğitimde kalır). ESC-50 kendi resmi 5. fold'unu test olarak kullanır.
    eval_only veri setlerinin tamamı teste düşer."""
    by_dataset = {}
    for r in records:
        by_dataset.setdefault(r.dataset, set()).add(r.speaker)

    test_groups = set()
    for ds, speakers in sorted(by_dataset.items()):
        if ds == "esc50" or ds in EVAL_ONLY_DATASETS:
            continue
        speakers = sorted(speakers)
        if len(speakers) < 2:
            continue
        n_test = min(len(speakers) - 1, max(1, round(len(speakers) * test_fraction)))
        rng = random.Random(f"{seed}-{ds}")
        test_groups.update(f"{ds}/{s}" for s in rng.sample(speakers, n_test))

    for r in records:
        if r.dataset == "esc50":
            r.split = "test" if r.path.rsplit("/", 1)[-1].startswith("5-") else "train"
        elif r.dataset in EVAL_ONLY_DATASETS:
            r.split = "test"
        else:
            r.split = "test" if r.group in test_groups else "train"
    return records


def write_manifest(records, path=MANIFEST_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for r in records:
            w.writerow(asdict(r))


def read_manifest(path=MANIFEST_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Manifest bulunamadı: {os.path.relpath(path, ROOT)}\n"
            "Önce veriyi indirip manifest oluşturun:\n"
            "  python scripts/download_data.py\n"
            "  python scripts/build_manifest.py")
    with open(path, newline="", encoding="utf-8") as f:
        return [Record(**row) for row in csv.DictReader(f)]


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
