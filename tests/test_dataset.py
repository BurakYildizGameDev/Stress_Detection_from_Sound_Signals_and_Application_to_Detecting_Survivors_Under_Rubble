import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataset
from dataset import (DATASETS, EMERGENCY_CLASSES, EMOTION_TO_CLASS, Record, Skip,
                     assign_splits, scan_dataset)


def parse(ds, name):
    return DATASETS[ds].parse(name, name)


@pytest.mark.parametrize("ds,name,speaker,emotion", [
    ("ravdess", "03-02-06-01-02-01-12.wav", "actor12", "fear"),
    ("ravdess", "03-02-02-01-01-01-01.wav", "actor01", "calm"),
    ("berlin", "03a01Fa.wav", "03", "happy"),
    ("berlin", "16b10Wb.wav", "16", "angry"),
    ("berlin", "11a02Ac.wav", "11", "fear"),
    ("tess", "OAF_back_angry.wav", "OAF", "angry"),
    ("tess", "YAF_shout_fear.wav", "YAF", "fear"),
    ("tess", "OA_bite_neutral.wav", "OAF", "neutral"),
    ("tess", "YAF_food_ps.wav", "YAF", "surprise"),
    ("subesco", "F_01_OISHI_S_10_ANGRY_1.wav", "F_01", "angry"),
    ("savee", "DC_sa03.wav", "DC", "sad"),
    ("savee", "KL_f12.wav", "KL", "fear"),
    ("jl_corpus", "female1_anxious_10a_1.wav", "female1", "anxious"),
    ("esc50", "1-100032-A-0.wav", "clip100032", "esc0"),
])
def test_parsers(ds, name, speaker, emotion):
    s, e, _ = parse(ds, name)
    assert (s, e) == (speaker, emotion)


def test_esc50_human_vocal_classes_are_skipped():
    with pytest.raises(Skip):
        parse("esc50", "1-17124-A-20.wav")      # crying_baby


def test_unknown_names_are_skipped():
    with pytest.raises(Skip):
        parse("berlin", "readme.wav")


def test_each_emotion_maps_to_one_known_class():
    assert set(EMOTION_TO_CLASS.values()) <= set(EMERGENCY_CLASSES)
    assert "scream" not in EMERGENCY_CLASSES
    # Eski anahtar kelime eşleşmesi "fearful"u hem stress hem panic'e koyuyordu
    assert EMOTION_TO_CLASS["fear"] == "panic"
    assert EMOTION_TO_CLASS["angry"] == "stress"


def test_tess_target_word_is_not_a_label(tmp_path, monkeypatch):
    make_wavs(tmp_path, "human/tess/OAF_angry", ["OAF_shout_angry.wav"])
    monkeypatch.setattr(dataset, "DATA_DIR", str(tmp_path))
    recs, _ = scan_dataset("tess")
    assert [r.emergency_class for r in recs] == ["stress"]


def make_wavs(root, sub, names):
    d = root.joinpath(*sub.split("/"))
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        (d / n).write_bytes(b"")


def test_scan_drops_duplicate_copies(tmp_path, monkeypatch):
    make_wavs(tmp_path, "human/tess/OAF_neutral", ["OAF_bite_neutral.wav", "OAF_back_neutral.wav"])
    make_wavs(tmp_path, "human/tess/copy/OAF_neutral", ["OA_bite_neutral.wav", "OAF_back_neutral.wav"])
    monkeypatch.setattr(dataset, "DATA_DIR", str(tmp_path))
    recs, skipped = scan_dataset("tess")
    assert sorted(r.recording_id for r in recs) == ["OAF_back_neutral", "OAF_bite_neutral"]
    assert skipped == {"aynı kaydın kopyası": 2}


def rec(ds, speaker, path="x.wav"):
    return Record(path=path, dataset=ds, license="", role="human", speaker=speaker,
                  recording_id=f"{speaker}-{path}", emotion="neutral", emergency_class="normal")


def test_split_is_speaker_disjoint():
    records = [rec(ds, f"s{i}", f"{j}.wav") for ds in ("ravdess", "subesco")
               for i in range(10) for j in range(5)]
    assign_splits(records)
    for ds in ("ravdess", "subesco"):
        train = {r.speaker for r in records if r.dataset == ds and r.split == "train"}
        test = {r.speaker for r in records if r.dataset == ds and r.split == "test"}
        assert train and test and not (train & test)
        assert len(test) == 2


def test_split_is_deterministic():
    a = assign_splits([rec("ravdess", f"s{i}") for i in range(10)])
    b = assign_splits([rec("ravdess", f"s{i}") for i in range(10)])
    assert [r.split for r in a] == [r.split for r in b]


def test_split_keeps_one_speaker_for_training():
    records = assign_splits([rec("tess", "OAF"), rec("tess", "YAF")])
    assert sorted(r.split for r in records) == ["test", "train"]


def test_esc50_uses_fold_5_as_test():
    records = [rec("esc50", "clip1", "data/non-human/esc50/5-1-A-1.wav"),
               rec("esc50", "clip2", "data/non-human/esc50/1-2-A-1.wav")]
    assign_splits(records)
    assert [r.split for r in records] == ["test", "train"]
