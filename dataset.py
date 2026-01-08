import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, 'data')
FEATURES_DIR = os.path.join(ROOT, 'features')
MODELS_DIR = os.path.join(ROOT, 'models')
MANIFEST_PATH = os.path.join(DATA_DIR, 'manifest.csv')

EMERGENCY_CLASSES = ('normal', 'stress', 'panic')

EMOTION_TO_CLASS = {
    'neutral': 'normal',
    'calm': 'normal',
    'angry': 'stress',
    'fear': 'panic',
}

ESC50_HUMAN_VOCAL = {
    20: 'crying_baby', 21: 'sneezing', 23: 'breathing',
    24: 'coughing', 26: 'laughing', 28: 'snoring',
}

class Skip(Exception):
    """Dosya manifest'e alinmaz; mesaj atlanma nedenidir."""

def _ravdess(name, _):
    p = name[:-4].split('-')
    if len(p) != 7:
        raise Skip('RAVDESS semasina uymuyor')
    emotions = {'01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad', '05': 'angry', '06': 'fear', '07': 'disgust', '08': 'surprise'}
    return f'actor{p[6]}', emotions[p[2]], name[:-4]

import re
def _berlin(name, _):
    m = re.fullmatch(r'(\d\d)([a-z]\d\d)([WLEAFTN])([a-z])\.wav', name)
    if not m:
        raise Skip('EMO-DB semasina uymuyor')
    emotions = {'W': 'angry', 'L': 'boredom', 'E': 'disgust', 'A': 'fear', 'F': 'happy', 'T': 'sad', 'N': 'neutral'}
    return m.group(1), emotions[m.group(3)], name[:-4]

def _tess(name, _):
    p = name[:-4].split('_')
    if len(p) != 3:
        raise Skip('TESS semasina uymuyor')
    speaker = {'OA': 'OAF'}.get(p[0], p[0])
    emotion = {'ps': 'surprise'}.get(p[2].lower(), p[2].lower())
    return speaker, emotion, f'{speaker}_{p[1]}_{emotion}'

def _subesco(name, _):
    p = name[:-4].split('_')
    if len(p) != 7:
        raise Skip('SUBESCO semasina uymuyor')
    return f'{p[0]}_{p[1]}', p[5].lower(), name[:-4]

def _savee(name, _):
    m = re.fullmatch(r'([A-Z]{2})_(a|d|f|h|n|sa|su)(\d+)\.wav', name)
    if not m:
        raise Skip('SAVEE semasina uymuyor')
    emotions = {'a': 'angry', 'd': 'disgust', 'f': 'fear', 'h': 'happy', 'n': 'neutral', 'sa': 'sad', 'su': 'surprise'}
    return m.group(1), emotions[m.group(2)], name[:-4]

def _jl_corpus(name, _):
    p = name[:-4].split('_')
    if len(p) < 4:
        raise Skip('JL-Corpus semasina uymuyor')
    return f'{p[0]}_{p[1]}', p[2].lower(), name[:-4]

def _crema_d(name, _):
    p = name[:-4].split('_')
    if len(p) != 4:
        raise Skip('CREMA-D semasina uymuyor')
    emotions = {'NEU': 'neutral', 'HAP': 'happy', 'SAD': 'sad', 'ANG': 'angry', 'FEA': 'fear', 'DIS': 'disgust'}
    if p[1] not in emotions:
        raise Skip(f'Bilinmeyen duygu: {p[1]}')
    return p[0], emotions[p[1]], name[:-4]

def _esc50(name, _):
    p = name[:-4].split('-')
    if len(p) != 5:
        raise Skip('ESC-50 semasina uymuyor')
    target = int(p[4])
    if target in ESC50_HUMAN_VOCAL:
        raise Skip(f'ESC-50 insan vokali ({ESC50_HUMAN_VOCAL[target]})')
    return f'env_{p[0]}', f'esc_{target}', name[:-4]

from dataclasses import dataclass, asdict, fields
@dataclass
class ClipRecord:
    dataset: str
    rel_path: str
    speaker: str
    emotion: str
    role: str
    split: str
    clip_id: str

import librosa
import numpy as np
def trim_silence(y, top_db=30):
    non_silent = librosa.effects.split(y, top_db=top_db)
    if len(non_silent) == 0:
        return y
    return np.concatenate([y[start:end] for start, end in non_silent])

def slice_windows(y, sr, win_sec=1.0, hop_sec=1.0):
    win_len = int(win_sec * sr)
    hop_len = int(hop_sec * sr)
    windows = []
    for start in range(0, max(1, len(y) - win_len + 1), hop_len):
        w = y[start:start + win_len]
        if len(w) == win_len:
            windows.append(w)
    return windows if windows else [y]

import csv
def write_manifest(records, path=MANIFEST_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not records:
        return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[f.name for f in fields(records[0])])
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))

def group_by_speaker(records):
    from collections import defaultdict
    groups = defaultdict(list)
    for r in records:
        groups[r.speaker].append(r)
    return groups

def split_speakers(speakers, test_fraction=0.2, seed=42):
    rng = random.Random(seed)
    spk_list = sorted(list(speakers))
    rng.shuffle(spk_list)
    n_test = max(1, int(len(spk_list) * test_fraction))
    return set(spk_list[n_test:]), set(spk_list[:n_test])
