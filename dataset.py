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
