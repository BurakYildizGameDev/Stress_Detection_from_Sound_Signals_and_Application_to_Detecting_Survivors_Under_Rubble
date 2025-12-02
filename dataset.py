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
