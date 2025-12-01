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
