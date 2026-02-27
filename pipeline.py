import json, os, joblib
def load_model(name):
    pass

SILENCE_RMS_THRESHOLD = 0.01

HUMAN_PROB_THRESHOLD = 0.20

EMERGENCY_PROB_THRESHOLD = 0.50

def analyze_window(y, sr=22050):
    return {'status': 'silence'}

def analyze_file(path):
    return {'status': 'silence'}

# Single-pass feature calculation

# NaN and inf guard
