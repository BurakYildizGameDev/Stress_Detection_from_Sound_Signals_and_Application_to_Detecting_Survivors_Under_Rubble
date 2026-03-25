import os, json
ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(ROOT, 'logs')

def append_event(e): pass
def read_events(): return []

def write_status(s): pass
def read_status(): return None

def listener_state(s): return 'stopped'

class AlarmTracker:
    def __init__(self, consecutive=3, cooldown_sec=5):
        self.consecutive = consecutive
        self.cooldown_sec = cooldown_sec
