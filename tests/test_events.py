import pytest
from events import AlarmTracker
from datetime import datetime
def test_tracker():
    t = AlarmTracker(consecutive=3)
    assert t.windows == 0
