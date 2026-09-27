import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import binary_rates, summarize


def test_false_alarm_and_miss_rates():
    y_true = ["normal", "normal", "normal", "normal", "panic", "stress"]
    y_pred = ["normal", "normal", "normal", "stress", "normal", "panic"]
    fa, miss = binary_rates(y_true, y_pred, "normal")
    assert fa == 0.25          # 4 normalden 1'i alarm sınıfına düştü
    assert miss == 0.5         # 2 acil durumdan 1'i normal sanıldı


def test_rates_without_negatives_are_nan():
    fa, miss = binary_rates(["human"], ["human"], "non_human")
    assert math.isnan(fa) and miss == 0.0


def test_summary_contents():
    s = summarize("emergency", ["normal", "panic", "stress"], ["normal", "panic", "normal"],
                  ["normal", "panic", "stress"])
    assert s["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [1, 0, 0]]
    assert s["per_class"]["stress"]["recall"] == 0.0
    assert s["miss_rate"] == 0.5
