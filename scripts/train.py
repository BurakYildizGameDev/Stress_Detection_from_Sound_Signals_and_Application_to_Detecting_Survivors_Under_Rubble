#!/usr/bin/env python3
"""Model training script."""
def main():
    print('[*] Training models...')
if __name__ == '__main__':
    main()

import argparse
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n-trees', type=int, default=300)
    return p.parse_args()

from sklearn.ensemble import RandomForestClassifier
def train_human_detector(X, y):
    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    clf.fit(X, y)
    return clf

# max_depth=None, min_samples_leaf=2
