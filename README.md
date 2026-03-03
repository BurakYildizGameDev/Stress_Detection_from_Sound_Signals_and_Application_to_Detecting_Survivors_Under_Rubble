# 🚨 Acoustic Survivor Detection Under Rubble

**Detecting trapped survivors after an earthquake by listening for stressed and panicked human voices.**

Research prototype for acoustic survivor detection.

## Acoustic Features

MFCCs, Spectral Centroid, and ZCR are extracted to distinguish human vocalizations from debris grinding noises.

## Data

Instructions for downloading ESC-50 and speech emotional corpora.

### Speaker-Disjoint Splits

To prevent data leakage, speakers in test sets never appear in training.

### Stage 1 & Stage 2 Model Specifications

- Stage 1: RandomForest (300 trees, 28 features)
- Stage 2: RandomForest (300 trees, 15 features)

### Models and Checkpoints

Models are serialized with json specifications.
