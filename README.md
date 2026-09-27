# Stress Detection from Sound Signals — Detecting Survivors Under Rubble

A two-stage acoustic pipeline that listens to a microphone in 1-second windows and
raises an alarm when it hears a stressed or panicked human voice. The target use
case is search and rescue after an earthquake.

```
audio (22.05 kHz, 1 s window)
  │
  ├─ RMS < 0.01 ─────────────────────────────► silence
  │
  ├─ Stage 1: human detector (RandomForest, 28 features)
  │     P(human) < 0.20 ─────────────────────► no_human
  │
  └─ Stage 2: emergency classifier (RandomForest, 15 features @ 16 kHz)
        normal / stress / scream / panic
        │
        └─ live_listener: 3 consecutive emergency windows ► alarm + alerts_log.json
                                                             (shown in app.py dashboard)
```

Feature definitions live in one place, [`features.py`](features.py), and are used
by both the training scripts and the live pipeline.

| Model | Features |
|---|---|
| Human detector | 13 MFCC mean, 13 MFCC std, ZCR mean, RMS mean |
| Emergency classifier | 13 MFCC mean, RMS mean, spectral centroid mean (resampled to 16 kHz) |

## Quick start

```bash
pip install -r requirements.txt

python audio_input.py "deneme sesleri/peopleTalk.mp3"   # analyse a file
python live_listener.py                                  # listen to the microphone
streamlit run app.py                                     # dashboard for alerts_log.json
python -m pytest tests                                   # tests
```

The trained models in `models/` are stored with Git LFS (`git lfs pull`). They were
saved with scikit-learn 1.8.0, which is why that version is pinned.

## Data

The datasets are **not** stored in this repository. Their licenses (several are
non-commercial or no-derivatives) do not allow redistribution. To download them
from their original sources:

```bash
python scripts/download_data.py --list      # sources and licenses
python scripts/download_data.py             # RAVDESS, EMO-DB, SUBESCO, TESS, ESC-50
```

SAVEE and JL-Corpus require registration or Kaggle, so the script prints manual
instructions for them.

| Dataset | Language | Used as |
|---|---|---|
| RAVDESS (song subset) | English | human |
| Berlin EMO-DB | German | human |
| TESS | English | human |
| SAVEE | English | human |
| JL-Corpus | English (NZ) | human |
| SUBESCO | Bangla | human |
| ESC-50 | — | non-human |

Training chain:

```
Emergency model: prepare_emergency_dataset.py → extract_emergency_features.py → train_emergency_classifier.py
Human detector:  sample_dataset.py → extract_human_features_fast.py → train_human_detector_mid.py
```

## Evaluation (honest numbers)

The emergency classifier on its own dataset (1542 clips: 500 normal / 500 stress /
42 scream / 500 panic):

| Evaluation | Accuracy | Macro-F1 |
|---|---|---|
| Random split (as in `train_emergency_classifier.py`) | 0.864 | 0.706 |
| Augmented copies kept on the same side of the split | 0.808 | 0.613 |
| Speaker-independent 5-fold | 0.630 ± 0.11 | 0.572 |
| Unseen corpus (SUBESCO) | 0.439 | 0.302 |

The random-split figure is inflated because augmented copies of a clip, and the
same speakers, appear in both train and test. The speaker-independent figure is
the realistic one.

## Known limitations

- **The `scream` class does not contain screams.** It is built by matching the word
  "shout" in file names. In TESS, "shout" is one of the spoken words, so the class
  contains 14 recordings of the word "shout" (42 after augmentation).
- **False alarms on non-speech sounds.** On `deneme sesleri/bird.mp3`, all 5 windows
  are classified as emergency. On `deneme woman scream.mp3`, most windows come out
  as `no_human`.
- **Train/serve mismatch in clip length.** The models were trained on 2–3 s clips
  but run on 1 s windows.
- **No Turkish speech in the training data.**

Planned fixes: real scream data (e.g. the AudioSet "Screaming" class), all ESC-50
classes as negatives, speaker-independent training, and knock/tap detection
(onset analysis) as a rubble-specific signal.

## License

Code: MIT (see [LICENSE](LICENSE)). The datasets keep their own licenses, listed
by `scripts/download_data.py --list`.
