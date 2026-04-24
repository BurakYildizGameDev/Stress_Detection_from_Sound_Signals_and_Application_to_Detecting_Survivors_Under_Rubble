# Evaluation plan

This system is an **operator aid**: it points a listener's attention at moments
that may contain a person in distress. Until it has passed the field tests in
section 3, it must not be described or used as a tool that decides where to dig or
whether someone is alive.

Two error types have very different costs under rubble, so every protocol reports
them separately, next to per-class precision/recall and a confusion matrix:

| Metric | Emergency classifier | Human detector |
|---|---|---|
| **False-alarm rate** | `normal` recordings predicted as `stress`/`panic` | `non_human` recordings predicted as `human` |
| **Miss rate** | `stress`/`panic` recordings predicted as `normal` | `human` recordings predicted as `non_human` |

A missed person is worse than a false alarm, but a system that alarms constantly
gets ignored, so both numbers matter.

---

## 1. Automated protocols (`scripts/evaluate.py`)

All three read `features/<task>.npz` and the saved models. They write JSON to
`reports/`.

| Protocol | Question it answers | How |
|---|---|---|
| `held-out` | How well does the model do on **speakers it has never heard**? | Speaker-disjoint split from `dataset.assign_splits`: ~20% of each corpus's speakers are held out; ESC-50 uses its official fold 5. Only original recordings are tested, never augmented copies. **This is the headline number.** |
| `cross-dataset` | What happens when the **language / recording setup** changes? | Leave-one-corpus-out. For the human detector, ESC-50 fold 5 is added to every test fold so false alarms can be measured. |
| `robustness` | How fast does performance fall apart in **rubble-like conditions**? | The held-out recordings are re-processed through `augment.CONDITIONS`: white noise at 20/10/0 dB SNR, −20 dB attenuation (distance), 1 kHz and 400 Hz low-pass (concrete/debris), and a combined `rubble_sim`. |

The random 80/20 split that the original scripts used is no longer reported: it put
augmented copies of the same recording, and the same speakers, on both sides.

### What the synthetic conditions do not cover

White noise and a Butterworth low-pass filter are crude stand-ins. Real rubble adds
strong reverberation, frequency-dependent absorption that varies by material, and
structured noise from machinery, generators and other rescuers. A good
`robustness` score is necessary but not sufficient.

---

## 2. Data requirements before trusting any number

- **Speaker-independent splits only.** Enforced by the manifest.
- **One label per recording**, taken from each corpus's own annotation scheme
  (`dataset.EMOTION_TO_CLASS`), not from substrings in file names.
- **No `scream` class** until real screams exist in the data. The previous 42
  "scream" clips were TESS speakers saying the word *shout*.
- **Acted speech is a proxy.** All emotion corpora are actors reading sentences. Real
  distress vocalisations (calling for help, moaning, crying, knocking) are different
  signals, and collecting them is the next data milestone.

---

## 3. Field test plan (not yet done)

Each condition is a controlled recording session with volunteers producing scripted
vocalisations (calm speech, calling for help, shouting, crying/moaning), plus
recordings with no person present to measure false alarms.

| Factor | Levels |
|---|---|
| **Device** | laptop microphone, USB measurement microphone, contact/geophone sensor, phone |
| **Distance** | 0.5 m, 2 m, 5 m, 10 m from source to sensor |
| **Barrier** | open air, one interior wall, concrete slab, loose debris pile (training-ground rubble) |
| **Background noise** | quiet, generator/excavator at typical site distance, other rescuers talking |
| **Speaker** | adult male, adult female, older adult; at least 10 people who are in no training set |

Report for every cell of the table:

- miss rate per vocalisation type, measured per **episode** (a whole call for help), not
  per 1-second window
- false alarms per hour of recording with no person present
- time from the start of a vocalisation to the operator alarm

Acceptance criteria, to be agreed with rescue practitioners before the tests, should
set a maximum miss rate at a given distance/barrier and a maximum false-alarm rate
per hour. The system is not ready for deployment claims until it meets them on
speakers and sites it has never seen.
