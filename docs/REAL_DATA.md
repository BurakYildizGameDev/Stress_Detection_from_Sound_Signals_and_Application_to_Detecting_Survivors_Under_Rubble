# Real whisper / moan data

The v2 classes `whisper` and `moan` are trained **only on synthetic audio**
(`whisper_converter.py` turns acted speech into LPC-whisper or low-passed, pitch-shifted
"moans"). A score on synthetic test audio partly measures the converter, not the
real-world skill. So every real recording we can get is used as an **external test
set** and never for training (`eval_only=True` in `dataset.DATASETS`): all of its
speakers go to the test split and `extract_features_v2.py` never synthesises from it.

`scripts/train_v2.py` reports these under `external` in `reports/<task>_v2.json`.

## In use

| Dataset | What it gives | How to get it | License |
|---|---|---|---|
| **VIVAE** (Holz et al., 2022) | 1085 non-verbal vocalisations, 11 speakers. `pain_low` + `pain_moderate` (89 files) → `moan`; `fear` (176) → `panic`. `pain_strong/peak` (screams), anger, achievement, pleasure and surprise get no emergency class but count as `human` for the human detector. | `python scripts/download_data.py vivae` ([Zenodo 4066235](https://doi.org/10.5281/zenodo.4066235)) | CC BY-NC 4.0 |

VIVAE is a set of acted studio recordings, not recordings of injured people. It is closer
to a real moan than the converter output, but still not field data.

## Not yet available: real whispers

There is **no real whisper data in the repo yet**, so whisper recall is only known on
synthetic audio. Candidates, in order of preference:

| Source | Notes |
|---|---|
| **CHAINS** corpus (University College Dublin) | 36 speakers, a whispered reading condition next to normal reading. Listed as free for research at <https://chains.ucd.ie/>; the server was unreachable when this was written (2026-09), try again or ask the authors. |
| **wTIMIT** | 48 speakers (US + Singaporean English), parallel normal/whispered TIMIT sentences, ~26 h each. Access through the authors. |
| **AISHELL6-Whisper** | 167 speakers, Mandarin, parallel whisper/normal. Gated on Hugging Face (academic request), CC BY-NC-SA 4.0. |
| **Your own recordings** | See below. This is also the only way to get Turkish whispers. |

## Adding recordings (`data/human/field/`)

Put WAV files in a `field/<label>/<speaker>/` folder:

```
data/human/field/
  whisper/spk01/utt001.wav
  whisper/spk02/utt001.wav
  moan/spk01/take01.wav
  normal/spk01/utt001.wav
```

`<label>` is one of `whisper`, `moan`, `normal`, `stress`, `panic`. Then:

```
python scripts/build_manifest.py
python scripts/extract_features_v2.py --task all
python scripts/train_v2.py --task all
```

Existing splits do not change when datasets are added (splits are seeded per dataset).

### Recording protocol for volunteers

- Get written consent that states the purpose and that the audio stays in a research
  dataset. Do not record people who are actually injured or in distress.
- At least 10 speakers, mixed gender and age; each reads the same 10 short Turkish
  phrases (e.g. "buradayım", "yardım edin", "sesimi duyan var mı") once normally and
  once whispered, plus 3–5 moans / groans of about 2 s.
- Record with the target microphone (INMP441 on the ESP32-S3) as well as a phone, at
  16 kHz or higher, 20–50 cm away. Note the distance and room for each take.
- If possible, repeat a few takes from inside a void (under a table covered with
  bricks or rubble, a collapsed-building training site of AFAD/AKUT) and measure that
  space's impulse response with `sweep_rir.py`. Those measured IRs are the best way to
  calibrate `rubble_acoustics.py`.

## Candidates not yet checked

- ECHO corpus, cross-cultural non-verbal vocalisations ([Zenodo 22754938](https://zenodo.org/records/22754938))
- Vocalisation Challenge Corpus VOC-C ([Zenodo 6457541](https://zenodo.org/records/6457541))

Check the license, labels (is there a pain / moan class?) and speaker IDs before adding
a parser to `dataset.py`.
