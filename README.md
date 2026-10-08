# Cough and Help Detection Backend

Real-time audio backend that detects **coughs** and spoken **"help"** requests, then streams results to Firebase for clinician-facing monitoring.

This repository implements the audio pipeline from:

> Yeaminur Rahman, Rezwana Mahfuza, Md. Abdul Hai, Rafsan Shartaj Uddin, Muhammad Iqbal Hossain.  
> **Real-Time Patient Ailment Monitoring Framework Collaborating Enhanced CNN Architectures.**  
> *2021 11th IEEE International Conference on Intelligent Data Acquisition and Advanced Computing Systems (IDAACS)*, pp. 1016–1021.  
> [IEEE Xplore](https://ieeexplore.ieee.org/document/9660938)

Place a local copy of the paper in [`docs/`](docs/) if you want it offline (recommended filename: `IDAACS_2021_Patient_Ailment_Monitoring.pdf`).

---

## Overview

The system continuously records short microphone clips, classifies cough vs non-cough with a CNN on mel spectrograms, tracks consecutive coughs within a time window, and listens for the word “help” when no cough is detected. Events are written to Firebase Realtime Database so a companion app or dashboard can show live status.

| Capability | Behavior |
|---|---|
| Cough detection | Mel spectrogram → VGG19 classifier (`model_VGG19_cough_acc88.75_best_50epoch.h5`) |
| Consecutive coughs | Counts coughs that occur within a 10-second window; stores count + timestamp |
| Help detection | Google Speech-to-Text on non-cough clips; flags if transcript contains `"help"` |
| Live sync | Updates Firebase keys used by a monitoring frontend |

In the paper, VGG19 outperformed DenseNet for cough identification from audio; this notebook uses that VGG19 model (~88.75% reported accuracy after 50 epochs).

---

## Architecture

```
Microphone (PyAudio)
        │
        ▼
  2s WAV clip (output.wav)
        │
        ├──────────────────────────────┐
        ▼                              ▼
 Mel spectrogram (librosa)     If predicted non-cough:
        │                      SpeechRecognition → Google STT
        ▼                              │
 VGG19 CNN (Keras .h5)                 ▼
        │                      "help" in transcript?
        ▼                              │
 Cough / Didn’t cough                  │
        │                              │
        └──────────┬───────────────────┘
                   ▼
         Firebase Realtime Database
```

Two threads run in parallel:

1. **`record()`** — captures 2-second mono audio at 44.1 kHz into `output.wav`
2. **`process()`** — loads the clip, builds a mel spectrogram image, runs the model, updates consecutive-cough logic and Firebase

---

## Firebase data written

| Path | Meaning |
|---|---|
| `gi` | `1` = cough detected, `0` = no cough |
| `last_cough` | Timestamp of last cough (`HH:MM:SS`) |
| `consecutive cough/cough` | Number of coughs in the current streak |
| `consecutive cough/time` | Timestamp associated with the streak |
| `last_help` | Timestamp when “help” was recognized |

---

## Requirements

- Python 3.7+ (TensorFlow 2.x era recommended for the saved `.h5` workflow)
- Working microphone
- Firebase Realtime Database project
- Google Speech Recognition access (used via `SpeechRecognition`)

### Python packages

```text
tensorflow
keras
librosa
numpy
matplotlib
pyaudio
SpeechRecognition
pyrebase
```

Install example:

```bash
pip install tensorflow librosa numpy matplotlib pyaudio SpeechRecognition pyrebase4
```

> **Note:** `PyAudio` often needs a platform-specific wheel or PortAudio. On Windows, use a matching wheel from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio) if `pip install pyaudio` fails.

---

## Setup

1. **Clone the repo** and open `Cough Detection.ipynb` in Jupyter / VS Code / Cursor.

2. **Add the trained model** next to the notebook (or update the path in the first cell):

   ```text
   model_VGG19_cough_acc88.75_best_50epoch.h5
   ```

3. **Configure Firebase** in the config cell:

   ```python
   config = {
     "apiKey": "your API key",
     "authDomain": "your_app.firebaseapp.com",
     "databaseURL": "https://your_app.firebaseio.com/",
     "storageBucket": "gs://your_app.appspot.com",
   }
   ```

4. **Set the spectrogram output path** in `process()` — the notebook currently saves to a Windows Pictures folder and loads images via `ImageDataGenerator.flow_from_directory`. Point both the `fig.savefig(...)` path and `new_dir` to a folder that contains a class subfolder expected by Keras (e.g. `.../cough/cough15.png`).

5. **Run all cells**, then start the recorder/processor threads (last non-empty cell). Keep the kernel alive while monitoring.

---

## How cough prediction works

1. Load `output.wav` with librosa  
2. Compute mel spectrogram → convert to dB  
3. Save as a PNG (axes off)  
4. Feed the image through `ImageDataGenerator` at `150×150`, rescaled by `1/255`  
5. `model.predict_classes(...)`  
   - `1` → cough  
   - `0` → no cough (then try speech / “help”)

Consecutive coughs: if another cough arrives within `timedelta(0, 10)` of the streak start, the counter increments and both count and time are pushed to Firebase; otherwise the streak resets.

---

## Project layout

```text
Cough-and-Help-Detection-backend/
├── Cough Detection.ipynb          # Real-time record + classify + Firebase sync
├── README.md
├── docs/                          # Optional: paper PDF for offline reading
└── model_VGG19_cough_acc88.75_best_50epoch.h5   # Not in git; add locally
```

---

## Paper context

The full framework in the paper also covers IoT vitals (temperature, heart rate, SpO₂) and a web UI for clinicians. **This repo is the audio backend slice**: cough classification (VGG19 vs DenseNet comparison favoring VGG19), consecutive-cough tracking, and Google Speech-to-Text for requesting nearby assistance — with results published to a centralized database for real-time monitoring.

---

## Limitations / next steps

- Model weights and Firebase credentials are not committed; supply them locally.
- Spectrogram paths are hardcoded for a specific Windows user layout — update before running elsewhere.
- `predict_classes` is legacy Keras API; newer TensorFlow versions may need `np.argmax(model.predict(...), axis=-1)`.
- Matplotlib’s `backend_qt4agg` import is deprecated; switch to a current backend if you hit import errors.
- For production, prefer a packaged script/service over a long-running notebook, and secure Firebase rules.

---

## Citation

```bibtex
@inproceedings{RahmanMHUH21,
  author    = {Yeaminur Rahman and Rezwana Mahfuza and Md. Abdul Hai
               and Rafsan Shartaj Uddin and Muhammad Iqbal Hossain},
  title     = {Real-Time Patient Ailment Monitoring Framework Collaborating
               Enhanced {CNN} Architectures},
  booktitle = {2021 11th IEEE International Conference on Intelligent Data
               Acquisition and Advanced Computing Systems: Technology and
               Applications (IDAACS)},
  pages     = {1016--1021},
  year      = {2021},
  publisher = {IEEE}
}
```

---

## License

Add a license file if you intend to distribute this project publicly.
