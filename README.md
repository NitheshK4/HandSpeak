<div align="center">

# 🤟 HandSpeak
### AI-Powered Real-Time Multicultural Sign Language Recognition

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-0F9D58?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

> **Real-time hand gesture recognition supporting American (ASL), Indian (ISL), British (BSL), and Universal sign languages — powered by a hybrid Rule-Based AI + Graph Neural Network + MLP + Random Forest system.**

---

![HandSpeak Demo Banner](https://img.shields.io/badge/Live%20Demo-localhost%3A8000-00f2fe?style=flat-square&logo=googlechrome&logoColor=white)

</div>

---

## ✨ Features

- 🎥 **Real-time webcam inference** — detects hand gestures live via browser
- ⚡ **Rule-Based Classifier** — instant, 90–95% accurate gesture detection using finger extension logic
- 🧠 **Graph Neural Network (GCN)** — custom hand skeleton GNN for deep structural understanding
- 📊 **Multi-Layer Perceptron (MLP)** — fast neural baseline for comparison
- 🌲 **Random Forest** — classic ML model for ensemble comparison
- 🌍 **Multicultural Support** — ASL, ISL, BSL, LSE (Spanish), Digits (0-9), and Universal gestures
- 🗣️ **Text-to-Speech** — speaks detected signs out loud
- ✍️ **Sentence Builder** — hold a pose to build full sentences word by word, with a **Backspace** button to edit mistakes
- ⌨️ **Hands-Free Keyboard Shortcuts** — control the interface easily using hotkeys (`[K]` camera, `[S]` TTS speech, `[C]` copy, `[B]` backspace, `[R]` reset)
- 💾 **Dataset Export Tool** — export the recorded custom gestures dataset in JSON format directly from the UI
- 📈 **Benchmarks & Historical Run Logs** — compare model accuracy, latency, and training time, and track accuracy history across training sessions
- 🖼️ **Interactive 3D Skeleton** — drag-to-orbit 3D hand visualizer
- 🎨 **Premium Dark UI** — glassmorphism design with smooth animations

---

## 🤟 Supported Signs

### ⚡ Universal (Works in ALL modes)
| Sign | Gesture |
|------|---------|
| 👋 Palm / Hello | Open hand, all 5 fingers extended |
| ✊ Fist / Yes | All fingers curled |
| 👍 Like / Thumbs Up | Thumb up, others curled |
| 👎 Dislike / Thumbs Down | Thumb down, others curled |
| ✌️ Peace / Victory | Index + Middle extended |
| ☝️ Point / One | Index finger only |
| 🤙 Call | Thumb + Pinky extended |
| 🤘 Rock | Index + Pinky extended |
| 🖕 MidFinger | Middle finger only |
| 🤏 OK | Thumb + Index circle, others open |
| 🤟 Love (ILY) | Thumb + Index + Pinky extended |
| 🔫 Gun | Thumb + Index, thumb up |
| 4️⃣ Four | 4 fingers extended (no thumb) |
| 3️⃣ Three | Index + Middle + Ring |
| 2️⃣ TwoUp | Index + Middle |
| 🤙 Pinkie | Pinky only |

### 🇺🇸 ASL (American Sign Language)
`Hello` · `Yes` · `No` · `Point` · `OK` · `Love`

### 🇮🇳 ISL (Indian Sign Language)
`Namaste` · `Alvida` · `TheekHai` · `PasandNahi` · `Vijay` · `Achaa`

### 🇬🇧 BSL (British Sign Language)
`Salute` · `Awesome` · `Dislike` · `Peace` · `Perfect`

### 🇪🇸 LSE (Spanish Sign Language)
`Hola` · `Si` · `No` · `Gracias`

### 🔢 Digits (Sign Language Digits)
`0` · `1` · `2` · `3` · `4` · `5` · `6` · `7` · `8` · `9`

---

## 🏗️ Architecture

```
HandSpeak/
├── Sign Language Dataset/   # Local folder containing photos of digits 0-9
├── backend/
│   ├── main.py              # FastAPI REST API server
│   ├── models.py            # GCN & MLP model definitions (PyTorch)
│   ├── train.py             # Training pipeline
│   ├── generate_data.py     # Synthetic landmark dataset generator
│   ├── process_kaggle_dataset.py # Kaggle/Local photo dataset processor & extractor
│   ├── rule_classifier.py   # ⚡ Rule-based finger extension classifier
│   ├── class_mapping.json   # Sign label ↔ index mapping
│   ├── gcn_model.pth        # Trained GNN weights
│   ├── mlp_model.pth        # Trained MLP weights
│   ├── rf_model.pkl         # Trained Random Forest
│   └── hand_landmarker.task # MediaPipe Tasks HandLandmarker model file
│
├── frontend/
│   ├── index.html           # Single-page application UI
│   ├── app.js               # Frontend logic + MediaPipe integration
│   └── styles.css           # Premium dark glassmorphism design
│
├── data/
│   ├── sign_dataset.json    # Augmented synthetic training dataset
│   └── training_metrics.json# Per-epoch accuracy & latency logs
│
└── README.md
```

---

## 🔬 How It Works

```
Webcam → MediaPipe Hands → 21 3D Landmarks
                                  │
          ┌───────────────────────┼──────────────────────┐
          ▼                       ▼                      ▼
   ⚡ Rule-Based          🧠 GCN / MLP             🌲 Random
   Classifier             (PyTorch)                 Forest
   (Finger States)        (2D features)             (sklearn)
          │                       │                      │
          └───────────────────────┼──────────────────────┘
                                  ▼
                      Best Prediction → UI Display
                                  │
                        🗣️ Text-to-Speech
                                  │
                        ✍️ Sentence Builder
```

### Landmark Preprocessing
- Raw 21-point 3D landmarks from MediaPipe
- Z-depth discarded (noisy in browser)
- Relative coordinates normalized by wrist→middle-MCP distance
- Features: `[x_rel, y_rel, dist_2d]` → shape `(21, 3)`

### Rule-Based Classifier (Primary)
- Checks if each finger's TIP is farther from wrist than its PIP joint
- Special thumb rule: checks vertical displacement
- Achieves **90–95% accuracy** without any training data
- Sub-millisecond inference latency

### GCN Model
- Custom Graph Convolutional Network on the hand skeleton graph
- 3 GCN layers: 3→32→64→128 features
- Global avg + max pooling readout
- Fully-connected classification head

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- A webcam
- Modern browser (Chrome recommended)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/NitheshK4/HandSpeak.git
cd HandSpeak

# 2. Install dependencies
pip install fastapi uvicorn torch torchvision scikit-learn numpy

# 3. Start the server
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

# 4. Open in browser
open http://localhost:8000
```

> **On first launch**, the server will automatically generate the synthetic dataset and train all three ML models. This takes ~2–3 minutes.

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` | REST API backend |
| `uvicorn` | ASGI server |
| `torch` | GCN & MLP deep learning |
| `scikit-learn` | Random Forest classifier |
| `numpy` | Landmark preprocessing |
| `mediapipe` | Real-time hand landmark detection (CDN) |
| `chart.js` | Training metrics visualization (CDN) |

---

## 🖥️ API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serves the HandSpeak frontend |
| `POST` | `/api/predict?culture=ASL` | Predict gesture from 21 landmarks |
| `GET` | `/api/metrics` | Get model training metrics |
| `POST` | `/api/train` | Trigger background model retraining |
| `POST` | `/api/save_gesture` | Save a new custom gesture sample |
| `GET` | `/api/export_dataset` | Export custom gesture dataset as JSON |
| `GET` | `/api/training_history` | Retrieve historical logs of validation metrics |

### Example Predict Request
```json
POST /api/predict?culture=ASL
{
  "landmarks": [
    [0.5, 0.9, 0.0],
    [0.42, 0.85, -0.01],
    ...
  ]
}
```

### Example Response
```json
{
  "rule": { "label": "Hello", "culture": "ASL", "confidence": 0.92, "latency_ms": 0.1 },
  "gnn":  { "label": "Hello", "culture": "ASL", "confidence": 0.88, "latency_ms": 2.3 },
  "mlp":  { "label": "Hello", "culture": "ASL", "confidence": 0.91, "latency_ms": 1.1 },
  "rf":   { "label": "Hello", "culture": "ASL", "confidence": 0.75, "latency_ms": 5.2 }
}
```

---

## 📊 Model Performance

| Model | Val Accuracy | Inference Latency |
|-------|-------------|-------------------|
| ⚡ Rule-Based | ~92% (avg) | <1 ms |
| 🧠 GNN (GCN) | ~85% | ~2–4 ms |
| 📊 MLP | ~88% | ~1–2 ms |
| 🌲 Random Forest | ~78% | ~3–6 ms |

> Performance measured on synthetic augmented dataset with 35 gesture classes across 4 cultures.

---

## 🎯 Usage Tips

1. **Use good lighting** — MediaPipe works best in well-lit rooms
2. **Keep hand centered** — position your hand in the middle of the camera frame
3. **Hold pose for ~1.5 seconds** — the sentence builder adds words after 10 stable frames
4. **Select your culture** — filter predictions to ASL, ISL, or BSL from the dropdown
5. **Record custom gestures** — use the Gesture Recorder tab to add new signs

---

## 🛠️ Customization

### Add a New Sign
1. Open `backend/generate_data.py`
2. Add a new template function `get_template_mygesture()`
3. Register it in `CULTURE_SIGNS` dictionary
4. Run `python3 backend/generate_data.py` to regenerate dataset
5. Run `python3 backend/train.py` to retrain models

### Add a New Culture
1. Add your culture key to `CULTURE_SIGNS` in `generate_data.py`
2. Add the `<option>` tag in `frontend/index.html` culture selector
3. Update `rule_classifier.py` to return culture-specific labels

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgements

- [MediaPipe](https://mediapipe.dev) — Real-time hand landmark detection
- [PyTorch](https://pytorch.org) — Deep learning framework
- [FastAPI](https://fastapi.tiangolo.com) — High-performance Python API
- [Chart.js](https://chartjs.org) — Beautiful charts for benchmarks

---

## 🆕 Recent Updates ( b6083ccc-5d47-43c1-8fca-cabc20566bb8 / June 2026 )

We implemented 6 key enhancements to the system:
1. **Sentence Builder Backspace**: Added a dedicated Backspace control and state management to let users delete mistakes in built sentences word by word.
2. **Accessibility Keyboard Shortcuts**: Introduced keyboard keys to toggle camera `[K]`, speech `[S]`, copy `[C]`, backspace `[B]`, and clear `[R]`.
3. **Dataset Export**: Created a backend endpoint `/api/export_dataset` and a UI download button to download the captured gesture dataset instantly.
4. **Historical Accuracies Log**: Implemented `data/training_history.json` logger in `backend/train.py` and rendered the historic runs log in a UI table on the benchmarks tab.
5. **Spanish (LSE) Sign Language**: Introduced support for Lengua de Señas Española with gestures `Hola`, `Si`, `No`, and `Gracias`.
6. **Sign Language Digits**: Integrated the kabilan03/sign-language-dataset (or local photos), extracted hand skeleton landmarks using the new MediaPipe Tasks HandLandmarker API, and successfully retrained all ML models (GNN, MLP, RF) on the combined dataset.

---

<div align="center">

Made with 🤟 by **[NitheshK4](https://github.com/NitheshK4)**

⭐ **Star this repo if you found it useful!** ⭐

</div>
