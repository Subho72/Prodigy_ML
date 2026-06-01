# 🤚 Hand Gesture Recognition — ProDigy Infotech Task-04

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0097A7?style=for-the-badge&logo=google&logoColor=white)

> A fully interactive **Hand Gesture Recognition** web application built with **Streamlit** that classifies 10 different hand gestures using **21 MediaPipe-style hand landmarks**, multiple ML classifiers, and rich visualisations — based on the [LeapGestRecog dataset on Kaggle](https://www.kaggle.com/gti-upm/leapgestrecog).

---

## 📸 App Overview

| Tab | Description |
|-----|-------------|
| 📊 Performance | Accuracy, 10-class Confusion Matrix, 5-Fold CV, per-class F1 scores |
| 🖐️ Gesture Gallery | Hand skeleton visualisation of all 10 gestures + landmark heatmap |
| 🗺️ Feature Space | PCA 2D scatter, feature importance, inter-gesture distance matrix |
| 🔍 Live Predict | Simulate any gesture or upload a hand image for real-time prediction |
| 📋 Data Explorer | Dataset summary, class distribution, landmark correlation heatmap |
| ℹ️ Pipeline | Full processing pipeline, landmark table, Kaggle integration guide |

---

## ✨ Features

- 🤚 **Hand Skeleton Visualisation** — All 10 gestures drawn as 2D skeletons using 21 hand landmarks
- 🧠 **4 Classifier Options** — Switch between MLP Neural Network, Random Forest, SVM (RBF), and Gradient Boosting from the sidebar
- 🎯 **10-Class Recognition** — All LeapGestRecog gesture classes: palm, l, fist, fist_moved, thumb, index, ok, palm_moved, c, down
- 🔄 **5-Fold Cross Validation** — Robust generalisation estimate with per-fold bar chart
- 🗺️ **PCA Feature Space** — 2D projection showing cluster separation between gesture classes
- 📊 **Top-5 Predictions** — Confidence probability bar for each gesture class
- 📁 **Image Upload** — Upload your own hand photo for gesture prediction
- 🔥 **Landmark Heatmap** — Average landmark pattern per gesture class visualised as a heatmap
- 📐 **Inter-Gesture Distance Matrix** — Euclidean distance between class centroids in feature space

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| `Streamlit` | Interactive web application framework |
| `scikit-learn` | MLP, RandomForest, SVM, GBM, StandardScaler, PCA, metrics |
| `MediaPipe` | Hand landmark detection (21 joints per hand) |
| `OpenCV` | Image processing, edge detection, feature extraction |
| `Pillow` | Image loading from file uploads |
| `pandas` | Dataset summary and classification report tables |
| `numpy` | Numerical computation and feature engineering |
| `matplotlib` & `seaborn` | Visualisations, heatmaps, skeleton drawing |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/hand-gesture-recognition.git
cd hand-gesture-recognition
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run gesture_app.py
```

Opens at `http://localhost:8501` 🎉

---

## 📁 Project Structure

```
hand-gesture-recognition/
│
├── gesture_app.py              # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── leapgestrecog/              # Kaggle dataset folder (place here)
    ├── 00/
        ├── 01_palm/
        ├── 02_l/
        ├── 03_fist/
        ├── 04_fist_moved/
        ├── 05_thumb/
        ├── 06_index/
        ├── 07_ok/
        ├── 08_palm_moved/
        ├── 09_c/
        └── 10_down/
```

---

## 📦 requirements.txt

```
streamlit>=1.28.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
opencv-python-headless>=4.8.0
Pillow>=10.0.0
mediapipe>=0.10.0
```

---

## 📊 Dataset — LeapGestRecog

**Source:** [Kaggle — LeapGestRecog by GTI-UPM](https://www.kaggle.com/gti-upm/leapgestrecog)

| Property | Detail |
|----------|--------|
| Subjects | 10 different people |
| Gesture Classes | 10 |
| Total Images | ~20,000 infrared images |
| Format | PNG, grayscale |
| Resolution | 640 × 240 pixels |
| Sensor | Leap Motion Controller |

### 🖐️ Gesture Classes

| ID | Gesture | Emoji | Description |
|----|---------|-------|-------------|
| 0 | `palm` | ✋ | Open palm facing forward |
| 1 | `l` | 🤙 | L-shape: thumb and index extended |
| 2 | `fist` | ✊ | Closed fist, all fingers curled |
| 3 | `fist_moved` | 👊 | Fist with slight wrist movement |
| 4 | `thumb` | 👍 | Thumb up, others curled |
| 5 | `index` | ☝️ | Index finger pointing up |
| 6 | `ok` | 👌 | OK sign: thumb-index circle |
| 7 | `palm_moved` | 🖐️ | Palm with lateral displacement |
| 8 | `c` | 🤌 | C-shape curl of all fingers |
| 9 | `down` | 👇 | Index pointing downward |

---

## 🔬 Model Pipeline

```
Input Image (Leap Motion IR / RGB photo)
              │
              ▼
     Hand Detection (MediaPipe / OpenCV)
              │
              ▼
  21 Landmark Extraction  ──→  x, y, z per joint = 63 features
              │
              ▼
     Feature Engineering
      ├── Normalise by wrist position
      ├── Finger extension states
      ├── Relative joint distances
      └── Wrist-to-tip vectors
              │
              ▼
       StandardScaler
              │
              ▼
     ML Classifier (choose one)
      ├── MLP Neural Network  (256→128→64)
      ├── Random Forest       (200 estimators)
      ├── SVM RBF Kernel      (C=10)
      └── Gradient Boosting   (150 estimators)
              │
              ▼
  Gesture Label + Confidence Score
```

---

## 🧠 Classifiers Compared

| Model | Architecture | Strength | Training Speed |
|-------|-------------|----------|---------------|
| **MLP Neural Network** | 256→128→64 hidden units | Best for complex non-linear patterns | Medium |
| **Random Forest** | 200 decision trees (ensemble) | Robust, interpretable, handles noise | Fast ✅ |
| **SVM (RBF kernel)** | C=10, gamma=scale | Excellent for small-medium datasets | Medium |
| **Gradient Boosting** | 150 estimators, lr=0.1 | High accuracy, boosted ensemble | Slow |

---

## 🖐️ Hand Landmark Map (MediaPipe)

```
                    8   12  16  20
                    |   |   |   |
                7   11  15  19
                |   |   |   |
            6   10  14  18
            |   |   |   |
        4   5   9   13  17
        |               |
        3               |
        |               |
        2       0───────┘
        |       (wrist)
        1
    (thumb)
```

| Landmark IDs | Finger |
|-------------|--------|
| 0 | Wrist |
| 1 – 4 | Thumb (CMC → Tip) |
| 5 – 8 | Index finger (MCP → Tip) |
| 9 – 12 | Middle finger (MCP → Tip) |
| 13 – 16 | Ring finger (MCP → Tip) |
| 17 – 20 | Pinky finger (MCP → Tip) |

---

## 📏 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | % of gestures correctly classified |
| **Precision** | TP / (TP + FP) per class |
| **Recall** | TP / (TP + FN) per class |
| **F1 Score** | Harmonic mean of Precision and Recall |
| **5-Fold CV** | Cross-validated accuracy for generalisation |
| **Confusion Matrix** | 10×10 matrix showing misclassification patterns |

---

## 📈 Sample Results (Synthetic Landmarks, MLP)

| Metric | Value |
|--------|-------|
| **Test Accuracy** | ~68–100% (varies by classifier) |
| **CV Mean Accuracy** | ~95%+ |
| **Gesture Classes** | 10 |
| **Feature Dimensions** | 63 (21 landmarks × 3 axes) |
| **Training Samples** | 2,000 (200 per class) |

> Real Kaggle dataset with ~20,000 images will yield significantly higher and more realistic accuracy scores.

---

## 🔗 Using the Real Kaggle Dataset

1. Download the dataset from [Kaggle](https://www.kaggle.com/gti-upm/leapgestrecog)
2. Extract into a `leapgestrecog/` folder in the project root

---

## 💡 Applications

- 🖥️ **Human-Computer Interaction** — Control interfaces without a mouse or keyboard
- 🎮 **Gaming** — Gesture-based game controls
- 🤖 **Robotics** — Command a robot arm using hand signs
- ♿ **Accessibility** — Enable communication for people with speech or motor disabilities
- 🔒 **Security** — Gesture-based authentication systems
- 🏥 **Healthcare** — Touchless control of medical equipment

---

## 🤝 Contributing

1. Fork the repository
2. Create a branch: `git checkout -b feature/your-feature`
3. Commit: `git commit -m "Add your feature"`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 👤 Author

Subham Sahoo

> ⭐ If this helped you, please **star the repository!**