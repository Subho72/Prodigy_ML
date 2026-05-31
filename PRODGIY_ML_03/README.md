# 🐱🐶 Cats vs Dogs — SVM Image Classifier

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)

> An interactive **Support Vector Machine (SVM)** web app built with **Streamlit** that classifies images of cats and dogs using hand-crafted image features, PCA dimensionality reduction, and an RBF-kernel SVM — inspired by the [Kaggle Dogs vs Cats dataset](https://www.kaggle.com/c/dogs-vs-cats/data).

---

## 📸 App Overview

| Tab | Description |
|-----|-------------|
| 📊 Model Performance | Accuracy, F1, AUC-ROC, Confusion Matrix, CV scores |
| 🗺️ Decision Boundary | PCA 2D SVM boundary with support vector markers |
| 🔍 Predict an Image | Upload your own image or try a dataset sample |
| 🖼️ Dataset Samples | Cat & dog image grids + PCA feature distributions |
| ℹ️ How It Works | Full pipeline, kernel comparison, Kaggle integration guide |

---

## ✨ Features

- 🎚️ **Live Hyperparameter Tuning** — Switch kernel (RBF / Linear / Poly / Sigmoid), adjust C and Gamma from the sidebar in real time
- 🖼️ **Image Upload & Prediction** — Upload any cat or dog photo and get instant SVM classification with probability scores
- 🌟 **Support Vector Visualisation** — Decision boundary plot with support vectors highlighted in yellow
- 📐 **PCA Dimensionality Reduction** — Features compressed to 80 principal components before SVM training
- 🔄 **5-Fold Cross Validation** — Robust generalisation estimate with per-fold bar chart
- 📈 **ROC Curve & AUC** — Full ROC analysis with AUC score
- 🔥 **Feature Distribution** — PCA component histograms comparing cats vs dogs

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| `Streamlit` | Interactive web application |
| `scikit-learn` | SVM, PCA, StandardScaler, metrics, cross-validation |
| `OpenCV` | Image processing, gradient computation, resizing |
| `Pillow` | Image loading from file uploads |
| `pandas` | Classification report table |
| `numpy` | Numerical computation |
| `matplotlib` & `seaborn` | Visualisations and heatmaps |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/cats-vs-dogs-svm.git
cd cats-vs-dogs-svm
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run svm_app.py
```

Opens at `http://localhost:8501` 🎉

---

## 📁 Project Structure

```
cats-vs-dogs-svm/
│
├── svm_app.py              # Main Streamlit application
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
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
```

---

## 📊 Dataset

**Source:** [Kaggle — Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats/data)

| Detail | Info |
|--------|------|
| Total Images | 200 (train) |
| Classes | Cat (0) / Dog (1) |
| Format | JPG |
| Naming | `cat.0.jpg`, `dog.0.jpg`, ... |

> The app ships with **procedurally generated** synthetic images matching the dataset structure so it runs out of the box with no downloads. Swap in the real Kaggle images for production-grade results.

---

## 🔬 Model Pipeline

```
Raw Image (any size)
       │
       ▼
  Resize → 64×64 RGB
       │
       ▼
 Feature Extraction
  ├── Pixel intensities    (32×32 = 1,024 features)
  ├── Colour histograms    (16 bins × 3 channels = 48)
  ├── Gradient magnitude   (mean, std, max = 3)
  ├── LBP texture approx  (64 features)
  └── Spatial quadrant stats (4×4 grid × 2 = 32)
       │
       ▼
  StandardScaler  →  PCA (80 components)
       │
       ▼
  SVM Classifier (RBF kernel)
       │
       ▼
    Cat 🐱  /  Dog 🐶
```

---

## ⚙️ Hyperparameters

| Parameter | Options | Recommended |
|-----------|---------|-------------|
| **Kernel** | `rbf`, `linear`, `poly`, `sigmoid` | `rbf` ✅ |
| **C** (Regularisation) | 0.01 → 100 | `10` |
| **Gamma** | `scale`, `auto` | `scale` |

---

## 🧠 SVM Kernels Explained

| Kernel | Decision Boundary | Best For |
|--------|------------------|----------|
| `rbf` | Non-linear (Gaussian) | Image classification ✅ |
| `linear` | Straight hyperplane | High-dim linearly separable |
| `poly` | Polynomial curve | Structured feature interaction |
| `sigmoid` | Neural-network-like | Experimental |

---

## 📏 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | % of images correctly classified |
| **Precision** | TP / (TP + FP) per class |
| **Recall** | TP / (TP + FN) per class |
| **F1 Score** | Harmonic mean of Precision & Recall |
| **AUC-ROC** | Area under the ROC curve (1.0 = perfect) |
| **5-Fold CV** | Cross-validated accuracy for generalisation |

---

## 📈 Sample Results (Synthetic Data, RBF kernel, C=10)

| Metric | Value |
|--------|-------|
| **Test Accuracy** | ~100% |
| **AUC-ROC** | 1.000 |
| **Support Vectors** | ~18 |
| **CV Mean Accuracy** | ~98%+ |
| **PCA Components** | 80 |

---

## 🔗 Using the Real Kaggle Dataset

1. Download and extract the dataset from [Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data)
2. Place all `.jpg` files inside a `train/` folder in the project root

> **Tip:** Start with `max_per_class=500` for faster training. SVM scales as O(n²) so large datasets may be slow — consider using a linear kernel for speed.

---

## 🤝 Contributing

1. Fork the repository
2. Create a branch: `git checkout -b feature/improvement`
3. Commit: `git commit -m "Add improvement"`
4. Push: `git push origin feature/improvement`
5. Open a Pull Request

---

## 👤 Author

Subham Sahoo💙

> ⭐ Found this helpful? Please **star the repository!**