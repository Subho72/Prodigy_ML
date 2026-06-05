<div align="center">

<!-- Banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=200&section=header&text=Machine-Learning&fontSize=68&fontColor=ffffff&fontAlignY=38&desc=Machine%20Learning%20Internship%20Tasks%20%E2%80%94%20Prodigy%20Infotech&descAlignY=58&descSize=18&animation=fadeIn" width="100%"/>

<br/>

<!-- Badges -->
![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0097A7?style=for-the-badge&logo=google&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-6366f1?style=for-the-badge)

<br/>

> *A collection of end-to-end Machine Learning projects completed during the **Prodigy Infotech ML Internship**,  
> covering regression, clustering, computer vision, and gesture recognition.*

<br/>

</div>

---

## 📋 Table of Contents

- [About the Internship](#-about-the-internship)
- [Task Overview](#-task-overview)
- [Project Details](#-project-details)
  - [Task 01 — House Price Prediction](#-task-01--house-price-prediction)
  - [Task 02 — Customer Segmentation](#-task-02--customer-segmentation)
  - [Task 03 — Cats vs Dogs Classifier](#-task-03--cats-vs-dogs-classifier)
  - [Task 04 — Hand Gesture Recognition](#-task-04--hand-gesture-recognition)
- [Tech Stack](#-tech-stack)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
- [Author](#-author)

---

## 🏢 About the Internship

This repository contains all tasks completed as part of the **Machine Learning Internship at [Prodigy Infotech](https://prodigyinfotech.dev/)** — a remote, project-based internship program designed to build practical ML skills through real-world datasets and industry-standard tools.

Each task targets a distinct machine learning paradigm, from classical regression and unsupervised clustering to deep learning-based image classification and gesture recognition.

---

## 🗺 Task Overview

| # | Task | Algorithm | Dataset | Status |
|---|------|-----------|---------|--------|
| 01 | House Price Prediction | Linear Regression | Kaggle House Prices | ✅ Completed |
| 02 | Customer Segmentation | K-Means Clustering | Mall Customer Dataset | ✅ Completed |
| 03 | Cats vs Dogs Classification | Support Vector Machine (SVM) | Kaggle Dogs vs Cats | ✅ Completed |
| 04 | Hand Gesture Recognition | CNN / Deep Learning | LeapGestRecog Dataset | ✅ Completed |

---

## 📂 Project Details

### 🏠 Task 01 — House Price Prediction

> **Goal:** Predict house sale prices using structural features.

**Algorithm:** Linear Regression  
**Dataset:** [Kaggle House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

**Key Features Used:**
- Square footage (GrLivArea)
- Number of bedrooms & bathrooms
- Overall quality and condition
- Neighborhood and year built

**Highlights:**
- Data preprocessing: handling missing values, feature encoding, outlier removal
- Exploratory Data Analysis (EDA) with correlation heatmaps
- Model evaluation using RMSE and R² score

**📁 Folder:** `PRODGIY_ML_01/`

---

### 🛍 Task 02 — Customer Segmentation

> **Goal:** Group retail customers into distinct segments based on purchase behavior.

**Algorithm:** K-Means Clustering  
**Dataset:** [Mall Customer Segmentation](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

**Key Features Used:**
- Annual Income
- Spending Score

**Highlights:**
- Elbow Method to determine optimal number of clusters
- Cluster visualization with 2D scatter plots
- Business insight generation per customer segment

**📁 Folder:** `PRODGIY_ML_02/`

---

### 🐱🐶 Task 03 — Cats vs Dogs Classifier

> **Goal:** Binary image classification using Support Vector Machines.

**Algorithm:** Support Vector Machine (SVM)  
**Dataset:** [Kaggle Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats/data)

**Key Steps:**
- Image resizing and grayscale conversion
- Feature flattening and normalization
- SVM with RBF kernel for non-linear classification
- Accuracy and confusion matrix evaluation

**Highlights:**
- Demonstrates SVM's application in computer vision
- Preprocessing pipeline for image datasets

**📁 Folder:** `PRODGIY_ML_03/`

---

### 🤚 Task 04 — Hand Gesture Recognition

> **Goal:** Recognize and classify hand gestures from images for human-computer interaction.

**Algorithm:** Convolutional Neural Network (CNN)  
**Dataset:** [LeapGestRecog — Kaggle](https://www.kaggle.com/gti-upm/leapgestrecog)

**Key Steps:**
- Dataset loading and label encoding
- CNN architecture with Conv2D, MaxPooling, and Dense layers
- Model training with validation split
- Accuracy benchmarking per gesture class

**Highlights:**
- Real-world application in gesture-based control systems
- Multi-class classification across 10 gesture categories

**📁 Folder:** `PRODGIY_ML_04/`

---

## 🛠 Tech Stack

<div align="center">

| Tool | Purpose |
|------|---------|
| **Python 3.8+** | Core programming language |
| **NumPy & Pandas** | Data manipulation and analysis |
| **Matplotlib & Seaborn** | Data visualization |
| **Scikit-learn** | ML algorithms and evaluation |
| **TensorFlow / Keras** | Deep learning (Task 04) |
| **OpenCV** | Image processing |
| **Jupyter Notebook** | Interactive development environment |

</div>

---

## 📁 Repository Structure

```
Prodigy_ML/
│
├── PRODGIY_ML_01/          # House Price Prediction
│   └── ...
│
├── PRODGIY_ML_02/          # Customer Segmentation
│   └── ...
│
├── PRODGIY_ML_03/          # Cats vs Dogs Classification
│   └── ...
│
├── PRODGIY_ML_04/          # Hand Gesture Recognition
│   └── ...
│
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
Python >= 3.8
pip (Python package manager)
```

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Subho72/Prodigy_ML.git
cd Prodigy_ML

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate       # Linux/macOS
venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow opencv-python jupyter
```

### Running a Task

```bash
# Navigate to a task folder and launch Jupyter
cd PRODGIY_ML_01
jupyter notebook
```

---

## 👤 Author

<div align="center">

**Subham Sahoo**  
Machine Learning Intern — Prodigy Infotech

[![GitHub](https://img.shields.io/badge/GitHub-Subho72-181717?style=for-the-badge&logo=github)](https://github.com/Subho72)

</div>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:24243e,50:302b63,100:0f0c29&height=100&section=footer" width="100%"/>

*Built with 💙 during the Prodigy Infotech Machine Learning Internship*

</div>
