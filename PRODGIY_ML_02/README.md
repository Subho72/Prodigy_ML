# 🛍️ Customer Segmentation using K-Means Clustering

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

> An interactive **K-Means Clustering** web application built with **Streamlit** that segments retail store customers based on their **Annual Income**, **Spending Score**, and **Age** — based on the [Kaggle Mall Customer Segmentation dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python).

---

## 📸 App Overview

| Tab | Description |
|-----|-------------|
| 🎯 Segments Overview | Cluster profile cards, KPI metrics, size distribution |
| 📈 Elbow & Silhouette | Optimal K selection charts |
| 🗺️ Cluster Maps | Scatter plot, PCA 2D projection, violin plot, feature heatmap |
| 📋 Data Explorer | Filterable clustered dataset with statistics |
| ℹ️ How It Works | Algorithm explanation and Kaggle data integration guide |

---

## ✨ Features

- 🔢 **Dynamic K Selection** — Adjust number of clusters (2–8) from the sidebar in real time
- 📊 **Multi-Feature Clustering** — Choose from Annual Income, Spending Score, and Age
- 👤 **Gender & Age Filtering** — Analyse specific customer subsets
- 📐 **Elbow Method** — Visualise inertia curve to find the optimal K
- 📏 **Silhouette Score** — Evaluate cluster quality for each K value
- 🗺️ **PCA 2D Projection** — Dimensionality reduction for high-dimensional cluster visualisation
- 🔥 **Feature Heatmap** — Compare cluster means across all features at a glance
- 🎻 **Violin Plots** — Age distribution breakdown per cluster
- ⭐ **Centroid Markers** — Star markers on scatter plots showing cluster centres

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| `Streamlit` | Interactive web application framework |
| `scikit-learn` | KMeans, StandardScaler, PCA, silhouette_score |
| `pandas` | Data manipulation and analysis |
| `numpy` | Numerical computation |
| `matplotlib` | Core plotting engine |
| `seaborn` | Heatmap and statistical visualisations |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/customer-segmentation-kmeans.git
cd customer-segmentation-kmeans
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run kmeans_app.py
```

The app opens at `http://localhost:8501` 🎉

---

## 📁 Project Structure

```
PRODGIY_ML_02/
│
├── app.py           # Main Streamlit application
├── Mall_Customers.csv      # Dataset (download from Kaggle)
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
```

---

## 📊 Dataset

**Source:** [Kaggle — Customer Segmentation Tutorial in Python](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

| Column | Description |
|--------|-------------|
| `CustomerID` | Unique customer identifier |
| `Gender` | Male / Female |
| `Age` | Age of the customer |
| `Annual Income (k$)` | Annual income in thousands of dollars |
| `Spending Score (1-100)` | Score assigned by the mall based on spending behaviour |

> The app ships with a **synthetic dataset** (200 samples) matching the Kaggle schema so it runs out of the box. Plug in the real CSV for full results.

---

## 🔗 Using the Real Kaggle Dataset

Download `Mall_Customers.csv` from [Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

```python
# Replace:
df_raw = generate_data()

# With:
df_raw = pd.read_csv("Mall_Customers.csv")
```

---

## 🔬 Model Details

### Algorithm — K-Means Clustering

```
1. Initialise K centroids randomly
2. Assign each point to the nearest centroid (Euclidean distance)
3. Recompute centroids as the mean of assigned points
4. Repeat steps 2–3 until convergence
```

### Pipeline

```
Raw Data ──► Filter (Gender / Age) ──► StandardScaler ──► KMeans ──► Cluster Labels
                                                                          │
                                                  PCA 2D ◄────────────────┘
```

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Silhouette Score** | Measures cluster cohesion and separation (range: −1 to 1; higher is better) |
| **Inertia (WCSS)** | Sum of squared distances from each point to its cluster centroid |
| **Elbow Method** | Identifies the K where the rate of inertia decrease slows significantly |

### Typical Customer Segments (K=5)

| Segment | Income | Spending | Profile |
|---------|--------|----------|---------|
| 💎 High Value Champions | High | High | Most valuable; target for loyalty rewards |
| 💸 Impulse Buyers | Low | High | Spend beyond means; target with offers |
| 💰 Careful Savers | High | Low | Earn well but save; need targeted incentives |
| 🌱 Potential Loyalists | Mid | Mid | Growing segment; nurture with engagement |
| 😴 At-Risk Customers | Low | Low | Low engagement; at risk of churn |

---

## 📈 Sample Results (Synthetic Data, K=5)

| Metric | Value |
|--------|-------|
| **Silhouette Score** | ~0.39 |
| **Inertia** | ~79.0 |
| **Total Customers** | 200 |
| **Features Used** | Annual Income, Spending Score |

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Commit your changes: `git commit -m "Add new feature"`
4. Push to the branch: `git push origin feature/new-feature`
5. Open a Pull Request

---

## 👤 Author

Subham Sahoo💜
> ⭐ If this project helped you, please **star the repository!**