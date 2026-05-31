# 🛍️ Customer Segmentation using K-Means Clustering

A Machine Learning project that uses the K-Means Clustering algorithm to segment retail store customers based on their annual income and spending score. The project includes an interactive Streamlit web application for visualizing customer groups and predicting customer segments.

---

## 📌 Project Overview

Customer segmentation helps businesses understand customer behavior and target specific groups with personalized marketing strategies.

In this project, K-Means Clustering is used to divide customers into different groups based on:

- 💰 Annual Income (k$)
- 🛒 Spending Score (1-100)

---

## 🚀 Features

- Customer Segmentation using K-Means Clustering
- Interactive Streamlit Dashboard
- Customer Cluster Visualization
- Customer Segment Prediction
- Data Visualization with Matplotlib

---

## 📊 Dataset

Dataset: Mall Customer Segmentation Data

Source: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

### Features Used

| Feature | Description |
|----------|-------------|
| CustomerID | Unique Customer ID |
| Gender | Male/Female |
| Age | Customer Age |
| Annual Income (k$) | Annual Income |
| Spending Score (1-100) | Spending Behavior Score |

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Streamlit
- Joblib

---

## 📂 Project Structure

```bash
customer_segmentation/
│
├── Mall_Customers.csv
├── customer_segmentation.py
├── app.py
├── kmeans_model.pkl
├── requirements.txt
└── README.md
```

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/your-username/customer-segmentation.git
cd customer-segmentation
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train Model

```bash
python customer_segmentation.py
```

### Run Streamlit App

```bash
streamlit run app.py
```

---

## 📈 Model

The project uses the K-Means Clustering algorithm.

```python
from sklearn.cluster import KMeans

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

kmeans = KMeans(
    n_clusters=5,
    random_state=42
)

kmeans.fit(X)
```

---

## 💻 Application Features

- Visualize customer segments
- Predict customer cluster
- Interactive Streamlit interface
- Scatter plot visualization

---

## 🎯 Business Applications

- Customer Behavior Analysis
- Targeted Marketing
- Product Recommendations
- Customer Retention
- Revenue Optimization

---

## 🔮 Future Improvements

- Elbow Method for optimal cluster selection
- Additional customer features
- Plotly visualizations
- Online deployment

---

## 📜 License

This project is developed for educational purposes.

---

## 👨‍💻 Author

**Subham**

⭐ If you found this project useful, consider giving it a star.