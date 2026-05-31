# 🏠 House Price Predictor — Linear Regression

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

> A fully interactive **Linear Regression** web app built with **Streamlit** that predicts house prices based on square footage, number of bedrooms, bathrooms, and more — inspired by the [Kaggle House Prices dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data).

---

## 📸 App Preview

| Model Metrics | Visualizations |
|---|---|
| R² Score, MAE, RMSE cards | Actual vs Predicted scatter |
| Feature coefficient chart | Residual distribution |
| Sidebar prediction panel | Correlation heatmap |

---

## ✨ Features

- 🔮 **Live Price Prediction** — Adjust sliders in the sidebar and instantly predict a house price
- 📊 **Model Performance Tab** — View R², MAE, RMSE with styled metric cards
- 📈 **Visualization Tab** — 4 charts: Actual vs Predicted, Residuals, Sq Ft scatter, Avg price by bathrooms
- 📋 **Dataset Explorer** — Browse data, summary stats, and a full correlation heatmap
- ℹ️ **How It Works Tab** — Step-by-step pipeline explanation and formula breakdown
- 🎨 **Custom Dark UI** — Styled with Playfair Display font, gold accent theme

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| `Streamlit` | Web app framework |
| `scikit-learn` | LinearRegression, StandardScaler, metrics |
| `pandas` | Data manipulation |
| `numpy` | Numerical computation |
| `matplotlib` & `seaborn` | Visualizations |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/house-price-predictor.git
cd house-price-predictor
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` 🎉

---

## 📁 Project Structure

```
PRODIGY_ML_01/
│
├── app.py      # Main Streamlit application
├── requirements.txt        # Python dependencies
├── train.csv
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

## 🔬 Model Details

### Features Used

| Feature | Description |
|---------|-------------|
| `GrLivArea` | Above-grade living area (square feet) |
| `BedroomAbvGr` | Number of bedrooms |
| `FullBath` | Number of bathrooms |
| `GarageCars` | Garage capacity (cars) |
| `YearBuilt` | Original construction year |
| `LotArea` | Lot size in square feet |

### Pipeline

```
Raw Data → Train/Test Split (80/20) → StandardScaler → LinearRegression → Evaluation
```

### Model Formula

```
Price = w₁·SqFt + w₂·Bedrooms + w₃·Bathrooms
      + w₄·GarageCars + w₅·YearBuilt + w₆·LotArea + bias
```

### Performance (on synthetic data)

| Metric | Value |
|--------|-------|
| **R² Score** | 0.9816 |
| **MAE** | ~$11,981 |
| **RMSE** | ~$14,563 |
| **Train Samples** | 1,200 |
| **Test Samples** | 300 |

---

## 🔗 Using Real Kaggle Data

1. Download `train.csv` from [Kaggle House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
2. Place it in the project root directory
3. Replace the `generate_data()` function in `house_price_app.py`:

```python
# Replace this line:
df = generate_data()

# With this:
df = pd.read_csv("train.csv")
df = df[["GrLivArea", "BedroomAbvGr", "FullBath", "GarageCars", "YearBuilt", "LotArea", "SalePrice"]].dropna()
```

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

1. Fork the repo
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

Subham Sahoo

> ⭐ If you found this helpful, please **star the repository**!