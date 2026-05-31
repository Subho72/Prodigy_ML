import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🏠 House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    h1 {
        font-family: 'Playfair Display', serif !important;
        font-size: 3rem !important;
        background: linear-gradient(90deg, #f7971e, #ffd200);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem !important;
    }

    .subtitle {
        text-align: center;
        color: #aab4c4;
        font-size: 1rem;
        margin-bottom: 2rem;
        letter-spacing: 0.05em;
    }

    .metric-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,215,0,0.2);
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: transform 0.2s;
    }

    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(255,215,0,0.5);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #ffd200;
        font-family: 'Playfair Display', serif;
    }

    .metric-label {
        font-size: 0.78rem;
        color: #8892a4;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-top: 0.3rem;
    }

    .prediction-box {
        background: linear-gradient(135deg, rgba(247,151,30,0.15), rgba(255,210,0,0.08));
        border: 2px solid #ffd200;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
    }

    .prediction-label {
        color: #aab4c4;
        font-size: 0.9rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }

    .prediction-value {
        font-family: 'Playfair Display', serif;
        font-size: 3rem;
        color: #ffd200;
        margin: 0.5rem 0;
    }

    .stSlider > div > div > div {
        background: linear-gradient(90deg, #f7971e, #ffd200) !important;
    }

    .stSidebar {
        background: rgba(15, 12, 41, 0.95) !important;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #302b63 100%) !important;
        border-right: 1px solid rgba(255,215,0,0.15);
    }

    .section-header {
        font-family: 'Playfair Display', serif;
        font-size: 1.4rem;
        color: #ffd200;
        border-bottom: 1px solid rgba(255,215,0,0.2);
        padding-bottom: 0.5rem;
        margin-bottom: 1.2rem;
    }

    .stButton > button {
        background: linear-gradient(90deg, #f7971e, #ffd200) !important;
        color: #0f0c29 !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        padding: 0.7rem 2rem !important;
        transition: all 0.3s !important;
        width: 100%;
    }

    .stButton > button:hover {
        transform: scale(1.03) !important;
        box-shadow: 0 6px 24px rgba(255,210,0,0.35) !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── Data Generation (Kaggle-like house prices dataset) ─────────────────────
@st.cache_data
def generate_data(n=1500, seed=42):
    np.random.seed(seed)
    sqft      = np.random.randint(500, 5000, n)
    bedrooms  = np.random.randint(1, 7, n)
    bathrooms = np.random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4], n)
    garage    = np.random.randint(0, 4, n)
    year_built= np.random.randint(1950, 2023, n)
    lot_area  = np.random.randint(1000, 20000, n)

    noise = np.random.normal(0, 15000, n)
    price = (
        80 * sqft
        + 12000 * bedrooms
        + 18000 * bathrooms
        + 8000  * garage
        + 100   * (year_built - 1950)
        + 1.5   * lot_area
        + noise
    )
    price = np.clip(price, 50000, 900000)

    df = pd.DataFrame({
        "GrLivArea":   sqft,
        "BedroomAbvGr":bedrooms,
        "FullBath":    bathrooms,
        "GarageCars":  garage,
        "YearBuilt":   year_built,
        "LotArea":     lot_area,
        "SalePrice":   price.astype(int)
    })
    return df


@st.cache_resource
def train_model(df):
    features = ["GrLivArea", "BedroomAbvGr", "FullBath", "GarageCars", "YearBuilt", "LotArea"]
    X = df[features]
    y = df["SalePrice"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_sc, y_train)

    y_pred = model.predict(X_test_sc)
    metrics = {
        "R² Score":  round(r2_score(y_test, y_pred), 4),
        "MAE":       round(mean_absolute_error(y_test, y_pred), 2),
        "RMSE":      round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
        "Train Size":len(X_train),
        "Test Size": len(X_test),
    }
    return model, scaler, features, X_test, y_test, y_pred, metrics, df[features + ["SalePrice"]]


# ─── Load data & model ───────────────────────────────────────────────────────
df = generate_data()
model, scaler, features, X_test, y_test, y_pred, metrics, full_df = train_model(df)

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("<h1>🏠 House Price Predictor</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">Linear Regression · Kaggle-style Dataset · Square Footage · Bedrooms · Bathrooms</p>', unsafe_allow_html=True)

# ─── Sidebar: Prediction Inputs ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏡 Enter House Details")
    st.markdown("---")

    sqft_input  = st.slider("📐 Square Footage (GrLivArea)", 300, 6000, 1500, step=50)
    bed_input   = st.slider("🛏️ Bedrooms",  1, 8, 3)
    bath_input  = st.select_slider("🚿 Bathrooms", options=[1, 1.5, 2, 2.5, 3, 3.5, 4], value=2)
    garage_input= st.slider("🚗 Garage Cars", 0, 4, 1)
    year_input  = st.slider("📅 Year Built", 1900, 2024, 2000)
    lot_input   = st.slider("🌿 Lot Area (sq ft)", 500, 25000, 8000, step=100)

    st.markdown("---")
    predict_btn = st.button("🔮 Predict Price")


# ─── Prediction ──────────────────────────────────────────────────────────────
predicted_price = None
if predict_btn:
    input_data = np.array([[sqft_input, bed_input, bath_input, garage_input, year_input, lot_input]])
    input_scaled = scaler.transform(input_data)
    predicted_price = model.predict(input_scaled)[0]

# ─── Main Layout ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Performance", "📈 Visualizations", "📋 Dataset Explorer", "ℹ️ How It Works"])

# ══ TAB 1: Metrics ═══════════════════════════════════════════════════════════
with tab1:
    if predicted_price:
        st.markdown(f"""
        <div class="prediction-box">
            <div class="prediction-label">Predicted House Price</div>
            <div class="prediction-value">${predicted_price:,.0f}</div>
            <div style="color:#8892a4; font-size:0.85rem;">Based on {sqft_input} sq ft · {bed_input} bed · {bath_input} bath · Garage {garage_input} · Built {year_input}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-header">📊 Model Metrics</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        (c1, "R² Score",   f"{metrics['R² Score']}",              "Coefficient of Determination"),
        (c2, "MAE",        f"${metrics['MAE']:,.0f}",              "Mean Absolute Error"),
        (c3, "RMSE",       f"${metrics['RMSE']:,.0f}",             "Root Mean Squared Error"),
        (c4, "Train Rows", f"{metrics['Train Size']:,}",           "Training Samples"),
        (c5, "Test Rows",  f"{metrics['Test Size']:,}",            "Testing Samples"),
    ]
    for col, title, val, sub in cards:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{title}</div>
                <div style="color:#5a6478;font-size:0.7rem;margin-top:0.2rem">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">🧠 Feature Coefficients</div>', unsafe_allow_html=True)
    coef_df = pd.DataFrame({
        "Feature": ["Sq. Footage", "Bedrooms", "Bathrooms", "Garage Cars", "Year Built", "Lot Area"],
        "Coefficient": model.coef_
    }).sort_values("Coefficient", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 3.5), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    colors = ["#f7971e" if c > 0 else "#e74c3c" for c in coef_df["Coefficient"]]
    bars = ax.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors, edgecolor="none", height=0.6)
    ax.axvline(0, color="#ffffff30", linewidth=1)
    ax.set_xlabel("Coefficient Value", color="#aab4c4", fontsize=9)
    ax.tick_params(colors="#aab4c4", labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Standardized Feature Coefficients", color="#ffd200", fontsize=11, pad=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ══ TAB 2: Visualizations ════════════════════════════════════════════════════
with tab2:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">🎯 Actual vs Predicted</div>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(6, 5), facecolor="#1a1a2e")
        ax2.set_facecolor("#1a1a2e")
        ax2.scatter(y_test, y_pred, alpha=0.4, color="#f7971e", s=18, edgecolors="none")
        mn = min(y_test.min(), y_pred.min())
        mx = max(y_test.max(), y_pred.max())
        ax2.plot([mn, mx], [mn, mx], "--", color="#ffd200", linewidth=1.5, label="Perfect Fit")
        ax2.set_xlabel("Actual Price ($)", color="#aab4c4", fontsize=9)
        ax2.set_ylabel("Predicted Price ($)", color="#aab4c4", fontsize=9)
        ax2.tick_params(colors="#aab4c4", labelsize=8)
        ax2.legend(facecolor="#0f0c29", edgecolor="#ffd20040", labelcolor="#aab4c4")
        for spine in ax2.spines.values(): spine.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    with col_b:
        st.markdown('<div class="section-header">📉 Residuals Distribution</div>', unsafe_allow_html=True)
        residuals = y_pred - np.array(y_test)
        fig3, ax3 = plt.subplots(figsize=(6, 5), facecolor="#1a1a2e")
        ax3.set_facecolor("#1a1a2e")
        ax3.hist(residuals, bins=40, color="#f7971e", edgecolor="#0f0c29", alpha=0.85)
        ax3.axvline(0, color="#ffd200", linewidth=1.5, linestyle="--")
        ax3.set_xlabel("Residual ($)", color="#aab4c4", fontsize=9)
        ax3.set_ylabel("Frequency", color="#aab4c4", fontsize=9)
        ax3.tick_params(colors="#aab4c4", labelsize=8)
        for spine in ax3.spines.values(): spine.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown('<div class="section-header">📐 Price vs Square Footage</div>', unsafe_allow_html=True)
        fig4, ax4 = plt.subplots(figsize=(6, 4.5), facecolor="#1a1a2e")
        ax4.set_facecolor("#1a1a2e")
        sample = df.sample(500, random_state=1)
        sc = ax4.scatter(sample["GrLivArea"], sample["SalePrice"],
                         c=sample["BedroomAbvGr"], cmap="plasma", alpha=0.6, s=20, edgecolors="none")
        cb = plt.colorbar(sc, ax=ax4)
        cb.set_label("Bedrooms", color="#aab4c4", fontsize=8)
        cb.ax.yaxis.set_tick_params(color="#aab4c4")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="#aab4c4", fontsize=8)
        ax4.set_xlabel("GrLivArea (sq ft)", color="#aab4c4", fontsize=9)
        ax4.set_ylabel("Sale Price ($)", color="#aab4c4", fontsize=9)
        ax4.tick_params(colors="#aab4c4", labelsize=8)
        for spine in ax4.spines.values(): spine.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

    with col_d:
        st.markdown('<div class="section-header">🚿 Avg Price by Bathrooms</div>', unsafe_allow_html=True)
        bath_avg = df.groupby("FullBath")["SalePrice"].mean().reset_index()
        fig5, ax5 = plt.subplots(figsize=(6, 4.5), facecolor="#1a1a2e")
        ax5.set_facecolor("#1a1a2e")
        bars = ax5.bar(bath_avg["FullBath"].astype(str), bath_avg["SalePrice"],
                       color=["#f7971e","#ffd200","#e5b800","#c49a00","#f7971e","#ffd200","#e5b800"],
                       edgecolor="none", width=0.55)
        for bar in bars:
            h = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., h + 2000, f'${h/1000:.0f}k',
                     ha='center', va='bottom', color='#aab4c4', fontsize=8)
        ax5.set_xlabel("Bathrooms", color="#aab4c4", fontsize=9)
        ax5.set_ylabel("Avg Sale Price ($)", color="#aab4c4", fontsize=9)
        ax5.tick_params(colors="#aab4c4", labelsize=8)
        for spine in ax5.spines.values(): spine.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close()


# ══ TAB 3: Dataset ═══════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">📋 Dataset Preview</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(
            full_df.head(50).style
            .background_gradient(subset=["SalePrice"], cmap="YlOrBr")
            .format({"SalePrice": "${:,.0f}"}),
            use_container_width=True, height=420
        )
    with col2:
        st.markdown("**📊 Summary Statistics**")
        st.dataframe(full_df.describe().round(1), use_container_width=True, height=420)

    st.markdown("**🔥 Correlation Heatmap**")
    fig6, ax6 = plt.subplots(figsize=(8, 5), facecolor="#1a1a2e")
    ax6.set_facecolor("#1a1a2e")
    corr = full_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="YlOrBr",
                ax=ax6, linewidths=0.5, linecolor="#1a1a2e",
                annot_kws={"size": 9, "color": "#0f0c29"},
                cbar_kws={"shrink": 0.8})
    ax6.tick_params(colors="#aab4c4", labelsize=9)
    for spine in ax6.spines.values(): spine.set_visible(False)
    plt.tight_layout()
    st.pyplot(fig6)
    plt.close()


# ══ TAB 4: How It Works ══════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">ℹ️ How This Works</div>', unsafe_allow_html=True)
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("""
        **🏗️ Dataset**
        - Synthetic data inspired by Kaggle's *House Prices: Advanced Regression Techniques*
        - 1,500 samples with realistic house attributes
        - Features: GrLivArea, Bedrooms, Bathrooms, Garage, YearBuilt, LotArea

        **🔬 Model Pipeline**
        1. **Generate** realistic data with noise
        2. **Split** → 80% train / 20% test
        3. **Scale** features with StandardScaler
        4. **Fit** `sklearn.LinearRegression`
        5. **Evaluate** on held-out test set

        **📐 Formula (conceptually)**
        ```
        Price = w₁·Sqft + w₂·Beds + w₃·Baths
              + w₄·Garage + w₅·YearBuilt + w₆·Lot + b
        ```
        """)
    with col_r:
        st.markdown("""
        **📏 Metrics Explained**

        | Metric | Meaning |
        |--------|---------|
        | **R²** | % variance explained (1.0 = perfect) |
        | **MAE** | Avg absolute dollar error |
        | **RMSE** | Penalizes large errors more |

        **🎯 How to Use**
        - Adjust sliders in the **sidebar**
        - Click **Predict Price** 🔮
        - See predicted value in **Model Performance** tab

        **🔗 Real Dataset**
        > Upload the Kaggle `train.csv` and replace the `generate_data()` function with `pd.read_csv('train.csv')` for real results!
        """)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border-color:rgba(255,215,0,0.15); margin-top:3rem;">
<p style="text-align:center; color:#5a6478; font-size:0.8rem;">
    🏠 House Price Predictor · Linear Regression · Built with Streamlit & scikit-learn
</p>
""", unsafe_allow_html=True)