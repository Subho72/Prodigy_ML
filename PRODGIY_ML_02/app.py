import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🛍️ Customer Segmentation",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #0d0d14; }
.block-container { padding: 1.5rem 2rem 3rem; }

h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 2.8rem !important;
    background: linear-gradient(135deg, #a78bfa, #f472b6, #fb923c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0 !important;
}
.subtitle {
    text-align: center; color: #6b7280;
    font-size: 0.95rem; letter-spacing: 0.06em;
    margin-bottom: 2rem;
}
.seg-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(167,139,250,0.2);
    border-radius: 18px; padding: 1.4rem 1.6rem;
    transition: transform .2s, border-color .2s;
    height: 100%;
}
.seg-card:hover {
    transform: translateY(-4px);
    border-color: rgba(167,139,250,0.55);
}
.seg-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.05rem; font-weight: 700;
    margin-bottom: 0.5rem;
}
.seg-stat { font-size: 1.8rem; font-weight: 700; font-family: 'Syne', sans-serif; }
.seg-label { font-size: 0.72rem; color: #6b7280; text-transform: uppercase; letter-spacing: .1em; }

.metric-pill {
    display: inline-block;
    background: rgba(167,139,250,0.12);
    border: 1px solid rgba(167,139,250,0.3);
    border-radius: 50px; padding: 0.35rem 1rem;
    font-size: 0.82rem; color: #c4b5fd;
    margin: 0.2rem;
}
.stButton > button {
    background: linear-gradient(135deg,#a78bfa,#f472b6) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; font-weight: 600 !important;
    width: 100%; padding: 0.65rem !important;
    transition: all .3s !important;
}
.stButton > button:hover {
    transform: scale(1.03) !important;
    box-shadow: 0 6px 28px rgba(167,139,250,0.4) !important;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0d0d14,#13101f) !important;
    border-right: 1px solid rgba(167,139,250,0.15);
}
.section-hdr {
    font-family: 'Syne', sans-serif;
    font-size: 1.25rem; color: #a78bfa;
    border-bottom: 1px solid rgba(167,139,250,0.2);
    padding-bottom: 0.4rem; margin-bottom: 1.2rem;
}
.info-box {
    background: rgba(167,139,250,0.07);
    border-left: 3px solid #a78bfa;
    border-radius: 0 12px 12px 0;
    padding: 0.9rem 1.2rem; margin: 0.5rem 0;
    font-size: 0.88rem; color: #d1d5db;
}
</style>
""", unsafe_allow_html=True)

# ─── Cluster colour palette ───────────────────────────────────────────────────
PALETTE = ["#a78bfa","#f472b6","#fb923c","#34d399","#60a5fa","#facc15","#f87171","#818cf8"]

CLUSTER_LABELS = {
    0: ("💎 High Value Champions",   "#a78bfa"),
    1: ("🛒 Regular Shoppers",       "#34d399"),
    2: ("💸 Big Spenders",           "#f472b6"),
    3: ("🌱 Potential Loyalists",    "#fb923c"),
    4: ("😴 At-Risk Customers",      "#60a5fa"),
}

# ─── Data Generation (Mall Customer Segmentation format) ─────────────────────
@st.cache_data
def generate_data(seed=42):
    np.random.seed(seed)
    n = 200
    customer_id = np.arange(1, n+1)
    gender = np.random.choice(["Male","Female"], n, p=[0.44,0.56])
    age = np.random.randint(18, 70, n)

    # Realistic bimodal income & spending
    income  = np.concatenate([
        np.random.normal(30, 8, 40),   # low income
        np.random.normal(55, 10, 80),  # mid income
        np.random.normal(85, 12, 80),  # high income
    ])[:n]
    income = np.clip(income, 15, 137).astype(int)

    spending = np.where(
        income < 40,  np.random.normal(40, 15, n),
        np.where(income < 70, np.random.normal(50, 20, n),
                              np.random.normal(70, 18, n))
    )
    spending = np.clip(spending, 1, 99).astype(int)

    df = pd.DataFrame({
        "CustomerID":           customer_id,
        "Gender":               gender,
        "Age":                  age,
        "Annual Income (k$)":   income,
        "Spending Score (1-100)": spending
    })
    return df

# ─── Elbow & Silhouette ───────────────────────────────────────────────────────
@st.cache_data
def compute_elbow(df, max_k=10):
    X = df[["Annual Income (k$)", "Spending Score (1-100)"]].values
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    inertias, sil_scores = [], []
    for k in range(2, max_k+1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(Xs)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(Xs, labels))
    return list(range(2, max_k+1)), inertias, sil_scores

# ─── Run K-Means ─────────────────────────────────────────────────────────────
@st.cache_data
def run_kmeans(df, k, features):
    X = df[features].values
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(Xs)
    centers_scaled = km.cluster_centers_
    centers_orig = sc.inverse_transform(centers_scaled)
    sil = silhouette_score(Xs, labels)
    inertia = km.inertia_

    # PCA for 2-D vis if >2 features
    pca = PCA(n_components=2)
    Xpca = pca.fit_transform(Xs)

    df2 = df.copy()
    df2["Cluster"] = labels
    df2["PCA_1"]   = Xpca[:,0]
    df2["PCA_2"]   = Xpca[:,1]
    return df2, centers_orig, sil, inertia, sc, km

# ─── Plot helpers ─────────────────────────────────────────────────────────────
DARK_BG  = "#0d0d14"
CARD_BG  = "#13101f"
GRID_COL = "#1f1b2e"
TEXT_COL = "#9ca3af"

def styled_fig(w=7, h=5):
    fig, ax = plt.subplots(figsize=(w,h), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=TEXT_COL, labelsize=8)
    for s in ax.spines.values(): s.set_visible(False)
    ax.grid(color=GRID_COL, linewidth=0.6, linestyle="--")
    return fig, ax

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
df_raw = pd.read_csv("Mall_Customers.csv")
ks, inertias, sil_scores = compute_elbow(df_raw)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Clustering Settings")
    st.markdown("---")

    k = st.slider("🔢 Number of Clusters (K)", 2, 8, 5)
    feature_options = ["Annual Income (k$)", "Spending Score (1-100)", "Age"]
    features = st.multiselect(
        "📊 Features for Clustering",
        feature_options,
        default=["Annual Income (k$)", "Spending Score (1-100)"]
    )
    if len(features) < 2:
        st.warning("Please select at least 2 features!")
        features = ["Annual Income (k$)", "Spending Score (1-100)"]

    gender_filter = st.multiselect("👤 Filter by Gender", ["Male","Female"], default=["Male","Female"])
    age_range = st.slider("🎂 Age Range", 18, 70, (18, 70))

    st.markdown("---")
    run_btn = st.button("🚀 Run Clustering")
    st.markdown("---")
    st.markdown("""
    <div style='color:#6b7280;font-size:0.78rem'>
    📌 <b style='color:#a78bfa'>Optimal K</b> is usually where the<br>
    elbow curve bends or silhouette<br>score is highest.
    </div>""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("<h1>🛍️ Customer Segmentation</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">K-Means Clustering · Mall Customer Dataset · Purchase Behaviour Analysis</p>',
            unsafe_allow_html=True)

# ─── Filter data ──────────────────────────────────────────────────────────────
df_filtered = df_raw[
    (df_raw["Gender"].isin(gender_filter)) &
    (df_raw["Age"] >= age_range[0]) &
    (df_raw["Age"] <= age_range[1])
].reset_index(drop=True)

# ─── Run model ────────────────────────────────────────────────────────────────
df_clustered, centers, sil, inertia, scaler, model = run_kmeans(df_filtered, k, features)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Segments Overview",
    "📈 Elbow & Silhouette",
    "🗺️ Cluster Maps",
    "📋 Data Explorer",
    "ℹ️ How It Works"
])

# ══════════════════ TAB 1 — SEGMENTS OVERVIEW ═════════════════════════════════
with tab1:
    # Top KPIs
    c1,c2,c3,c4 = st.columns(4)
    kpi_data = [
        (c1, f"{len(df_filtered):,}", "Total Customers", "#a78bfa"),
        (c2, f"{k}",                  "Segments Found",  "#f472b6"),
        (c3, f"{sil:.3f}",            "Silhouette Score","#34d399"),
        (c4, f"{inertia:,.0f}",       "Inertia (WCSS)",  "#fb923c"),
    ]
    for col, val, lbl, clr in kpi_data:
        with col:
            st.markdown(f"""
            <div class="seg-card" style="border-color:{clr}30">
                <div class="seg-stat" style="color:{clr}">{val}</div>
                <div class="seg-label">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-hdr">📦 Cluster Profiles</div>', unsafe_allow_html=True)

    cols = st.columns(min(k, 5))
    for i in range(k):
        clr = PALETTE[i % len(PALETTE)]
        seg_df = df_clustered[df_clustered["Cluster"]==i]
        n_cust = len(seg_df)
        pct    = n_cust/len(df_clustered)*100
        label, _ = CLUSTER_LABELS.get(i, (f"Cluster {i}", clr))

        # dominant gender
        dom_gender = seg_df["Gender"].mode()[0] if "Gender" in seg_df else "—"

        with cols[i % len(cols)]:
            stats_html = ""
            for f in features:
                mean_val = seg_df[f].mean()
                stats_html += f'<div style="margin:.25rem 0"><span style="color:#6b7280;font-size:.75rem">{f}</span><br><b style="color:{clr}">{mean_val:.1f}</b></div>'

            st.markdown(f"""
            <div class="seg-card" style="border-color:{clr}50; margin-bottom:.8rem">
                <div class="seg-title" style="color:{clr}">{label}</div>
                <div style="margin:.6rem 0">
                    <span class="metric-pill">👥 {n_cust} customers</span>
                    <span class="metric-pill">{pct:.1f}%</span>
                    <span class="metric-pill">{'♂' if dom_gender=='Male' else '♀'} {dom_gender}</span>
                </div>
                {stats_html}
                <div style="margin-top:.6rem;color:#6b7280;font-size:.75rem">
                    Avg Age: <b style="color:{clr}">{seg_df['Age'].mean():.1f}</b>
                </div>
            </div>""", unsafe_allow_html=True)

    # Cluster size bar
    st.markdown('<div class="section-hdr">📊 Cluster Size Distribution</div>', unsafe_allow_html=True)
    counts = df_clustered["Cluster"].value_counts().sort_index()
    fig, ax = styled_fig(9, 3.5)
    bars = ax.bar(
        [f"C{i}" for i in counts.index],
        counts.values,
        color=[PALETTE[i % len(PALETTE)] for i in counts.index],
        edgecolor="none", width=0.55
    )
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+1, str(int(h)),
                ha='center', va='bottom', color=TEXT_COL, fontsize=9)
    ax.set_xlabel("Cluster", color=TEXT_COL, fontsize=9)
    ax.set_ylabel("No. of Customers", color=TEXT_COL, fontsize=9)
    ax.set_title("Customers per Cluster", color="#d1d5db", fontsize=11, pad=8)
    plt.tight_layout()
    st.pyplot(fig); plt.close()


# ══════════════════ TAB 2 — ELBOW & SILHOUETTE ════════════════════════════════
with tab2:
    st.markdown('<div class="section-hdr">📐 Optimal K Selection</div>', unsafe_allow_html=True)
    col_e, col_s = st.columns(2)

    with col_e:
        fig, ax = styled_fig(6, 4.5)
        ax.plot(ks, inertias, marker='o', color="#a78bfa", linewidth=2.5,
                markersize=7, markerfacecolor="#f472b6", markeredgewidth=0)
        ax.axvline(k, color="#f472b6", linewidth=1.5, linestyle="--", alpha=0.7,
                   label=f"Selected K={k}")
        ax.fill_between(ks, inertias, alpha=0.08, color="#a78bfa")
        ax.set_xlabel("Number of Clusters (K)", color=TEXT_COL, fontsize=9)
        ax.set_ylabel("Inertia (WCSS)", color=TEXT_COL, fontsize=9)
        ax.set_title("🪄 Elbow Method", color="#d1d5db", fontsize=11, pad=8)
        ax.legend(facecolor=DARK_BG, edgecolor="#a78bfa40", labelcolor=TEXT_COL, fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_s:
        fig, ax = styled_fig(6, 4.5)
        colors_sil = ["#f472b6" if ki==k else "#a78bfa" for ki in ks]
        ax.bar(ks, sil_scores, color=colors_sil, edgecolor="none", width=0.6)
        ax.axhline(sil, color="#fb923c", linewidth=1.5, linestyle="--",
                   label=f"K={k}: {sil:.3f}")
        ax.set_xlabel("Number of Clusters (K)", color=TEXT_COL, fontsize=9)
        ax.set_ylabel("Silhouette Score", color=TEXT_COL, fontsize=9)
        ax.set_title("📏 Silhouette Score", color="#d1d5db", fontsize=11, pad=8)
        ax.legend(facecolor=DARK_BG, edgecolor="#a78bfa40", labelcolor=TEXT_COL, fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown(f"""
    <div class="info-box">
    📌 <b>Interpretation:</b> Elbow method shows K=<b>{ks[np.argmin(np.diff(np.diff(inertias)))+1]}</b>
    as a natural bend. Silhouette score for K=<b>{k}</b> is
    <b>{sil:.3f}</b> — closer to 1.0 is better. Values above 0.5 indicate good separation.
    </div>""", unsafe_allow_html=True)


# ══════════════════ TAB 3 — CLUSTER MAPS ══════════════════════════════════════
with tab3:
    st.markdown('<div class="section-hdr">🗺️ Cluster Visualizations</div>', unsafe_allow_html=True)
    col_m1, col_m2 = st.columns(2)

    # ── Primary scatter (first 2 features)
    with col_m1:
        fx, fy = features[0], features[1]
        fig, ax = styled_fig(6.5, 5)
        for i in range(k):
            seg = df_clustered[df_clustered["Cluster"]==i]
            clr = PALETTE[i % len(PALETTE)]
            ax.scatter(seg[fx], seg[fy], c=clr, alpha=0.75, s=40, edgecolors="none", label=f"C{i}")
        # plot centroids
        for i, c in enumerate(centers):
            ax.scatter(c[0], c[1], marker="*", s=260, c=PALETTE[i % len(PALETTE)],
                       edgecolors="white", linewidths=0.8, zorder=5)
        ax.set_xlabel(fx, color=TEXT_COL, fontsize=9)
        ax.set_ylabel(fy, color=TEXT_COL, fontsize=9)
        ax.set_title(f"{fx} vs {fy}", color="#d1d5db", fontsize=11, pad=8)
        ax.legend(facecolor=DARK_BG, edgecolor="#a78bfa40", labelcolor=TEXT_COL,
                  fontsize=8, markerscale=1.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── PCA 2D projection
    with col_m2:
        fig, ax = styled_fig(6.5, 5)
        for i in range(k):
            seg = df_clustered[df_clustered["Cluster"]==i]
            clr = PALETTE[i % len(PALETTE)]
            ax.scatter(seg["PCA_1"], seg["PCA_2"], c=clr, alpha=0.75, s=40,
                       edgecolors="none", label=f"C{i}")
        ax.set_xlabel("PCA Component 1", color=TEXT_COL, fontsize=9)
        ax.set_ylabel("PCA Component 2", color=TEXT_COL, fontsize=9)
        ax.set_title("PCA 2D Projection", color="#d1d5db", fontsize=11, pad=8)
        ax.legend(facecolor=DARK_BG, edgecolor="#a78bfa40", labelcolor=TEXT_COL,
                  fontsize=8, markerscale=1.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── Income vs Spending (always shown for context)
    st.markdown('<div class="section-hdr">💰 Income vs Spending Score by Cluster</div>',
                unsafe_allow_html=True)
    col_m3, col_m4 = st.columns(2)

    with col_m3:
        fig, ax = styled_fig(6.5, 4.8)
        for i in range(k):
            seg = df_clustered[df_clustered["Cluster"]==i]
            clr = PALETTE[i % len(PALETTE)]
            ax.scatter(seg["Annual Income (k$)"], seg["Spending Score (1-100)"],
                       c=clr, alpha=0.7, s=40, edgecolors="none")
            # centroid marker
            ax.scatter(seg["Annual Income (k$)"].mean(), seg["Spending Score (1-100)"].mean(),
                       marker="*", s=220, c=clr, edgecolors="white", linewidths=0.7, zorder=5)
        ax.set_xlabel("Annual Income (k$)", color=TEXT_COL, fontsize=9)
        ax.set_ylabel("Spending Score", color=TEXT_COL, fontsize=9)
        ax.set_title("Income vs Spending", color="#d1d5db", fontsize=11, pad=8)
        handles = [mpatches.Patch(color=PALETTE[i % len(PALETTE)], label=f"C{i}") for i in range(k)]
        ax.legend(handles=handles, facecolor=DARK_BG, edgecolor="#a78bfa40",
                  labelcolor=TEXT_COL, fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_m4:
        # Age distribution per cluster (violin)
        fig, ax = styled_fig(6.5, 4.8)
        data_by_cluster = [df_clustered[df_clustered["Cluster"]==i]["Age"].values for i in range(k)]
        parts = ax.violinplot(data_by_cluster, positions=range(k),
                              showmeans=True, showmedians=False)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(PALETTE[i % len(PALETTE)])
            pc.set_edgecolor("none"); pc.set_alpha(0.75)
        parts["cmeans"].set_color("#ffffff80")
        parts["cbars"].set_color("#ffffff30")
        parts["cmins"].set_color("#ffffff30")
        parts["cmaxes"].set_color("#ffffff30")
        ax.set_xticks(range(k))
        ax.set_xticklabels([f"C{i}" for i in range(k)], color=TEXT_COL, fontsize=9)
        ax.set_ylabel("Age", color=TEXT_COL, fontsize=9)
        ax.set_title("Age Distribution by Cluster", color="#d1d5db", fontsize=11, pad=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── Heatmap of cluster means
    st.markdown('<div class="section-hdr">🔥 Cluster Feature Heatmap</div>', unsafe_allow_html=True)
    cluster_means = df_clustered.groupby("Cluster")[["Age","Annual Income (k$)","Spending Score (1-100)"]].mean()
    fig, ax = styled_fig(9, 3.2)
    sns.heatmap(cluster_means.T, annot=True, fmt=".1f",
                cmap="RdPu", ax=ax,
                linewidths=0.5, linecolor=DARK_BG,
                annot_kws={"size":9, "color":"white"},
                cbar_kws={"shrink":0.7})
    ax.tick_params(colors=TEXT_COL, labelsize=9)
    ax.set_xlabel("Cluster", color=TEXT_COL)
    ax.set_ylabel("")
    ax.set_title("Feature Averages per Cluster", color="#d1d5db", fontsize=11, pad=8)
    plt.tight_layout(); st.pyplot(fig); plt.close()


# ══════════════════ TAB 4 — DATA EXPLORER ════════════════════════════════════
with tab4:
    st.markdown('<div class="section-hdr">📋 Clustered Dataset</div>', unsafe_allow_html=True)

    # filter by cluster
    filter_cls = st.multiselect("Filter by Cluster", list(range(k)),
                                default=list(range(k)),
                                format_func=lambda x: f"Cluster {x}")
    show_df = df_clustered[df_clustered["Cluster"].isin(filter_cls)][
        ["CustomerID","Gender","Age","Annual Income (k$)","Spending Score (1-100)","Cluster"]
    ]
    st.dataframe(
        show_df.style.background_gradient(subset=["Annual Income (k$)","Spending Score (1-100)"],
                                           cmap="RdPu")
                     .map(lambda x: f"color: {PALETTE[x % len(PALETTE)]};font-weight:700"
                               if isinstance(x, (int, np.integer)) else "",
                               subset=["Cluster"]),
        use_container_width=True, height=420
    )

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("**📊 Descriptive Stats**")
        st.dataframe(df_clustered[features+["Age"]].describe().round(2), use_container_width=True)
    with col_s2:
        st.markdown("**📌 Cluster Counts**")
        cnt = df_clustered["Cluster"].value_counts().sort_index().reset_index()
        cnt.columns = ["Cluster","Count"]
        cnt["% Share"] = (cnt["Count"]/cnt["Count"].sum()*100).round(1)
        st.dataframe(cnt, use_container_width=True, height=230)


# ══════════════════ TAB 5 — HOW IT WORKS ════════════════════════════════════
with tab5:
    st.markdown('<div class="section-hdr">ℹ️ K-Means Clustering Explained</div>', unsafe_allow_html=True)
    col_h1, col_h2 = st.columns(2)

    with col_h1:
        st.markdown("""
**🧠 What is K-Means?**

K-Means is an unsupervised ML algorithm that groups data points into **K clusters** by minimising
the within-cluster sum of squares (inertia/WCSS).

**🔄 Algorithm Steps**
1. Randomly initialise **K centroids**
2. Assign each point to the **nearest centroid** (Euclidean distance)
3. Recompute centroids as the **mean** of each cluster
4. Repeat 2-3 until centroids converge

**📦 Features Used**
| Feature | Description |
|---------|-------------|
| `Annual Income (k$)` | Yearly income in thousands |
| `Spending Score (1-100)` | Mall-assigned spending behaviour |
| `Age` | Customer age (optional) |
        """)

    with col_h2:
        st.markdown("""
**📏 Evaluation Metrics**

| Metric | Meaning |
|--------|---------|
| **Silhouette Score** | Cluster tightness & separation (0–1) |
| **Inertia (WCSS)** | Sum of squared distances to centroid |
| **Elbow Method** | Find K where inertia drop slows down |

**🎨 Typical Customer Segments**
- 💎 **High Income, High Spend** → VIP / Champions
- 💸 **Low Income, High Spend** → Impulse Buyers
- 💰 **High Income, Low Spend** → Careful Savers
- 🌱 **Mid Income, Mid Spend** → Potential Loyalists
- 😴 **Low Income, Low Spend** → At-Risk / Churn

**🔗 Real Kaggle Dataset**

Download `Mall_Customers.csv` from [Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
and replace `generate_data()`:
```python
df = pd.read_csv("Mall_Customers.csv")
```
        """)

    st.markdown(f"""
    <div class="info-box">
    ✅ <b>Current run:</b> K=<b>{k}</b> clusters on <b>{len(df_filtered)}</b> customers
    using features: <b>{', '.join(features)}</b>.
    Silhouette = <b>{sil:.3f}</b> | Inertia = <b>{inertia:,.0f}</b>
    </div>""", unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border-color:rgba(167,139,250,0.15);margin-top:3rem">
<p style="text-align:center;color:#374151;font-size:0.8rem">
🛍️ Customer Segmentation · K-Means Clustering · Streamlit + scikit-learn
</p>""", unsafe_allow_html=True)