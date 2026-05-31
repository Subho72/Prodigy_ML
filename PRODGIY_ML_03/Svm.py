import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
from pathlib import Path
import cv2
from PIL import Image
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_curve, auc)
from sklearn.decomposition import PCA
import io, os, warnings, time
warnings.filterwarnings("ignore")

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🐱🐶 Cats vs Dogs SVM",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Fira+Code:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.stApp { background: #080b14; }
.block-container { padding: 1.8rem 2.2rem 3rem; }

h1 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 2.7rem !important;
    background: linear-gradient(135deg, #38bdf8, #818cf8, #fb7185);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-align: center; margin-bottom: 0 !important;
}
.subtitle {
    text-align: center; color: #475569;
    font-size: 0.92rem; letter-spacing: .07em; margin-bottom: 2rem;
}
.kpi-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(56,189,248,0.18);
    border-radius: 16px; padding: 1.2rem 1.4rem;
    text-align: center; transition: transform .2s, border-color .2s;
}
.kpi-card:hover { transform: translateY(-3px); border-color: rgba(56,189,248,.45); }
.kpi-val  { font-size: 2rem; font-weight: 700; font-family:'Space Grotesk',sans-serif; }
.kpi-lbl  { font-size: .72rem; color: #64748b; text-transform:uppercase; letter-spacing:.1em; margin-top:.25rem; }

.pred-box {
    border-radius: 20px; padding: 1.8rem 2rem;
    text-align: center; margin-top: 1rem;
}
.pred-val { font-size: 2.4rem; font-weight: 700; font-family:'Space Grotesk',sans-serif; }
.pred-conf { font-size: .88rem; color: #94a3b8; margin-top: .4rem; }

.section-hdr {
    font-size: 1.2rem; font-weight: 700; color: #38bdf8;
    border-bottom: 1px solid rgba(56,189,248,.2);
    padding-bottom: .4rem; margin-bottom: 1.2rem;
}
.badge {
    display:inline-block; border-radius:6px;
    padding:.25rem .75rem; font-size:.78rem;
    font-weight:600; margin:.2rem;
}
.info-box {
    background: rgba(56,189,248,.06);
    border-left: 3px solid #38bdf8;
    border-radius: 0 12px 12px 0;
    padding: .9rem 1.2rem; margin:.6rem 0;
    font-size:.87rem; color:#cbd5e1;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#080b14,#0e1220) !important;
    border-right: 1px solid rgba(56,189,248,.12);
}
.stButton > button {
    background: linear-gradient(135deg,#38bdf8,#818cf8) !important;
    color: #080b14 !important; border: none !important;
    border-radius: 12px !important; font-weight: 700 !important;
    width: 100%; padding: .65rem !important; transition: all .3s !important;
}
.stButton > button:hover {
    transform: scale(1.03) !important;
    box-shadow: 0 6px 28px rgba(56,189,248,.38) !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE   = 64
N_SAMPLES  = 200        # 150 cats + 150 dogs (synthetic)
DARK_BG    = "#080b14"
CARD_BG    = "#0e1220"
GRID_COL   = "#1e293b"
TEXT_COL   = "#64748b"
CAT_CLR    = "#fb7185"
DOG_CLR    = "#38bdf8"

# ─── Feature Extraction (HOG-inspired manual features) ───────────────────────
def extract_features(img_array: np.ndarray) -> np.ndarray:
    """Extract a rich feature vector from an image array (H,W,3) uint8."""
    img = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)

    features = []

    # 1. Pixel intensities (flattened, downsampled 32x32)
    small = cv2.resize(gray, (32, 32)).flatten() / 255.0
    features.extend(small)

    # 2. Colour histogram (R,G,B — 16 bins each)
    for ch in range(3):
        hist, _ = np.histogram(img[:,:,ch], bins=16, range=(0,256))
        features.extend(hist / hist.sum())

    # 3. Gradient magnitude stats (mean, std, max)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    features.extend([mag.mean()/255, mag.std()/255, mag.max()/255])

    # 4. Texture: LBP-approximation (local binary pattern mean over blocks)
    h, w = gray.shape
    lbp_vals = []
    for r in range(0, h-2, 8):
        for c in range(0, w-2, 8):
            patch = gray[r:r+3, c:c+3]
            center = patch[1,1]
            code = (patch.flatten() >= center).astype(int)
            lbp_vals.append(code.mean())
    features.extend(lbp_vals[:64])

    # 5. Mean & std per quadrant (spatial layout)
    for row in range(4):
        for col in range(4):
            quad = gray[row*16:(row+1)*16, col*16:(col+1)*16]
            features.extend([quad.mean()/255, quad.std()/255])

    return np.array(features, dtype=np.float32)


# ─── Synthetic image generation (realistic noise-based) ──────────────────────
def make_cat_image(seed):
    rng = np.random.RandomState(seed)
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    # Warm beige/grey body
    base = rng.randint(150, 220)
    img[:,:,0] = np.clip(base + rng.randint(-20,20,(IMG_SIZE,IMG_SIZE)), 0, 255)
    img[:,:,1] = np.clip(base - 20 + rng.randint(-15,15,(IMG_SIZE,IMG_SIZE)), 0, 255)
    img[:,:,2] = np.clip(base - 40 + rng.randint(-10,10,(IMG_SIZE,IMG_SIZE)), 0, 255)
    # Pointy ears (triangles top)
    for ex in [IMG_SIZE//4, 3*IMG_SIZE//4]:
        pts = np.array([[ex-10,5],[ex+10,5],[ex,0]], np.int32)
        cv2.fillPoly(img, [pts], (rng.randint(80,150), rng.randint(60,110), rng.randint(40,90)))
    # Eyes (round)
    for ex in [IMG_SIZE//3, 2*IMG_SIZE//3]:
        cv2.circle(img, (ex, IMG_SIZE//3), 5, (20,20,20), -1)
        cv2.circle(img, (ex, IMG_SIZE//3), 2, (80,200,120), -1)
    # Nose (small triangle)
    cv2.circle(img, (IMG_SIZE//2, IMG_SIZE//2), 3, (200, 100, 120), -1)
    # Whiskers
    for side, color in [(-1,(180,180,180)),(1,(180,180,180))]:
        for dy in [-3,0,3]:
            x1 = IMG_SIZE//2 + side*3
            x2 = IMG_SIZE//2 + side*22
            cv2.line(img,(x1,IMG_SIZE//2+dy),(x2,IMG_SIZE//2+dy),color,1)
    img = cv2.GaussianBlur(img,(3,3),0)
    return img

def make_dog_image(seed):
    rng = np.random.RandomState(seed)
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    # Warm brown/golden body
    base = rng.randint(130, 200)
    img[:,:,0] = np.clip(base + rng.randint(-15,15,(IMG_SIZE,IMG_SIZE)), 0, 255)
    img[:,:,1] = np.clip(base - 60 + rng.randint(-15,15,(IMG_SIZE,IMG_SIZE)), 0, 255)
    img[:,:,2] = np.clip(20 + rng.randint(-10,10,(IMG_SIZE,IMG_SIZE)), 0, 255)
    # Floppy ears (rectangles on sides)
    for ex, ex2 in [(0, IMG_SIZE//5), (4*IMG_SIZE//5, IMG_SIZE)]:
        cv2.rectangle(img, (ex,10),(ex2,IMG_SIZE//2),
                      (rng.randint(80,140), rng.randint(30,70), 0), -1)
    # Eyes (bigger, rounder)
    for ex in [IMG_SIZE//3, 2*IMG_SIZE//3]:
        cv2.circle(img, (ex, IMG_SIZE//3), 7, (30,20,10), -1)
        cv2.circle(img, (ex, IMG_SIZE//3), 3, (100,60,20), -1)
        cv2.circle(img, (ex+2, IMG_SIZE//3-2), 1, (255,255,255), -1)
    # Snout (big rounded rectangle)
    cv2.ellipse(img, (IMG_SIZE//2, IMG_SIZE//2+5), (14,10), 0, 0, 360,
                (int(base*0.9), int((base-60)*0.9), 0), -1)
    # Nose (big black oval)
    cv2.ellipse(img, (IMG_SIZE//2, IMG_SIZE//2-2), (7,5), 0, 0, 360, (20,15,10), -1)
    img = cv2.GaussianBlur(img,(3,3),0)
    return img


# ─── Data & Model Cache ───────────────────────────────────────────────────────
DATASET_PATH = r"C:\Users\user\Desktop\PRODGIY_ML_03\train"

def build_dataset():
    X = []
    y = []
    raw_imgs = []

    for folder in ["cats", "dogs"]:

        folder_path = os.path.join(DATASET_PATH, folder)

        for img_name in os.listdir(folder_path)[:1000]:

            img_path = os.path.join(folder_path, img_name)

            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            raw_imgs.append(img)

            features = extract_features(img)

            X.append(features)

            if folder == "cats":
                y.append(0)
            else:
                y.append(1)

    return np.array(X), np.array(y), raw_imgs

@st.cache_resource
def train_svm(kernel, C, gamma):

    X, y, raw_imgs = build_dataset()

    if len(X) == 0:
        st.error("Dataset is empty.")
        st.stop()

    scaler = StandardScaler()

    X_sc = scaler.fit_transform(X)

    pca = PCA(
        n_components=min(80, X_sc.shape[0]-1, X_sc.shape[1]),
        random_state=42
    )

    X_pca = pca.fit_transform(X_sc)

    X_train, X_test, y_train, y_test = train_test_split(
        X_pca,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    svm = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        probability=True,
        random_state=42
    )

    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)
    y_prob = svm.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    report = classification_report(
        y_test,
        y_pred,
        target_names=["Cat", "Dog"],
        output_dict=True
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    cv_scores = cross_val_score(
        svm,
        X_pca,
        y,
        cv=cv,
        scoring="accuracy"
    )

    return (
        svm,
        scaler,
        pca,
        X_test,
        y_test,
        y_pred,
        y_prob,
        acc,
        cm,
        report,
        cv_scores
    )

def predict_image(img_array, svm, scaler, pca):
    feat = extract_features(img_array).reshape(1,-1)
    feat_sc  = scaler.transform(feat)
    feat_pca = pca.transform(feat_sc)
    pred = svm.predict(feat_pca)[0]
    proba = svm.predict_proba(feat_pca)[0]  
    return pred, proba


# ─── Plot helper ─────────────────────────────────────────────────────────────
def styled_fig(w=7, h=5):
    fig, ax = plt.subplots(figsize=(w,h), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=TEXT_COL, labelsize=8)
    for s in ax.spines.values(): s.set_visible(False)
    ax.grid(color=GRID_COL, linewidth=0.6, linestyle="--")
    return fig, ax


# ═══════════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ SVM Settings")
    st.markdown("---")
    kernel = st.selectbox("🔧 Kernel", ["rbf","linear","poly","sigmoid"], index=0)
    C      = st.select_slider("🎚️ Regularisation (C)",
                               options=[0.01,0.1,0.5,1,5,10,50,100], value=10)
    gamma  = st.selectbox("🌐 Gamma", ["scale","auto"], index=0)
    st.markdown("---")
    train_btn = st.button("🚀 Train SVM")
    st.markdown("---")
    st.markdown("""
    <div style='font-size:.78rem;color:#475569'>
    <b style='color:#38bdf8'>Tips</b><br>
    • <b>RBF</b> kernel works best for image data<br>
    • Higher <b>C</b> = less regularisation<br>
    • <b>PCA</b> reduces features to 80 dims<br>
    • <b>5-fold CV</b> used for validation
    </div>""", unsafe_allow_html=True)

# ─── Train model ─────────────────────────────────────────────────────────────
with st.spinner("🐾 Training SVM... please wait"):
    svm, scaler, pca, X_test, y_test, y_pred, y_prob, acc, cm, report, cv_scores = \
        train_svm(kernel, C, gamma)

X, y, raw_imgs = build_dataset()

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("<h1>🐱 Cats vs Dogs 🐶</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">Support Vector Machine · Image Classification · HOG Features + PCA · scikit-learn</p>',
            unsafe_allow_html=True)

# ─── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Model Performance",
    "🗺️ Decision Boundary",
    "🔍 Predict an Image",
    "🖼️ Dataset Samples",
    "ℹ️ How It Works"
])


# ══════════════════ TAB 1 — PERFORMANCE ═══════════════════════════════════════
with tab1:
    c1,c2,c3,c4,c5 = st.columns(5)
    kpis = [
        (c1, f"{acc*100:.1f}%",           "Test Accuracy",     "#38bdf8"),
        (c2, f"{cv_scores.mean()*100:.1f}%","CV Accuracy (5-fold)","#818cf8"),
        (c3, f"{cv_scores.std()*100:.2f}%", "CV Std Dev",       "#fb7185"),
        (c4, f"{report['Cat']['f1-score']:.3f}", "Cat F1-Score","#fb7185"),
        (c5, f"{report['Dog']['f1-score']:.3f}", "Dog F1-Score","#38bdf8"),
    ]
    for col, val, lbl, clr in kpis:
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="border-color:{clr}30">
                <div class="kpi-val" style="color:{clr}">{val}</div>
                <div class="kpi-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_cm, col_roc = st.columns(2)

    # Confusion Matrix
    with col_cm:
        st.markdown('<div class="section-hdr">🧩 Confusion Matrix</div>', unsafe_allow_html=True)
        fig, ax = styled_fig(5.5, 4.5)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Cat","Dog"], yticklabels=["Cat","Dog"],
                    ax=ax, linewidths=.5, linecolor=DARK_BG,
                    annot_kws={"size":14,"color":"white","weight":"bold"},
                    cbar_kws={"shrink":.7})
        ax.set_xlabel("Predicted", color=TEXT_COL, fontsize=9)
        ax.set_ylabel("Actual",    color=TEXT_COL, fontsize=9)
        ax.tick_params(colors=TEXT_COL, labelsize=10)
        ax.set_title("Confusion Matrix", color="#e2e8f0", fontsize=11, pad=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # ROC Curve
    with col_roc:
        st.markdown('<div class="section-hdr">📈 ROC Curve</div>', unsafe_allow_html=True)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        fig, ax = styled_fig(5.5, 4.5)
        ax.plot(fpr, tpr, color="#818cf8", lw=2.5, label=f"AUC = {roc_auc:.3f}")
        ax.fill_between(fpr, tpr, alpha=0.1, color="#818cf8")
        ax.plot([0,1],[0,1],"--", color=TEXT_COL, lw=1.2, label="Random")
        ax.set_xlabel("False Positive Rate", color=TEXT_COL, fontsize=9)
        ax.set_ylabel("True Positive Rate",  color=TEXT_COL, fontsize=9)
        ax.set_title("ROC Curve", color="#e2e8f0", fontsize=11, pad=8)
        ax.legend(facecolor=DARK_BG, edgecolor="#38bdf840", labelcolor=TEXT_COL, fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    col_cv, col_rep = st.columns(2)

    # CV Scores bar
    with col_cv:
        st.markdown('<div class="section-hdr">🔄 Cross-Validation Scores</div>', unsafe_allow_html=True)
        fig, ax = styled_fig(5.5, 3.8)
        fold_colors = ["#38bdf8","#818cf8","#fb7185","#34d399","#fb923c"]
        bars = ax.bar(range(1,6), cv_scores*100, color=fold_colors,
                      edgecolor="none", width=0.55)
        ax.axhline(cv_scores.mean()*100, color="#fbbf24", lw=1.5,
                   linestyle="--", label=f"Mean={cv_scores.mean()*100:.1f}%")
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+.4,
                    f"{b.get_height():.1f}%", ha="center", va="bottom",
                    color=TEXT_COL, fontsize=8)
        ax.set_xlabel("Fold", color=TEXT_COL, fontsize=9)
        ax.set_ylabel("Accuracy (%)", color=TEXT_COL, fontsize=9)
        ax.set_title("5-Fold CV Results", color="#e2e8f0", fontsize=11, pad=8)
        ax.legend(facecolor=DARK_BG, edgecolor="#38bdf840", labelcolor=TEXT_COL, fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Classification Report table
    with col_rep:
        st.markdown('<div class="section-hdr">📋 Classification Report</div>', unsafe_allow_html=True)
        rep_df = pd.DataFrame(report).T.round(3)
        rep_df = rep_df.loc[["Cat","Dog","accuracy","macro avg","weighted avg"]]
        st.dataframe(
            rep_df.style.background_gradient(subset=["precision","recall","f1-score"], cmap="Blues")
                        .format({"precision":"{:.3f}","recall":"{:.3f}",
                                 "f1-score":"{:.3f}","support":"{:.0f}"}),
            use_container_width=True, height=230
        )


# ══════════════════ TAB 2 — DECISION BOUNDARY ════════════════════════════════
with tab2:
    st.markdown('<div class="section-hdr">🗺️ PCA Decision Space</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    📌 The plot below shows the SVM decision boundary in 2D PCA space.
    Real classification uses <b>80 PCA components</b>; this view projects onto the first 2 for visualisation.
    </div>""", unsafe_allow_html=True)

    col_db, col_sv = st.columns([3,2])
    with col_db:
        # Refit 2D for visualisation only
        X_sc_vis  = scaler.transform(X)
        X_2d      = pca.transform(X_sc_vis)[:,:2]
        import numpy as np

        print("Unique Classes:", np.unique(y))
        print("Class Count:", len(np.unique(y)))
        
        if len(np.unique(y)) < 2:
            st.error(f"Dataset contains only one class: {np.unique(y)}")
            st.stop()

        svm_vis = SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=42)
        svm_vis.fit(X_2d, y)

        x_min, x_max = X_2d[:,0].min()-1, X_2d[:,0].max()+1
        y_min, y_max = X_2d[:,1].min()-1, X_2d[:,1].max()+1
        xx, yy = np.meshgrid(np.linspace(x_min,x_max,200),
                             np.linspace(y_min,y_max,200))
        Z = svm_vis.predict_proba(np.c_[xx.ravel(),yy.ravel()])[:,1].reshape(xx.shape)

        fig, ax = styled_fig(7, 5.5)
        ax.contourf(xx, yy, Z, levels=50, cmap="coolwarm", alpha=0.6)
        ax.contour(xx, yy, Z, levels=[0.5], colors="white", linewidths=1.5, linestyles="--")
        cats = X_2d[y==0]; dogs = X_2d[y==1]
        ax.scatter(cats[:,0], cats[:,1], c=CAT_CLR, s=30, alpha=0.8,
                   edgecolors="none", label="🐱 Cat")
        ax.scatter(dogs[:,0], dogs[:,1], c=DOG_CLR, s=30, alpha=0.8,
                   edgecolors="none", label="🐶 Dog")
        # Support vectors
        sv = svm_vis.support_vectors_
        ax.scatter(sv[:,0], sv[:,1], s=120, edgecolors="yellow",
                   facecolors="none", linewidths=1.2, label="Support Vectors")
        ax.set_xlabel("PCA Component 1", color=TEXT_COL, fontsize=9)
        ax.set_ylabel("PCA Component 2", color=TEXT_COL, fontsize=9)
        ax.set_title(f"SVM Decision Boundary ({kernel} kernel)", color="#e2e8f0", fontsize=11, pad=8)
        ax.legend(facecolor=DARK_BG, edgecolor="#38bdf840", labelcolor=TEXT_COL, fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_sv:
        st.markdown('<div class="section-hdr">⚙️ SVM Info</div>', unsafe_allow_html=True)
        n_sv_total = len(svm_vis.support_vectors_)
        n_sv_cat   = svm_vis.n_support_[0]
        n_sv_dog   = svm_vis.n_support_[1]

        st.markdown(f"""
        <div style="background:rgba(255,255,255,.03);border-radius:14px;padding:1.2rem">
            <div style="margin-bottom:.8rem">
                <span style="color:#64748b;font-size:.75rem">KERNEL</span><br>
                <span style="color:#38bdf8;font-weight:700;font-size:1.1rem">{kernel.upper()}</span>
            </div>
            <div style="margin-bottom:.8rem">
                <span style="color:#64748b;font-size:.75rem">REGULARISATION (C)</span><br>
                <span style="color:#818cf8;font-weight:700;font-size:1.1rem">{C}</span>
            </div>
            <div style="margin-bottom:.8rem">
                <span style="color:#64748b;font-size:.75rem">SUPPORT VECTORS</span><br>
                <span style="color:#fbbf24;font-weight:700;font-size:1.1rem">{n_sv_total}</span>
            </div>
            <div style="margin-bottom:.8rem">
                <span style="color:#64748b;font-size:.75rem">CAT SVs / DOG SVs</span><br>
                <span style="color:{CAT_CLR};font-weight:700">{n_sv_cat}</span>
                <span style="color:#475569"> / </span>
                <span style="color:{DOG_CLR};font-weight:700">{n_sv_dog}</span>
            </div>
            <div>
                <span style="color:#64748b;font-size:.75rem">GAMMA</span><br>
                <span style="color:#34d399;font-weight:700;font-size:1.1rem">{gamma}</span>
            </div>
        </div>""", unsafe_allow_html=True)

        # PCA variance
        st.markdown('<br><div class="section-hdr">📐 PCA Variance (Top 15)</div>', unsafe_allow_html=True)
        fig2, ax2 = styled_fig(5, 3.5)
        ev = pca.explained_variance_ratio_[:15] * 100
        ax2.bar(range(1,16), ev, color="#818cf8", edgecolor="none", width=0.7)
        ax2.set_xlabel("Component", color=TEXT_COL, fontsize=9)
        ax2.set_ylabel("Variance Explained (%)", color=TEXT_COL, fontsize=9)
        ax2.set_title("PCA Explained Variance", color="#e2e8f0", fontsize=10, pad=6)
        plt.tight_layout(); st.pyplot(fig2); plt.close()


# ══════════════════ TAB 3 — PREDICT ══════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-hdr">🔍 Classify Your Own Image</div>', unsafe_allow_html=True)

    col_up, col_out = st.columns([1,1])
    with col_up:
        uploaded = st.file_uploader(
            "Upload a cat or dog image (JPG / PNG)",
            type=["jpg","jpeg","png"]
        )
        st.markdown("---")
        st.markdown("**Or try a sample from the dataset:**")
        sample_type = st.radio("Sample", ["🐱 Cat sample","🐶 Dog sample"], horizontal=True)
        use_sample  = st.button("🎲 Use Sample Image")

    with col_out:
        img_to_predict = None

        if uploaded is not None:
            pil_img = Image.open(uploaded).convert("RGB")
            img_to_predict = np.array(pil_img)

        elif use_sample:
            if "Cat" in sample_type:
                idx = np.random.randint(0, N_SAMPLES//2)
                img_to_predict = raw_imgs[idx]
            else:
                idx = np.random.randint(N_SAMPLES//2, N_SAMPLES)
                img_to_predict = raw_imgs[idx]

        if img_to_predict is not None:
            pred, proba = predict_image(img_to_predict, svm, scaler, pca)
            label   = "Cat 🐱" if pred == 0 else "Dog 🐶"
            conf    = proba[pred] * 100
            clr     = CAT_CLR if pred == 0 else DOG_CLR
            bg_clr  = f"rgba({251 if pred==0 else 56},{113 if pred==0 else 189},{133 if pred==0 else 248},0.08)"

            # Show image
            fig_img, ax_img = plt.subplots(1,1,figsize=(4,4), facecolor=DARK_BG)
            ax_img.imshow(img_to_predict)
            ax_img.axis("off")
            ax_img.set_title("Input Image", color="#e2e8f0", fontsize=10)
            plt.tight_layout(); st.pyplot(fig_img); plt.close()

            st.markdown(f"""
            <div class="pred-box" style="background:{bg_clr};border:2px solid {clr}">
                <div class="pred-val" style="color:{clr}">{label}</div>
                <div class="pred-conf">Confidence: <b>{conf:.1f}%</b></div>
                <div style="margin-top:.8rem;display:flex;justify-content:center;gap:1rem">
                    <span>🐱 Cat: <b style="color:{CAT_CLR}">{proba[0]*100:.1f}%</b></span>
                    <span>🐶 Dog: <b style="color:{DOG_CLR}">{proba[1]*100:.1f}%</b></span>
                </div>
            </div>""", unsafe_allow_html=True)

            # Prob bar
            fig_bar, ax_bar = styled_fig(5, 2)
            ax_bar.barh(["Cat 🐱","Dog 🐶"], [proba[0]*100, proba[1]*100],
                        color=[CAT_CLR, DOG_CLR], edgecolor="none", height=0.45)
            ax_bar.set_xlim(0,100)
            ax_bar.set_xlabel("Probability (%)", color=TEXT_COL, fontsize=9)
            ax_bar.set_title("Prediction Probability", color="#e2e8f0", fontsize=10, pad=6)
            ax_bar.tick_params(colors=TEXT_COL, labelsize=9)
            plt.tight_layout(); st.pyplot(fig_bar); plt.close()
        else:
            st.markdown("""
            <div style="text-align:center;color:#475569;padding:3rem 0;font-size:1.1rem">
                📁 Upload an image or click "Use Sample Image"
            </div>""", unsafe_allow_html=True)


# ══════════════════ TAB 4 — DATASET SAMPLES ══════════════════════════════════
with tab4:
    st.markdown('<div class="section-hdr">🖼️ Synthetic Dataset Samples</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    These are <b>procedurally generated</b> cat and dog images for demo purposes.
    Plug in the real <b>Kaggle Dogs vs Cats</b> dataset to get production-grade results.
    </div>""", unsafe_allow_html=True)

    col_cats, col_dogs = st.columns(2)
    with col_cats:
        st.markdown(f"**🐱 Cat Samples** — <span style='color:{CAT_CLR}'>Class 0</span>",
                    unsafe_allow_html=True)
        fig, axes = plt.subplots(3, 4, figsize=(8,6), facecolor=DARK_BG)
        fig.patch.set_facecolor(DARK_BG)
        for i, ax in enumerate(axes.flat):
            idx = np.random.randint(0, N_SAMPLES//2)
            ax.imshow(raw_imgs[idx])
            ax.axis("off")
            ax.set_title("Cat", color=CAT_CLR, fontsize=7, pad=2)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_dogs:
        st.markdown(f"**🐶 Dog Samples** — <span style='color:{DOG_CLR}'>Class 1</span>",
                    unsafe_allow_html=True)
        fig, axes = plt.subplots(3, 4, figsize=(8,6), facecolor=DARK_BG)
        fig.patch.set_facecolor(DARK_BG)
        for i, ax in enumerate(axes.flat):
            idx = np.random.randint(N_SAMPLES//2, N_SAMPLES)
            ax.imshow(raw_imgs[idx])
            ax.axis("off")
            ax.set_title("Dog", color=DOG_CLR, fontsize=7, pad=2)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Feature distribution
    st.markdown('<div class="section-hdr">📊 Feature Distribution (PCA Components)</div>',
                unsafe_allow_html=True)
    X_sc  = scaler.transform(X)
    X_pca_all = pca.transform(X_sc)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), facecolor=DARK_BG)
    for i, ax in enumerate(axes):
        ax.set_facecolor(DARK_BG)
        ax.hist(X_pca_all[y==0, i], bins=25, color=CAT_CLR, alpha=0.7,
                edgecolor="none", label="Cat")
        ax.hist(X_pca_all[y==1, i], bins=25, color=DOG_CLR, alpha=0.7,
                edgecolor="none", label="Dog")
        ax.set_title(f"PCA Component {i+1}", color="#e2e8f0", fontsize=10, pad=6)
        ax.tick_params(colors=TEXT_COL, labelsize=8)
        for s in ax.spines.values(): s.set_visible(False)
        ax.grid(color=GRID_COL, lw=0.5, ls="--")
        ax.legend(facecolor=DARK_BG, edgecolor="none", labelcolor=TEXT_COL, fontsize=8)
    plt.tight_layout(); st.pyplot(fig); plt.close()


# ══════════════════ TAB 5 — HOW IT WORKS ══════════════════════════════════════
with tab5:
    st.markdown('<div class="section-hdr">ℹ️ SVM Image Classification Pipeline</div>',
                unsafe_allow_html=True)
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.markdown("""
**🧠 What is SVM?**

A **Support Vector Machine** finds the optimal hyperplane that maximally separates two classes.
It works by maximising the **margin** between the nearest data points (support vectors) of each class.

**🔄 Full Pipeline**
```
Raw Image (64×64 RGB)
    │
    ▼
Feature Extraction
 ├─ Pixel intensities (32×32 = 1024)
 ├─ Colour histograms (48 bins)
 ├─ Gradient magnitude stats (3)
 ├─ LBP texture approximation (64)
 └─ Spatial quadrant stats (32)
    │
    ▼
StandardScaler  →  PCA (80 dims)
    │
    ▼
SVM Classifier (RBF kernel)
    │
    ▼
Cat 🐱  /  Dog 🐶
```

**🎯 Kernels Explained**

| Kernel | Best For |
|--------|----------|
| `rbf` | Non-linear, most image tasks ✅ |
| `linear` | Linearly separable data |
| `poly` | Polynomial decision boundary |
| `sigmoid` | Neural-network-like mapping |
        """)
    with col_h2:
        st.markdown("""
**📐 Feature Engineering**

Since SVM is not natively spatial, we extract hand-crafted features:

- **Pixel intensities** — raw spatial info at lower resolution
- **Colour histograms** — colour distribution per channel
- **Gradient stats** — edge strength (texture/shape cues)
- **LBP approximation** — local texture patterns
- **Quadrant stats** — coarse spatial layout

**📏 Evaluation**

| Metric | Description |
|--------|-------------|
| **Accuracy** | % correctly classified |
| **Precision** | TP / (TP + FP) |
| **Recall** | TP / (TP + FN) |
| **F1 Score** | Harmonic mean of P & R |
| **AUC-ROC** | Area under ROC curve |
| **5-Fold CV** | Generalisation estimate |

        """)

    st.markdown(f"""
    <div class="info-box">
    ✅ <b>Current model:</b> Kernel=<b>{kernel}</b> | C=<b>{C}</b> |
    Gamma=<b>{gamma}</b> | Test Accuracy=<b>{acc*100:.1f}%</b> |
    CV Mean=<b>{cv_scores.mean()*100:.1f}%</b> ± <b>{cv_scores.std()*100:.2f}%</b>
    </div>""", unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border-color:rgba(56,189,248,.1);margin-top:3rem">
<p style="text-align:center;color:#1e293b;font-size:.8rem">
🐾 Cats vs Dogs · SVM Classifier · Streamlit + scikit-learn + OpenCV
</p>""", unsafe_allow_html=True)