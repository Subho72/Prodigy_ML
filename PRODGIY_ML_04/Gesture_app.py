import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
import os, warnings, time
warnings.filterwarnings("ignore")

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🤚 Hand Gesture Recognition",
    page_icon="🤚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html,body,[class*="css"]{ font-family:'Outfit',sans-serif; }
.stApp{ background:#060810; }
.block-container{ padding:1.8rem 2.2rem 3rem; }

h1{
    font-family:'Outfit',sans-serif !important;
    font-size:2.9rem !important;
    background:linear-gradient(135deg,#22d3ee,#a78bfa,#f472b6,#fb923c);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    text-align:center; margin-bottom:0 !important;
}
.subtitle{
    text-align:center; color:#374151;
    font-size:.92rem; letter-spacing:.07em; margin-bottom:2rem;
}
.kpi-card{
    background:rgba(255,255,255,.03);
    border:1px solid rgba(34,211,238,.15);
    border-radius:16px; padding:1.2rem 1.4rem;
    text-align:center; transition:transform .2s,border-color .2s;
}
.kpi-card:hover{ transform:translateY(-3px); border-color:rgba(34,211,238,.45); }
.kpi-val{ font-size:2rem; font-weight:800; }
.kpi-lbl{ font-size:.72rem; color:#4b5563; text-transform:uppercase; letter-spacing:.1em; margin-top:.25rem; }
.pred-box{
    border-radius:20px; padding:2rem;
    text-align:center; margin-top:1rem;
}
.pred-name{ font-size:2rem; font-weight:800; margin:.4rem 0; }
.pred-conf{ font-size:.88rem; color:#6b7280; }
.section-hdr{
    font-size:1.2rem; font-weight:700; color:#22d3ee;
    border-bottom:1px solid rgba(34,211,238,.2);
    padding-bottom:.4rem; margin-bottom:1.2rem;
}
.info-box{
    background:rgba(34,211,238,.05);
    border-left:3px solid #22d3ee;
    border-radius:0 12px 12px 0;
    padding:.9rem 1.2rem; margin:.6rem 0;
    font-size:.87rem; color:#9ca3af;
}
.warn-box{
    background:rgba(251,146,60,.07);
    border-left:3px solid #fb923c;
    border-radius:0 12px 12px 0;
    padding:.9rem 1.2rem; margin:.6rem 0;
    font-size:.87rem; color:#9ca3af;
}
section[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#060810,#0c0f1a) !important;
    border-right:1px solid rgba(34,211,238,.1);
}
.stButton>button{
    background:linear-gradient(135deg,#22d3ee,#a78bfa) !important;
    color:#060810 !important; border:none !important;
    border-radius:12px !important; font-weight:700 !important;
    width:100%; padding:.65rem !important; transition:all .3s !important;
}
.stButton>button:hover{
    transform:scale(1.03) !important;
    box-shadow:0 6px 28px rgba(34,211,238,.35) !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
# LeapGestRecog exact folder names
GESTURE_FOLDERS = [
    "01_palm", "02_l", "03_fist", "04_fist_moved",
    "05_thumb", "06_index", "07_ok", "08_palm_moved",
    "09_c", "10_down"
]

GESTURES = {
    0:  ("palm",        "✋", "#22d3ee",  "Open palm facing forward"),
    1:  ("l",           "🤙", "#a78bfa",  "L-shape: thumb + index extended"),
    2:  ("fist",        "✊", "#f472b6",  "Closed fist, all fingers curled"),
    3:  ("fist_moved",  "👊", "#fb923c",  "Fist with slight wrist movement"),
    4:  ("thumb",       "👍", "#34d399",  "Thumb up, others curled"),
    5:  ("index",       "☝️", "#fbbf24",  "Index finger pointing up"),
    6:  ("ok",          "👌", "#60a5fa",  "OK sign: thumb-index circle"),
    7:  ("palm_moved",  "🖐️", "#e879f9",  "Palm with lateral displacement"),
    8:  ("c",           "🤌", "#f87171",  "C-shape curl of all fingers"),
    9:  ("down",        "👇", "#4ade80",  "Index pointing downward"),
}

N_CLASSES  = 10
IMG_SIZE   = 64
N_FEATURES = IMG_SIZE * IMG_SIZE  # 4096 raw pixels (grayscale, resized)

DARK_BG  = "#060810"
GRID_COL = "#111827"
TEXT_COL = "#4b5563"

# ─── Feature Extraction from real LeapGestRecog images ───────────────────────
def extract_features(img_gray: np.ndarray) -> np.ndarray:
    """
    Extract a rich feature vector from a grayscale hand image.
    Works with LeapGestRecog IR images (640x240 or any size).
    """
    # 1. Resize to fixed size
    resized = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0

    features = []

    # A) Flattened pixel intensities (4096 features)
    features.extend(resized.flatten())

    # B) HOG-like gradient features (edge info)
    gx  = cv2.Sobel(resized, cv2.CV_32F, 1, 0, ksize=3)
    gy  = cv2.Sobel(resized, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    ang = np.arctan2(gy, gx)

    # Gradient histogram (18 orientation bins)
    hist_mag, _ = np.histogram(ang, bins=18, range=(-np.pi, np.pi),
                               weights=mag)
    features.extend(hist_mag / (hist_mag.sum() + 1e-8))

    # C) Spatial block statistics (4x4 grid → 16 blocks)
    h, w = resized.shape
    bh, bw = h // 4, w // 4
    for r in range(4):
        for c in range(4):
            block = resized[r*bh:(r+1)*bh, c*bw:(c+1)*bw]
            features.extend([block.mean(), block.std()])  # 32 features

    # D) Global stats
    features.extend([
        resized.mean(), resized.std(),
        resized.max(),  resized.min(),
        np.percentile(resized, 25),
        np.percentile(resized, 75),
    ])

    return np.array(features, dtype=np.float32)


# ─── Real dataset loader ─────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_real_dataset(root: str, max_per_class: int = 200):
    """
    Load LeapGestRecog dataset from disk.

    Expected structure:
        root/
          00/  (subject 0)
            01_palm/
            02_l/
            ...
          01/  (subject 1)
            ...

    Default root folder name: leapGestRecog/
    """
    X, y, loaded_paths = [], [], []
    errors = []

    subjects = sorted([
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ])

    for subject in subjects:
        for cls_id, cls_name in enumerate(GESTURE_FOLDERS):
            folder = os.path.join(root, subject, cls_name)
            if not os.path.isdir(folder):
                continue
            count = 0
            for fname in sorted(os.listdir(folder)):
                if count >= max_per_class:
                    break
                if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                fpath = os.path.join(folder, fname)
                img   = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    errors.append(fpath)
                    continue
                feats = extract_features(img)
                X.append(feats)
                y.append(cls_id)
                loaded_paths.append(fpath)
                count += 1

    if len(X) == 0:
        return None, None, [], errors

    return np.array(X, dtype=np.float32), np.array(y), loaded_paths, errors


def load_sample_images(root: str, n_per_class: int = 3):
    """Load a few sample images per class for the gallery tab."""
    samples = {}
    subjects = sorted([
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ])
    for cls_id, cls_name in enumerate(GESTURE_FOLDERS):
        imgs = []
        for subject in subjects:
            folder = os.path.join(root, subject, cls_name)
            if not os.path.isdir(folder):
                continue
            for fname in sorted(os.listdir(folder))[:2]:
                if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                fpath = os.path.join(folder, fname)
                img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    imgs.append(img)
            if len(imgs) >= n_per_class:
                break
        samples[cls_id] = imgs[:n_per_class]
    return samples


@st.cache_resource
def train_model(clf_name: str, root: str, max_per_class: int):
    X, y, paths, errors = load_real_dataset(root, max_per_class)

    if X is None:
        return None

    scaler = StandardScaler()
    n_components = min(100, X.shape[0], X.shape[1])
    pca    = PCA(n_components=n_components, random_state=42)

    X_sc  = scaler.fit_transform(X)
    X_pca = pca.fit_transform(X_sc)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_pca, y, test_size=0.2, random_state=42, stratify=y
    )

    classifiers = {
        "MLP Neural Network":   MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            max_iter=300, random_state=42, early_stopping=True),
        "Random Forest":        RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1),
        "SVM (RBF)":            SVC(
            kernel="rbf", C=10, gamma="scale",
            probability=True, random_state=42),
        "Gradient Boosting":    GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1,
            max_depth=4, random_state=42),
    }
    clf = classifiers[clf_name]
    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te) if hasattr(clf, "predict_proba") else None

    acc  = accuracy_score(y_te, y_pred)
    cm   = confusion_matrix(y_te, y_pred)
    rep  = classification_report(
        y_te, y_pred,
        target_names=[GESTURES[i][0] for i in range(N_CLASSES)],
        output_dict=True
    )

    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_sc  = cross_val_score(clf, X_pca, y, cv=cv, scoring="accuracy")

    pca2d   = PCA(n_components=2, random_state=42)
    X_2d    = pca2d.fit_transform(X_sc)

    return {
        "clf": clf, "scaler": scaler, "pca": pca, "pca2d": pca2d,
        "X": X, "X_sc": X_sc, "X_pca": X_pca, "X_2d": X_2d,
        "X_te": X_te, "y": y, "y_te": y_te, "y_pred": y_pred,
        "acc": acc, "cm": cm, "rep": rep, "cv_sc": cv_sc,
        "paths": paths, "errors": errors,
        "n_samples": len(X), "pca_var": pca.explained_variance_ratio_,
    }


def predict_image(img_gray: np.ndarray, result: dict):
    feat     = extract_features(img_gray).reshape(1, -1)
    feat_sc  = result["scaler"].transform(feat)
    feat_pca = result["pca"].transform(feat_sc)
    pred     = result["clf"].predict(feat_pca)[0]
    if hasattr(result["clf"], "predict_proba"):
        proba = result["clf"].predict_proba(feat_pca)[0]
    else:
        proba = np.zeros(N_CLASSES); proba[pred] = 1.0
    return pred, proba


def styled_fig(w=7, h=5):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=TEXT_COL, labelsize=8)
    for s in ax.spines.values(): s.set_visible(False)
    ax.grid(color=GRID_COL, linewidth=0.5, linestyle="--")
    return fig, ax


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    dataset_path = st.text_input(
        "📂 Dataset Root Path",
        value="leapGestRecog/",
        help="Path to the extracted LeapGestRecog folder (e.g. leapGestRecog/)"
    )
    max_per_class = st.slider(
        "🔢 Max images per class per subject",
        min_value=10, max_value=300, value=50, step=10,
        help="Lower = faster training, Higher = better accuracy"
    )
    clf_name = st.selectbox("🧠 Classifier", [
        "Random Forest",
        "MLP Neural Network",
        "SVM (RBF)",
        "Gradient Boosting",
    ])
    st.markdown("---")
    train_btn = st.button("🚀 Load & Train")
    st.markdown("---")
    st.markdown("""
    <div style='font-size:.78rem;color:#374151;line-height:1.9'>
    <b style='color:#22d3ee'>Dataset structure:</b><br>
    <code style='color:#a78bfa'>leapGestRecog/<br>
    &nbsp;├── 00/<br>
    &nbsp;│&nbsp;&nbsp; ├── 01_palm/<br>
    &nbsp;│&nbsp;&nbsp; ├── 02_l/<br>
    &nbsp;│&nbsp;&nbsp; └── ...<br>
    &nbsp;├── 01/<br>
    &nbsp;└── ...</code>
    </div>""", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("<h1>🤚 Hand Gesture Recognition</h1>", unsafe_allow_html=True)
st.markdown("""<p class="subtitle">
ProDigy Infotech · LeapGestRecog Dataset · 10 Gesture Classes ·
HOG + Pixel Features · PCA + scikit-learn
</p>""", unsafe_allow_html=True)

# ─── Check dataset & load ────────────────────────────────────────────────────
dataset_exists = os.path.isdir(dataset_path)

if not dataset_exists:
    st.markdown(f"""
    <div class="warn-box">
    ⚠️ <b>Dataset not found</b> at <code>{dataset_path}</code><br><br>
    Please download the <b>LeapGestRecog</b> dataset from
    <a href="https://www.kaggle.com/gti-upm/leapgestrecog" target="_blank"
       style="color:#22d3ee">Kaggle</a>,
    extract it, and either:<br>
    • Place the <code>leapGestRecog/</code> folder next to <code>gesture_app.py</code><br>
    • Or enter the full path in the sidebar
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    ### 📥 Quick Setup
    ```bash
    # Install Kaggle CLI
    pip install kaggle

    # Download dataset
    kaggle datasets download -d gti-upm/leapgestrecog
    unzip leapgestrecog.zip -d leapGestRecog/

    # Run app
    streamlit run Gesture_app.py
    ```
    """)
    st.stop()

# ─── Train ───────────────────────────────────────────────────────────────────
with st.spinner(f"🤚 Loading dataset & training {clf_name}... please wait"):
    result = train_model(clf_name, dataset_path, max_per_class)

if result is None:
    st.error("❌ Could not load any images. Check the dataset path and folder structure.")
    st.stop()

# Unpack
clf    = result["clf"]
acc    = result["acc"]
cm     = result["cm"]
rep    = result["rep"]
cv_sc  = result["cv_sc"]
X      = result["X"]
X_sc   = result["X_sc"]
X_2d   = result["X_2d"]
y      = result["y"]
y_te   = result["y_te"]
y_pred = result["y_pred"]

if result["errors"]:
    st.warning(f"⚠️ {len(result['errors'])} images could not be loaded (corrupt/missing).")

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Performance",
    "🖼️ Gesture Gallery",
    "🗺️ Feature Space",
    "🔍 Live Predict",
    "📋 Data Explorer",
    "ℹ️ Pipeline",
])


# ══ TAB 1 — PERFORMANCE ══════════════════════════════════════════════════════
with tab1:
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (c1, f"{acc*100:.1f}%",           "Test Accuracy",        "#22d3ee"),
        (c2, f"{cv_sc.mean()*100:.1f}%",  "CV Accuracy (5-fold)", "#a78bfa"),
        (c3, f"{cv_sc.std()*100:.2f}%",   "CV Std Dev",           "#f472b6"),
        (c4, f"{result['n_samples']:,}",  "Total Images Loaded",  "#34d399"),
        (c5, f"{N_CLASSES}",              "Gesture Classes",       "#fb923c"),
    ]
    for col, val, lbl, clr in kpis:
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="border-color:{clr}25">
                <div class="kpi-val" style="color:{clr}">{val}</div>
                <div class="kpi-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_cm, col_cv = st.columns(2)

    with col_cm:
        st.markdown('<div class="section-hdr">🧩 Confusion Matrix</div>', unsafe_allow_html=True)
        gnames = [f"{GESTURES[i][1]} {GESTURES[i][0]}" for i in range(N_CLASSES)]
        fig, ax = styled_fig(7, 5.5)
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrBr",
                    xticklabels=gnames, yticklabels=gnames, ax=ax,
                    linewidths=.4, linecolor=DARK_BG,
                    annot_kws={"size": 7, "color": "white", "weight": "bold"},
                    cbar_kws={"shrink": .7})
        ax.set_xlabel("Predicted", color=TEXT_COL, fontsize=8)
        ax.set_ylabel("Actual",    color=TEXT_COL, fontsize=8)
        ax.tick_params(colors=TEXT_COL, labelsize=7, rotation=30)
        ax.set_title("10-Class Confusion Matrix", color="#e2e8f0", fontsize=11, pad=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_cv:
        st.markdown('<div class="section-hdr">🔄 5-Fold Cross-Validation</div>', unsafe_allow_html=True)
        cvcolors = ["#22d3ee","#a78bfa","#f472b6","#34d399","#fb923c"]
        fig, ax = styled_fig(6, 5.5)
        bars = ax.bar(range(1, 6), cv_sc * 100, color=cvcolors,
                      edgecolor="none", width=0.6)
        ax.axhline(cv_sc.mean() * 100, color="#fbbf24", lw=1.8, linestyle="--",
                   label=f"Mean = {cv_sc.mean()*100:.2f}%")
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + .3,
                    f"{b.get_height():.1f}%", ha="center", va="bottom",
                    color=TEXT_COL, fontsize=9)
        ax.set_xlabel("Fold", color=TEXT_COL, fontsize=9)
        ax.set_ylabel("Accuracy (%)", color=TEXT_COL, fontsize=9)
        ax.set_title("5-Fold CV Results", color="#e2e8f0", fontsize=11, pad=8)
        ax.legend(facecolor=DARK_BG, edgecolor="#22d3ee30",
                  labelcolor=TEXT_COL, fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Per-class F1 bar
    st.markdown('<div class="section-hdr">📏 Per-Class F1 Score</div>', unsafe_allow_html=True)
    f1s  = [rep[GESTURES[i][0]]["f1-score"] for i in range(N_CLASSES)]
    clrs = [GESTURES[i][2] for i in range(N_CLASSES)]
    lbls = [f"{GESTURES[i][1]} {GESTURES[i][0]}" for i in range(N_CLASSES)]
    fig, ax = styled_fig(11, 3.5)
    bars = ax.bar(lbls, f1s, color=clrs, edgecolor="none", width=0.6)
    ax.axhline(np.mean(f1s), color="#fbbf24", lw=1.5, linestyle="--",
               label=f"Mean = {np.mean(f1s):.3f}")
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + .005,
                f"{b.get_height():.2f}", ha="center", va="bottom",
                color=TEXT_COL, fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("F1 Score", color=TEXT_COL, fontsize=9)
    ax.tick_params(colors=TEXT_COL, labelsize=9, axis="x", rotation=20)
    ax.set_title("F1 Score per Gesture Class", color="#e2e8f0", fontsize=11, pad=8)
    ax.legend(facecolor=DARK_BG, edgecolor="#22d3ee30",
              labelcolor=TEXT_COL, fontsize=9)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Classification report table
    st.markdown('<div class="section-hdr">📋 Classification Report</div>', unsafe_allow_html=True)
    rows = [GESTURES[i][0] for i in range(N_CLASSES)] + ["accuracy","macro avg","weighted avg"]
    rep_df = pd.DataFrame(rep).T.loc[rows].round(3)
    st.dataframe(
        rep_df.style.background_gradient(
            subset=["precision","recall","f1-score"], cmap="YlOrBr"
        ).format({
            "precision": "{:.3f}", "recall": "{:.3f}",
            "f1-score": "{:.3f}",  "support": "{:.0f}"
        }),
        use_container_width=True, height=380
    )


# ══ TAB 2 — GESTURE GALLERY ══════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-hdr">🖼️ Real Dataset Images — LeapGestRecog</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    📌 Real infrared hand images captured by a <b>Leap Motion Controller</b>.
    Each row shows sample images from the actual dataset for each gesture class.
    </div>""", unsafe_allow_html=True)

    with st.spinner("Loading sample images..."):
        samples = load_sample_images(dataset_path, n_per_class=4)

    for cls_id in range(N_CLASSES):
        name, emoji, clr, desc = GESTURES[cls_id]
        imgs = samples.get(cls_id, [])
        if not imgs:
            continue

        st.markdown(f"""
        <div style="color:{clr};font-weight:700;font-size:1rem;
                    margin:.8rem 0 .3rem">
            {emoji} {name.upper()} &nbsp;
            <span style="color:#4b5563;font-weight:400;font-size:.82rem">
            — {desc}
            </span>
        </div>""", unsafe_allow_html=True)

        cols = st.columns(len(imgs))
        for col, img in zip(cols, imgs):
            with col:
                fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5), facecolor=DARK_BG)
                ax.imshow(img, cmap="gray", interpolation="bilinear")
                ax.axis("off")
                # Coloured border effect
                for spine in ax.spines.values():
                    spine.set_edgecolor(clr)
                    spine.set_linewidth(2)
                    spine.set_visible(True)
                plt.tight_layout(pad=0.1)
                st.pyplot(fig); plt.close()

    # Pixel intensity distribution
    st.markdown('<div class="section-hdr">📊 Pixel Intensity Distribution per Gesture</div>',
                unsafe_allow_html=True)
    fig, axes = plt.subplots(2, 5, figsize=(14, 5), facecolor=DARK_BG)
    fig.patch.set_facecolor(DARK_BG)
    for gi, ax in enumerate(axes.flat):
        if gi >= N_CLASSES:
            ax.set_visible(False); continue
        _, emoji, clr, _ = GESTURES[gi]
        mask = y == gi
        if mask.sum() == 0:
            ax.set_visible(False); continue
        pixel_vals = X[mask, :IMG_SIZE*IMG_SIZE].flatten()
        ax.hist(pixel_vals, bins=30, color=clr, alpha=0.8, edgecolor="none")
        ax.set_facecolor(DARK_BG)
        ax.set_title(f"{emoji} {GESTURES[gi][0]}", color=clr, fontsize=8, pad=4)
        ax.tick_params(colors=TEXT_COL, labelsize=6)
        for s in ax.spines.values(): s.set_visible(False)
    plt.suptitle("Pixel Intensity Histograms per Class",
                 color="#e2e8f0", fontsize=11, y=1.01)
    plt.tight_layout(); st.pyplot(fig); plt.close()


# ══ TAB 3 — FEATURE SPACE ════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-hdr">🗺️ 2D PCA Feature Space</div>',
                unsafe_allow_html=True)
    col_pca, col_info = st.columns([3, 2])

    with col_pca:
        fig, ax = styled_fig(8, 6.5)
        for gi in range(N_CLASSES):
            mask = y == gi
            _, emoji, clr, _ = GESTURES[gi]
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=clr, alpha=0.45, s=15, edgecolors="none",
                       label=f"{emoji} {GESTURES[gi][0]}")
        ax.set_xlabel("PCA Component 1", color=TEXT_COL, fontsize=9)
        ax.set_ylabel("PCA Component 2", color=TEXT_COL, fontsize=9)
        ax.set_title("Gesture Clusters in PCA 2D Space",
                     color="#e2e8f0", fontsize=11, pad=8)
        ax.legend(facecolor=DARK_BG, edgecolor="#22d3ee20",
                  labelcolor=TEXT_COL, fontsize=8,
                  markerscale=2, ncol=2, loc="upper right")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_info:
        # PCA explained variance
        st.markdown('<div class="section-hdr">📐 PCA Variance Explained</div>',
                    unsafe_allow_html=True)
        ev  = result["pca_var"][:20] * 100
        cum = np.cumsum(result["pca_var"]) * 100
        fig2, ax2 = styled_fig(5.5, 4)
        ax2.bar(range(1, 21), ev, color="#a78bfa",
                edgecolor="none", width=0.75, alpha=0.85)
        ax2_r = ax2.twinx()
        ax2_r.plot(range(1, 21), cum[:20], color="#22d3ee",
                   lw=2, marker="o", markersize=4)
        ax2_r.set_ylabel("Cumulative %", color="#22d3ee", fontsize=8)
        ax2_r.tick_params(colors="#22d3ee", labelsize=7)
        ax2_r.set_facecolor(DARK_BG)
        ax2.set_xlabel("Component", color=TEXT_COL, fontsize=8)
        ax2.set_ylabel("Variance %", color=TEXT_COL, fontsize=8)
        ax2.set_title("Top 20 PCA Components", color="#e2e8f0", fontsize=10, pad=6)
        plt.tight_layout(); st.pyplot(fig2); plt.close()

        # Inter-gesture distance matrix
        st.markdown('<div class="section-hdr">📏 Inter-Gesture Distances</div>',
                    unsafe_allow_html=True)
        centroids = np.array([
            X_sc[y == gi].mean(axis=0) for gi in range(N_CLASSES)
        ])
        dist_mat  = np.sqrt(
            ((centroids[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
        )
        fig3, ax3 = styled_fig(5.5, 4.5)
        sns.heatmap(dist_mat, ax=ax3, cmap="Blues",
                    xticklabels=[GESTURES[i][1] for i in range(N_CLASSES)],
                    yticklabels=[GESTURES[i][1] for i in range(N_CLASSES)],
                    annot=True, fmt=".1f",
                    annot_kws={"size": 7, "color": "white"},
                    cbar_kws={"shrink": .7})
        ax3.tick_params(colors=TEXT_COL, labelsize=9)
        ax3.set_title("Centroid Distance Matrix",
                      color="#e2e8f0", fontsize=9, pad=6)
        plt.tight_layout(); st.pyplot(fig3); plt.close()

    # Feature importance (if Random Forest)
    if hasattr(clf, "feature_importances_"):
        st.markdown('<div class="section-hdr">🔍 Top 20 Important Features (Pixel Positions)</div>',
                    unsafe_allow_html=True)
        fi      = clf.feature_importances_
        top_idx = np.argsort(fi)[::-1][:20]
        fig_fi, ax_fi = styled_fig(11, 3.5)
        clr_fi  = [GESTURES[i % N_CLASSES][2] for i in range(20)]
        ax_fi.bar(range(20), fi[top_idx], color=clr_fi, edgecolor="none", width=0.7)
        ax_fi.set_xticks(range(20))
        ax_fi.set_xticklabels(
            [f"px{i}" for i in top_idx],
            color=TEXT_COL, fontsize=7, rotation=45
        )
        ax_fi.set_ylabel("Importance", color=TEXT_COL, fontsize=9)
        ax_fi.set_title("Random Forest Feature Importances",
                         color="#e2e8f0", fontsize=11, pad=8)
        plt.tight_layout(); st.pyplot(fig_fi); plt.close()


# ══ TAB 4 — LIVE PREDICT ═════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-hdr">🔍 Predict Gesture from Image</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    📁 Upload a <b>hand gesture image</b> (JPG/PNG) — ideally similar to
    LeapGestRecog IR images. The model will extract features and classify the gesture.
    </div>""", unsafe_allow_html=True)

    col_in, col_out = st.columns([1, 1])

    with col_in:
        uploaded = st.file_uploader(
            "Upload hand gesture image",
            type=["jpg", "jpeg", "png"]
        )

        # OR pick from dataset
        st.markdown("**Or pick a real image from the dataset:**")
        pick_cls = st.selectbox(
            "Gesture class",
            options=list(range(N_CLASSES)),
            format_func=lambda x: f"{GESTURES[x][1]} {GESTURES[x][0]}"
        )
        pick_btn = st.button("🎲 Pick Random Sample from Dataset")

    pred_gi, pred_proba, disp_img = None, None, None

    if uploaded is not None:
        pil     = Image.open(uploaded).convert("L")  # grayscale
        disp_img= np.array(pil)
        pred_gi, pred_proba = predict_image(disp_img, result)

    elif pick_btn:
        # Find images for chosen class in dataset
        found = []
        for subject in sorted(os.listdir(dataset_path)):
            folder = os.path.join(dataset_path, subject, GESTURE_FOLDERS[pick_cls])
            if not os.path.isdir(folder): continue
            for fname in os.listdir(folder):
                if fname.lower().endswith((".png",".jpg",".jpeg")):
                    found.append(os.path.join(folder, fname))
        if found:
            rng      = np.random.RandomState(int(time.time()) % 9999)
            fpath    = rng.choice(found)
            disp_img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            pred_gi, pred_proba = predict_image(disp_img, result)
        else:
            st.warning("No images found for this class in the dataset path.")

    with col_out:
        if disp_img is not None and pred_gi is not None:
            # Show image
            fig_img, ax_img = plt.subplots(1, 1, figsize=(4, 4), facecolor=DARK_BG)
            ax_img.imshow(disp_img, cmap="gray")
            ax_img.axis("off")
            ax_img.set_title("Input Image", color="#e2e8f0", fontsize=10)
            plt.tight_layout(); st.pyplot(fig_img); plt.close()

            name, emoji, clr, desc = GESTURES[pred_gi]
            conf = pred_proba[pred_gi] * 100
            rgb  = tuple(int(clr.lstrip("#")[i:i+2], 16) for i in (0,2,4))
            bg   = f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.08)"

            st.markdown(f"""
            <div class="pred-box" style="background:{bg};border:2px solid {clr}">
                <div style="font-size:3rem">{emoji}</div>
                <div class="pred-name" style="color:{clr}">{name.upper()}</div>
                <div class="pred-conf">
                    Confidence: <b>{conf:.1f}%</b><br>
                    <span style="font-size:.8rem">{desc}</span>
                </div>
            </div>""", unsafe_allow_html=True)

            # Top-5 bar
            st.markdown("<br>", unsafe_allow_html=True)
            top5     = np.argsort(pred_proba)[::-1][:5]
            top5_clr = [GESTURES[i][2] for i in top5]
            top5_lbl = [f"{GESTURES[i][1]} {GESTURES[i][0]}" for i in top5]
            fig_p, ax_p = styled_fig(5.5, 3.2)
            ax_p.barh(top5_lbl[::-1], pred_proba[top5[::-1]] * 100,
                      color=top5_clr[::-1], edgecolor="none", height=0.55)
            ax_p.set_xlim(0, 115)
            ax_p.set_xlabel("Probability (%)", color=TEXT_COL, fontsize=9)
            ax_p.set_title("Top-5 Predictions", color="#e2e8f0", fontsize=10, pad=6)
            ax_p.tick_params(colors=TEXT_COL, labelsize=9)
            for i, v in enumerate(pred_proba[top5[::-1]] * 100):
                ax_p.text(v + 1, i, f"{v:.1f}%", va="center",
                          color=TEXT_COL, fontsize=8)
            plt.tight_layout(); st.pyplot(fig_p); plt.close()
        else:
            st.markdown("""
            <div style="text-align:center;color:#374151;padding:4rem 0;font-size:1rem">
                📁 Upload an image or click "Pick Random Sample"
            </div>""", unsafe_allow_html=True)


# ══ TAB 5 — DATA EXPLORER ════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-hdr">📋 Dataset Statistics</div>',
                unsafe_allow_html=True)

    col_d1, col_d2 = st.columns([2, 1])
    with col_d1:
        summary = []
        for gi in range(N_CLASSES):
            name, emoji, clr, desc = GESTURES[gi]
            mask = y == gi
            if mask.sum() == 0:
                continue
            summary.append({
                "Gesture":         f"{emoji} {name}",
                "Class ID":        gi,
                "Images Loaded":   int(mask.sum()),
                "Mean Intensity":  round(float(X[mask, :IMG_SIZE*IMG_SIZE].mean()), 4),
                "Std Intensity":   round(float(X[mask, :IMG_SIZE*IMG_SIZE].std()), 4),
                "Description":     desc,
            })
        df_sum = pd.DataFrame(summary)
        st.dataframe(
            df_sum.style.background_gradient(
                subset=["Images Loaded", "Mean Intensity"],
                cmap="YlOrBr"
            ),
            use_container_width=True, height=390
        )

    with col_d2:
        st.markdown("**📊 Images per Class**")
        fig_d, ax_d = styled_fig(4, 5)
        counts = [(y == gi).sum() for gi in range(N_CLASSES)]
        labels = [f"{GESTURES[i][1]} {GESTURES[i][0]}" for i in range(N_CLASSES)]
        bar_c  = [GESTURES[i][2] for i in range(N_CLASSES)]
        ax_d.barh(labels[::-1], counts[::-1], color=bar_c[::-1],
                  edgecolor="none", height=0.65)
        for i, v in enumerate(counts[::-1]):
            ax_d.text(v + 1, i, str(v), va="center",
                      color=TEXT_COL, fontsize=8)
        ax_d.set_xlabel("# Images", color=TEXT_COL, fontsize=9)
        ax_d.set_title("Images Loaded per Class",
                        color="#e2e8f0", fontsize=10, pad=6)
        ax_d.tick_params(colors=TEXT_COL, labelsize=8)
        plt.tight_layout(); st.pyplot(fig_d); plt.close()

    # Feature correlation heatmap
    st.markdown('<div class="section-hdr">🔗 Feature Correlation (first 30 features)</div>',
                unsafe_allow_html=True)
    corr = pd.DataFrame(X[:, :30]).corr()
    fig_c, ax_c = styled_fig(11, 5)
    sns.heatmap(corr, ax=ax_c, cmap="coolwarm", center=0,
                linewidths=.2, linecolor=DARK_BG,
                xticklabels=False, yticklabels=False,
                cbar_kws={"shrink": .7})
    ax_c.set_title("Feature Correlation Matrix (first 30 pixel features)",
                   color="#e2e8f0", fontsize=11, pad=8)
    plt.tight_layout(); st.pyplot(fig_c); plt.close()


# ══ TAB 6 — PIPELINE ═════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-hdr">ℹ️ Full Processing Pipeline</div>',
                unsafe_allow_html=True)
    col_p1, col_p2 = st.columns(2)

    with col_p1:
        st.markdown(f"""
**🔄 Pipeline Steps**
```
leapGestRecog/
  00/ → 01_palm/*.png  ─┐
  00/ → 02_l/*.png     ─┤  Load grayscale
  ...                   │  images per class
  09/ → 10_down/*.png  ─┘
         │
         ▼
  extract_features(img)
  ├── Resize → {IMG_SIZE}×{IMG_SIZE}
  ├── Pixel flatten   (4096 feats)
  ├── Gradient hist   (18 feats)
  ├── 4×4 block stats (32 feats)
  └── Global stats    (6 feats)
         │
         ▼
  StandardScaler
         │
         ▼
  PCA (100 components)
         │
         ▼
  Train/Test Split (80/20)
         │
         ▼
  {clf_name}
         │
         ▼
  Test Accuracy: {acc*100:.1f}%
  CV Mean:       {cv_sc.mean()*100:.1f}%
```

**📂 Dataset Folder Structure**
```
leapGestRecog/
  ├── 00/           ← Subject 0
  │   ├── 01_palm/
  │   ├── 02_l/
  │   ├── 03_fist/
  │   ├── 04_fist_moved/
  │   ├── 05_thumb/
  │   ├── 06_index/
  │   ├── 07_ok/
  │   ├── 08_palm_moved/
  │   ├── 09_c/
  │   └── 10_down/
  ├── 01/           ← Subject 1
  └── ...           ← Up to 09
```
        """)

    with col_p2:
        st.markdown("""
**🧠 Classifiers**

| Model | Notes |
|-------|-------|
| Random Forest | Best speed, interpretable |
| MLP Neural Net | Best accuracy on large data |
| SVM (RBF) | Solid for medium-sized data |
| Gradient Boosting | High accuracy, slower |

**🖼️ Feature Extraction Detail**

| Feature Group | Size | Description |
|--------------|------|-------------|
| Pixel intensities | 4096 | Resized 64×64 grayscale |
| Gradient histogram | 18 | HOG-style orientation bins |
| Block statistics | 32 | Mean+Std in 4×4 grid |
| Global statistics | 6 | Mean, Std, Max, Min, Q1, Q3 |
| **Total** | **4152** | Per image |

**📥 Download Dataset**
```bash
# Kaggle CLI
pip install kaggle
kaggle datasets download \\
    -d gti-upm/leapgestrecog
unzip leapgestrecog.zip \\
    -d leapGestRecog/
```

**📊 LeapGestRecog Info**

| Property | Value |
|----------|-------|
| Sensor | Leap Motion Controller |
| Image type | Infrared (grayscale) |
| Resolution | 640 × 240 px |
| Subjects | 10 people |
| Classes | 10 gestures |
| Total images | ~20,000 |
        """)

    st.markdown(f"""
    <div class="info-box">
    ✅ <b>Current run:</b>
    Classifier = <b>{clf_name}</b> |
    Dataset = <code>{dataset_path}</code> |
    Images loaded = <b>{result['n_samples']:,}</b> |
    Test Accuracy = <b>{acc*100:.1f}%</b> |
    CV = <b>{cv_sc.mean()*100:.1f}% ± {cv_sc.std()*100:.2f}%</b>
    </div>""", unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border-color:rgba(34,211,238,.1);margin-top:3rem">
<p style="text-align:center;color:#1f2937;font-size:.8rem">
🤚 Hand Gesture Recognition · ProDigy Infotech Task-04 ·
LeapGestRecog Dataset · Streamlit + scikit-learn + OpenCV
</p>""", unsafe_allow_html=True)