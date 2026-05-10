"""
Titanic Survival Prediction — Streamlit Demo
============================================
Deployed at: https://titanic-survival-jameskoero.streamlit.app

Trains the model on first run (auto-downloads data).
No model.joblib needed — self-contained for Streamlit Cloud.

Author: James Koero · jmskoero@gmail.com · github.com/jameskoero
"""
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from urllib.request import urlretrieve

# ── sklearn imports ────────────────────────────────────────────────
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered",
)

# ── Constants ──────────────────────────────────────────────────────
SEED = 42
NUM = ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone", "HasCabin"]
CAT = ["Sex", "Title", "Embarked"]
MODEL_PATH = Path("outputs/model.joblib")
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"


# ── Feature engineering (matches titanic_model.py exactly) ────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False).fillna("Rare")
    rare = ["Lady","Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"]
    df["Title"] = df["Title"].replace(rare, "Rare")
    df["Title"] = df["Title"].replace({"Mlle":"Miss","Ms":"Miss","Mme":"Mrs"})
    df["Title"] = df["Title"].where(df["Title"].isin(["Mr","Mrs","Miss","Master","Rare"]),"Rare")
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"]    = (df["FamilySize"] == 1).astype(int)
    df["HasCabin"]   = df["Cabin"].notna().astype(int)
    age_med = df.groupby(["Pclass","Sex"])["Age"].transform("median")
    df["Age"] = df["Age"].fillna(age_med).fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    return df


def build_model():
    """Build the sklearn Pipeline."""
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
    ])
    return Pipeline([("prep", prep), ("clf", LogisticRegression(
        max_iter=4000, solver="liblinear", random_state=SEED
    ))])


# ── Load or train model ────────────────────────────────────────────
@st.cache_resource(show_spinner="Training model on Titanic data…")
def get_model():
    """Load saved model OR train from scratch (Streamlit Cloud compatible)."""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    # Auto-download data
    data_path = Path("train.csv")
    if not data_path.exists():
        try:
            urlretrieve(DATA_URL, data_path)
        except Exception:
            st.error("Could not download training data. Check internet connection.")
            return None

    df = pd.read_csv(data_path)
    df_eng = engineer_features(df)
    X, y = df_eng[NUM + CAT], df_eng["Survived"]
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    pipe = build_model()
    cv   = StratifiedKFold(5, shuffle=True, random_state=SEED)
    grid = GridSearchCV(pipe,
                        {"clf__C": [0.01, 0.1, 0.5, 1], "clf__penalty": ["l1", "l2"]},
                        scoring="f1", cv=cv, n_jobs=-1, refit=True)
    grid.fit(X_tr, y_tr)
    return grid.best_estimator_


# ── Exact SHAP for linear model (no shap package) ─────────────────
def linear_shap(model, X_row: pd.DataFrame):
    prep = model.named_steps["prep"]
    clf  = model.named_steps["clf"]
    z    = prep.transform(X_row)[0]
    ohe  = prep.named_transformers_["cat"]
    cat_names = [f"{c}={v}" for c, cats in zip(CAT, ohe.categories_) for v in cats]
    names = NUM + cat_names
    phi   = clf.coef_[0] * z
    df    = pd.DataFrame({"feature": names, "phi": phi})
    df["abs"] = df["phi"].abs()
    return df.sort_values("abs", ascending=False).head(6).reset_index(drop=True)


# ── Build single-passenger DataFrame ──────────────────────────────
def make_passenger(pclass, sex, age, sibsp, parch, fare, embarked, has_cabin):
    if sex == "female":
        title = "Miss" if age < 18 else "Mrs"
    else:
        title = "Master" if age < 15 else "Mr"
    family = sibsp + parch + 1
    return pd.DataFrame([{
        "Pclass": pclass, "Age": age, "SibSp": sibsp, "Parch": parch,
        "Fare": fare, "FamilySize": family, "IsAlone": int(family == 1),
        "HasCabin": int(has_cabin), "Sex": sex, "Title": title,
        "Embarked": embarked,
        # dummy cols engineer_features needs
        "Name": f"Test, {title}. Passenger", "Cabin": "C1" if has_cabin else np.nan,
    }])


# ════════════════════════════════════════════════════════════════════
# UI
# ════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='background:#0A1628;padding:24px 20px;border-radius:12px;margin-bottom:20px;text-align:center;border:1px solid #C9A84C44'>
<h1 style='color:#C9A84C;margin:0;font-size:26px'>🚢 TITANIC SURVIVAL PREDICTOR</h1>
<p style='color:#94a3b8;margin:8px 0 0;font-size:13px'>
Senior-Grade Logistic Regression · SHAP Explainability · Built in Kisumu, Kenya
</p>
</div>
""", unsafe_allow_html=True)

# Sidebar — story context
with st.sidebar:
    st.markdown("## 📖 The Story")
    st.info("**Sex=female coefficient: +2.61**\n\n"
            "e^2.61 = **13.5×** survival odds.\n\n"
            "That is a historical document, not a model output.")
    st.warning("**Thomas Andrews**\n\n"
               "Ship's designer · Age 39 · 1st class · Had a cabin\n\n"
               "Model predicted: **91% survival**\n\nHe died — gave his lifeboat seat away.")
    st.markdown("---")
    st.markdown("**Built by James Koero**\nKisumu, Kenya\n\n"
                "[GitHub](https://github.com/jameskoero) · "
                "[LinkedIn](https://linkedin.com/in/jameskoero)")

# Load model
model = get_model()
if model is None:
    st.stop()

# ── Input ──────────────────────────────────────────────────────────
st.markdown("### 🧑 Passenger Profile")
col1, col2 = st.columns(2)

with col1:
    pclass   = st.selectbox("Ticket Class", [1, 2, 3],
                             format_func=lambda x: f"{x}{'st' if x==1 else 'nd' if x==2 else 'rd'} Class")
    sex      = st.radio("Sex", ["male", "female"])
    age      = st.slider("Age", 0.17, 80.0, 30.0, step=1.0)
    embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"],
                             format_func=lambda x: {"S":"Southampton","C":"Cherbourg","Q":"Queenstown"}[x])

with col2:
    sibsp     = st.number_input("Siblings / Spouses aboard", 0, 8, 0)
    parch     = st.number_input("Parents / Children aboard", 0, 6, 0)
    fare      = st.slider("Fare ($)", 0.0, 300.0, 32.0, step=1.0)
    has_cabin = st.checkbox("Had a cabin record?", value=False,
                             help="Cabin record presence is a proxy for wealth")

# ── Predict ────────────────────────────────────────────────────────
if st.button("🔮 Predict Survival", use_container_width=True, type="primary"):
    passenger = make_passenger(pclass, sex, age, sibsp, parch, fare, embarked, has_cabin)
    passenger_eng = engineer_features(passenger)
    X_pred = passenger_eng[NUM + CAT]

    proba = model.predict_proba(X_pred)[0][1]
    survived = proba >= 0.5
    color    = "#22c55e" if survived else "#ef4444"
    label    = "SURVIVED" if survived else "DID NOT SURVIVE"
    emoji    = "✅" if survived else "❌"

    # Result card
    st.markdown(f"""
    <div style='background:{color}18;border:2px solid {color};border-radius:12px;
                padding:24px;text-align:center;margin:16px 0'>
        <div style='font-size:48px'>{emoji}</div>
        <div style='font-size:28px;font-weight:bold;color:{color}'>{label}</div>
        <div style='font-size:20px;color:#94a3b8;margin-top:6px'>
            Survival probability: <strong style='color:{color}'>{proba:.1%}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # SHAP breakdown
    st.markdown("### 🔍 Why — SHAP Feature Contributions")
    st.caption("Exact SHAP values computed analytically from LR coefficients (φᵢ = βᵢ × zᵢ)")

    shap_df = linear_shap(model, X_pred)
    for _, row in shap_df.iterrows():
        direction = "↑ Survival" if row["phi"] > 0 else "↓ Survival"
        c = "#22c55e" if row["phi"] > 0 else "#ef4444"
        pct = min(100, int(row["abs"] * 60))
        st.markdown(f"""
        <div style='display:flex;align-items:center;gap:10px;margin-bottom:6px;
                    padding:8px 12px;background:#0d1f35;border-radius:6px'>
            <span style='font-family:monospace;font-size:11px;color:#C9A84C;
                         min-width:140px;flex-shrink:0'>{row['feature']}</span>
            <div style='flex:1;background:#1a3050;border-radius:3px;height:8px'>
                <div style='width:{pct}%;background:{c};height:100%;border-radius:3px'></div>
            </div>
            <span style='font-size:11px;color:{c};min-width:100px;text-align:right'>
                {row['phi']:+.3f} {direction}
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption(
        "The model learned the average. The errors are the exceptions. "
        "And in the exceptions, you find the actual human story."
    )

st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#475569;font-size:12px">'
    '"What is living in YOUR model\'s 18% error rate?"'
    '</div>',
    unsafe_allow_html=True
  )

