"""
Titanic Survival Prediction — FastAPI
POST /predict → probability + analytical SHAP
Deploys to Render free tier.
"""
from __future__ import annotations
import os, json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="Senior-grade LR pipeline · analytical SHAP · built in Kisumu, Kenya",
    version="2.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global model state ────────────────────────────────────────────────
_model   = None
_NUMERIC = ["Pclass","Age","SibSp","Parch","Fare","FamilySize","IsAlone","HasCabin"]
_CATEG   = ["Sex","Title","Embarked"]

def _train_and_cache():
    """Train on startup using 3-tier data fallback."""
    global _model
    import joblib, warnings
    warnings.filterwarnings("ignore")
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from titanic_model import load_data, engineer_features

    df  = load_data()
    df  = engineer_features(df)
    X   = df[_NUMERIC + _CATEG]
    y   = df["Survived"]

    prep = ColumnTransformer([
        ("num", StandardScaler(), _NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), _CATEG),
    ])
    pipe = Pipeline([("prep", prep),
                     ("clf",  LogisticRegression(C=0.5, max_iter=1000, random_state=42))])

    cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(pipe, {"clf__C":[0.1,0.5,1.0]}, cv=cv,
                        scoring="roc_auc", n_jobs=-1)
    grid.fit(X, y)
    _model = grid.best_estimator_

    os.makedirs("outputs", exist_ok=True)
    joblib.dump(_model, "outputs/model.joblib")
    print(f"Model trained. Best C={grid.best_params_['clf__C']}, "
          f"CV AUC={grid.best_score_:.4f}")

@app.on_event("startup")
def startup():
    import joblib
    global _model
    path = "outputs/model.joblib"
    if os.path.exists(path):
        _model = joblib.load(path)
        print("Model loaded from disk.")
    else:
        print("No saved model — training now...")
        _train_and_cache()

# ── Schemas ───────────────────────────────────────────────────────────
class PassengerIn(BaseModel):
    Pclass:   int   = Field(..., ge=1, le=3, example=1)
    Sex:      str   = Field(...,              example="male")
    Age:      float = Field(..., ge=0, le=100,example=39.0)
    SibSp:    int   = Field(0,  ge=0,         example=0)
    Parch:    int   = Field(0,  ge=0,         example=0)
    Fare:     float = Field(..., ge=0,         example=42.0)
    Embarked: str   = Field("S",              example="S")
    Cabin:    str | None = Field(None,        example="A36")

class PredictionOut(BaseModel):
    survival_probability: float
    prediction:           str
    confidence:           str
    shap_top5:            dict
    top_factor:           str

# ── Endpoints ─────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "Titanic Survival Prediction API",
        "version": "2.0.0",
        "docs":    "/docs",
        "health":  "/health",
        "predict": "POST /predict",
        "author":  "James Koero · Kisumu, Kenya",
        "github":  "https://github.com/jameskoero/titanic-survival-prediction",
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}

@app.post("/predict", response_model=PredictionOut)
def predict(p: PassengerIn):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    from titanic_model import engineer_features

    # Build minimal raw row
    name_map = {
        ("male",   True):  "Mr",
        ("female", True):  "Mrs",
        ("male",   False): "Master",
        ("female", False): "Miss",
    }
    is_adult = p.Age >= 18
    fake_title = name_map.get((p.Sex, is_adult), "Mr")

    row = pd.DataFrame([{
        "PassengerId": 0, "Survived": 0,
        "Pclass": p.Pclass,
        "Name":   f"Test, {fake_title}. Passenger",
        "Sex":    p.Sex,
        "Age":    p.Age,
        "SibSp":  p.SibSp,
        "Parch":  p.Parch,
        "Ticket": "000000",
        "Fare":   p.Fare,
        "Cabin":  p.Cabin,
        "Embarked": p.Embarked,
    }])

    df_eng = engineer_features(row)
    X      = df_eng[_NUMERIC + _CATEG]

    prob       = float(_model.predict_proba(X)[0][1])
    prediction = "SURVIVED" if prob >= 0.5 else "DID NOT SURVIVE"
    gap        = abs(prob - 0.5)
    confidence = "High" if gap > 0.3 else "Medium" if gap > 0.15 else "Low"

    # Analytical SHAP: φ_i = β_i × z_i
    prep      = _model.named_steps["prep"]
    clf       = _model.named_steps["clf"]
    X_tr      = prep.transform(X)
    cat_names = [f"{c}={v}"
                 for c, cats in zip(_CATEG, prep.named_transformers_["cat"].categories_)
                 for v in cats]
    all_names  = _NUMERIC + cat_names
    shap_all   = {n: round(float(w * v), 4)
                  for n, w, v in zip(all_names, clf.coef_[0], X_tr[0])}

    # Top 5 by absolute value
    top5 = dict(sorted(shap_all.items(),
                        key=lambda x: abs(x[1]),
                        reverse=True)[:5])
    top_factor = max(shap_all, key=lambda k: abs(shap_all[k]))

    return PredictionOut(
        survival_probability=round(prob, 4),
        prediction=prediction,
        confidence=confidence,
        shap_top5=top5,
        top_factor=top_factor,
    )
