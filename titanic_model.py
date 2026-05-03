# ================================================================
# titanic_model.py
# Project : Titanic Survival Prediction
# Author  : James Koero (jameskoero)
# IDE     : PyramIDE (Android)
# Model   : Logistic Regression
# Note    : NO seaborn — pure matplotlib only
# ================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')          # REQUIRED for PyramIDE / Android
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

import warnings
warnings.filterwarnings('ignore')

# ================================================================
# CONFIGURATION — change paths here if needed
# ================================================================
CSV_PATH    = 'titanic.csv'        # place titanic.csv in same folder
OUTPUT_PNG  = 'titanic_results.png'

# ================================================================
# STEP 1 — LOAD DATA
# Download titanic.csv from:
# https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
# Save it in the same folder as this script on your phone.
# ================================================================

def load_data(path):
    try:
        df = pd.read_csv(path)
        print("[OK] titanic.csv loaded successfully.")
        print(f"     Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
        return df
    except FileNotFoundError:
        print("[ERROR] titanic.csv not found!")
        print("  Download it from:")
        print("  https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
        print("  Save it in the SAME folder as this script.")
        raise


# ================================================================
# STEP 2 — EXPLORATORY DATA ANALYSIS
# ================================================================

def run_eda(df):
    print("=" * 50)
    print("  EXPLORATORY DATA ANALYSIS")
    print("=" * 50)

    print(f"Columns : {list(df.columns)}\n")

    print("Missing values:")
    missing = df.isnull().sum()
    print(missing[missing > 0], "\n")

    survived     = df['Survived'].sum()
    not_survived = len(df) - survived
    rate         = (survived / len(df)) * 100

    print(f"Survivors      : {survived}")
    print(f"Did not survive: {not_survived}")
    print(f"Survival rate  : {rate:.1f}%\n")


# ================================================================
# STEP 3 — PREPROCESSING
# ================================================================

def preprocess(df):

    data = df.copy()

    # Keep only useful columns
    cols = ['Survived', 'Pclass', 'Sex', 'Age',
            'SibSp', 'Parch', 'Fare', 'Embarked']
    data = data[cols].copy()

    # Fill missing values
    data['Age']      = data['Age'].fillna(data['Age'].median())
    data['Fare']     = data['Fare'].fillna(data['Fare'].median())
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

    # Encode Sex: male=0, female=1
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

    # Encode Embarked: S=0, C=1, Q=2
    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    # Feature engineering
    data['FamilySize'] = data['SibSp'] + data['Parch']
    data['IsAlone']    = (data['FamilySize'] == 0).astype(int)

    # Age group buckets
    data['AgeGroup'] = pd.cut(
        data['Age'],
        bins=[0, 12, 18, 35, 60, 100],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)

    print("[OK] Preprocessing complete.")

    X = data.drop('Survived', axis=1)
    y = data['Survived']

    print(f"     Features used: {list(X.columns)}\n")
    return X, y


# ================================================================
# STEP 4 — SPLIT AND SCALE
# ================================================================

def split_and_scale(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"[OK] Train samples : {X_train.shape[0]}")
    print(f"     Test  samples : {X_test.shape[0]}\n")

    return X_train, X_test, y_train, y_test


# ================================================================
# STEP 5 — TRAIN MODEL
# ================================================================

def train_model(X_train, y_train):
    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42,
        solver='lbfgs'
    )
    model.fit(X_train, y_train)
    print("[OK] Model trained successfully.\n")
    return model


# ================================================================
# STEP 6 — EVALUATE MODEL
# ================================================================

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc     = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm      = confusion_matrix(y_test, y_pred)
    report  = classification_report(
        y_test, y_pred,
        target_names=['Did Not Survive', 'Survived']
    )

    print("=" * 50)
    print("  EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Accuracy  : {acc * 100:.2f}%")
    print(f"  ROC-AUC   : {roc_auc:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\nClassification Report:\n{report}")

    return y_pred, y_prob, acc, roc_auc, cm


# ================================================================
# STEP 7 — SAVE RESULTS CHART (pure matplotlib, NO seaborn)
# ================================================================

def save_results_chart(model, X_test, y_test, y_prob, cm, feature_names):

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.patch.set_facecolor('#1a1a2e')

    title_txt = (
        'Titanic Survival Prediction — Logistic Regression\n'
        'Author: James Koero  |  IDE: PyramIDE'
    )
    fig.suptitle(title_txt, fontsize=13, fontweight='bold',
                 color='white', y=0.98)

    # ---- 1. Confusion Matrix ----
    ax = axes[0, 0]
    ax.set_facecolor('#16213e')
    ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title('Confusion Matrix', fontweight='bold', color='white')
    classes    = ['Did Not\nSurvive', 'Survived']
    tick_marks = np.arange(2)
    ax.set_xticks(tick_marks); ax.set_xticklabels(classes, color='white')
    ax.set_yticks(tick_marks); ax.set_yticklabels(classes, color='white')
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center', fontsize=16, fontweight='bold',
                    color='white' if cm[i, j] > thresh else 'black')
    ax.set_xlabel('Predicted', color='white')
    ax.set_ylabel('Actual', color='white')
    ax.tick_params(colors='white')

    # ---- 2. ROC Curve ----
    ax = axes[0, 1]
    ax.set_facecolor('#16213e')
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc     = roc_auc_score(y_test, y_prob)
    ax.plot(fpr, tpr, color='#00d4ff', lw=2.5,
            label=f'ROC AUC = {roc_auc:.3f}')
    ax.plot([0, 1], [0, 1], color='gray', lw=1.2,
            linestyle='--', label='Random Classifier')
    ax.fill_between(fpr, tpr, alpha=0.15, color='#00d4ff')
    ax.set_title('ROC Curve', fontweight='bold', color='white')
    ax.set_xlabel('False Positive Rate', color='white')
    ax.set_ylabel('True Positive Rate', color='white')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#16213e', labelcolor='white', fontsize=9)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    # ---- 3. Feature Coefficients ----
    ax = axes[1, 0]
    ax.set_facecolor('#16213e')
    coefs   = model.coef_[0]
    indices = np.argsort(np.abs(coefs))[::-1]
    colors  = ['#2ecc71' if c > 0 else '#e74c3c' for c in coefs[indices]]
    y_pos   = np.arange(len(coefs))
    ax.barh(y_pos, coefs[indices], color=colors, edgecolor='#1a1a2e', height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in indices],
                       fontsize=9, color='white')
    ax.axvline(0, color='white', linewidth=0.8, linestyle='--')
    ax.set_title('Feature Coefficients', fontweight='bold', color='white')
    ax.set_xlabel('Coefficient Value', color='white')
    ax.tick_params(colors='white')
    pos_p = mpatches.Patch(color='#2ecc71', label='Increases survival')
    neg_p = mpatches.Patch(color='#e74c3c', label='Decreases survival')
    ax.legend(handles=[pos_p, neg_p], facecolor='#16213e',
              labelcolor='white', fontsize=8)

    # ---- 4. Prediction Probability Distribution ----
    ax = axes[1, 1]
    ax.set_facecolor('#16213e')
    bins         = np.linspace(0, 1, 25)
    y_test_arr   = np.array(y_test)
    survived_p   = y_prob[y_test_arr == 1]
    not_surv_p   = y_prob[y_test_arr == 0]
    ax.hist(not_surv_p, bins=bins, alpha=0.7, color='#e74c3c',
            label='Did Not Survive', edgecolor='#1a1a2e')
    ax.hist(survived_p, bins=bins, alpha=0.7, color='#2ecc71',
            label='Survived', edgecolor='#1a1a2e')
    ax.axvline(0.5, color='white', linestyle='--',
               linewidth=1.5, label='Threshold = 0.5')
    ax.set_title('Predicted Probability Distribution',
                 fontweight='bold', color='white')
    ax.set_xlabel('Probability of Survival', color='white')
    ax.set_ylabel('Count', color='white')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#16213e', labelcolor='white', fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[OK] Chart saved → {OUTPUT_PNG}\n")


# ================================================================
# MAIN PIPELINE
# ================================================================

def main():
    print("\n" + "=" * 50)
    print("  TITANIC SURVIVAL PREDICTION")
    print("  James Koero | PyramIDE | Logistic Regression")
    print("=" * 50 + "\n")

    df                               = load_data(CSV_PATH)
    run_eda(df)
    X, y                             = preprocess(df)
    feature_names                    = list(X.columns)
    X_train, X_test, y_train, y_test = split_and_scale(X, y)
    model                            = train_model(X_train, y_train)
    y_pred, y_prob, acc, roc_auc, cm = evaluate_model(model, X_test, y_test)
    save_results_chart(model, X_test, y_test, y_prob, cm, feature_names)

    print("=" * 50)
    print(f"  FINAL ACCURACY  : {acc * 100:.2f}%")
    print(f"  ROC-AUC SCORE   : {roc_auc:.4f}")
    print("  STATUS          : COMPLETE ✓")
    print("=" * 50 + "\n")
    print("Files saved in this folder:")
    print(f"  → {OUTPUT_PNG}")


if __name__ == '__main__':
    main()
