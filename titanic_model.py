import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix,
classification_report, roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# ■■ 1. LOAD DATA ■■
df = pd.read_csv('train.csv')
print('Dataset shape:', df.shape)
print(df.head())

# ■■ 2. EXPLORATORY DATA ANALYSIS ■■
print('\nMissing values:')
print(df.isnull().sum())
print('\nSurvival rate:', df['Survived'].mean().round(3))

# ■■ 3. FEATURE ENGINEERING ■■
df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace(['Lady','Countess','Capt','Col','Don',
'Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare')
df['Title'] = df['Title'].replace('Mlle','Miss')
df['Title'] = df['Title'].replace('Ms','Miss')
df['Title'] = df['Title'].replace('Mme','Mrs')
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# ■■ 4. HANDLE MISSING VALUES ■■
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)

# ■■ 5. ENCODE CATEGORICAL VARIABLES ■■
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
title_map = {'Mr':0,'Miss':1,'Mrs':2,'Master':3,'Rare':4}
df['Title'] = df['Title'].map(title_map).fillna(0)

# ■■ 6. SELECT FEATURES ■■
features = ['Pclass','Sex','Age','Fare','Embarked',
'FamilySize','IsAlone','Title']
X = df[features]
y = df['Survived']

# ■■ 7. TRAIN-TEST SPLIT ■■
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42, stratify=y)

# ■■ 8. SCALE FEATURES ■■
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# ■■ 9. TRAIN MODEL ■■
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_sc, y_train)

# ■■ 10. EVALUATE ■■
y_pred = model.predict(X_test_sc)
y_prob = model.predict_proba(X_test_sc)[:,1]
print('\n=== MODEL RESULTS ===')
print(f'Accuracy : {accuracy_score(y_test, y_pred):.4f}')
print(f'ROC-AUC : {roc_auc_score(y_test, y_prob):.4f}')
print('\nClassification Report:')
print(classification_report(y_test, y_pred))

# ■■ 11. PLOT CONFUSION MATRIX ■■
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
ax.imshow(cm, cmap='Blues')
ax.set_title('Confusion Matrix', fontweight='bold')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i,j]), ha='center', va='center',
        color='white' if cm[i,j]>cm.max()/2 else 'black', fontsize=14)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print('Chart saved: confusion_matrix.png')

# ■■ 12. PLOT ROC CURVE ■■
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, linewidth=2, label=f'AUC = {roc_auc:.2f}')
ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
ax.set_title('ROC Curve', fontweight='bold')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print('Chart saved: roc_curve.png')

# ■■ 13. FEATURE DISTRIBUTION CHART ■■
fig, ax = plt.subplots(figsize=(7, 5))
sns.histplot(data=df, x='Age', hue='Survived', bins=30,
             kde=True, alpha=0.5, element='step', ax=ax)
ax.set_title('Age Distribution by Survival', fontweight='bold')
ax.set_xlabel('Age')
ax.set_ylabel('Count')
plt.tight_layout()
plt.savefig('feature_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print('Chart saved: feature_distribution.png')

# Feature importance (coefficients) - retained
fig, ax = plt.subplots(figsize=(6, 5))
coef_df = pd.DataFrame({'Feature':features, 'Coefficient':model.coef_[0]})
coef_df = coef_df.reindex(coef_df['Coefficient'].abs().sort_values(ascending=True).index)
ax.barh(coef_df['Feature'], coef_df['Coefficient'],
color=['red' if c<0 else 'green' for c in coef_df['Coefficient']])
ax.set_title('Feature Coefficients', fontweight='bold')
ax.axvline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig('titanic_results.png', dpi=150, bbox_inches='tight')
plt.close()
print('Chart saved: titanic_results.png')
