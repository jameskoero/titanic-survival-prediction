import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix,
classification_report, roc_auc_score)
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
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].imshow(cm, cmap='Blues')
axes[0].set_title('Confusion Matrix', fontweight='bold')
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, str(cm[i,j]), ha='center', va='center',
        color='white' if cm[i,j]>cm.max()/2 else 'black', fontsize=14)

# Feature importance (coefficients)
coef_df = pd.DataFrame({'Feature':features, 'Coefficient':model.coef_[0]})
coef_df = coef_df.reindex(coef_df['Coefficient'].abs().sort_values(ascending=True).index)
axes[1].barh(coef_df['Feature'], coef_df['Coefficient'],
color=['red' if c<0 else 'green' for c in coef_df['Coefficient']])
axes[1].set_title('Feature Coefficients', fontweight='bold')
axes[1].axvline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig('titanic_results.png', dpi=150, bbox_inches='tight')
plt.show()
print('Chart saved: titanic_results.png')
