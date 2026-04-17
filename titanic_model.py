"""Train and evaluate a Titanic survival prediction model with visualizations."""
# pylint: disable=invalid-name

import warnings

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_auc_score, roc_curve, auc)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
df['Title'] = df['Title'].replace(
    ['Lady', 'Countess', 'Capt', 'Col', 'Don',
     'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],
    'Rare'
)
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# ■■ 4. HANDLE MISSING VALUES ■■
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)

# ■■ 5. ENCODE CATEGORICAL VARIABLES ■■
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
title_map = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4}
df['Title'] = df['Title'].map(title_map).fillna(0)

# ■■ 6. SELECT FEATURES ■■
features = [
    'Pclass', 'Sex', 'Age', 'Fare', 'Embarked',
    'FamilySize', 'IsAlone', 'Title'
]
X = df[features]
y = df['Survived']

# ■■ 7. TRAIN-TEST SPLIT ■■
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ■■ 8. SCALE FEATURES ■■
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# ■■ 9. TRAIN MODEL ■■
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_sc, y_train)

# ■■ 10. EVALUATE ■■
y_pred = model.predict(X_test_sc)
y_prob = model.predict_proba(X_test_sc)[:, 1]
print('\n=== MODEL RESULTS ===')
print(f'Accuracy : {accuracy_score(y_test, y_pred):.4f}')
print(f'ROC-AUC : {roc_auc_score(y_test, y_prob):.4f}')
print('\nClassification Report:')
print(classification_report(y_test, y_pred))

# ■■ 11. PLOT CONFUSION MATRIX ■■
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, cmap='Blues', aspect='auto')
ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Did Not Survive', 'Survived'])
ax.set_yticklabels(['Did Not Survive', 'Survived'])

# Add text annotations
for i in range(2):
    for j in range(2):
        ax.text(
            j,
            i,
            f'{cm[i, j]}\n({100 * cm[i, j] / cm.sum():.1f}%)',
            ha='center',
            va='center',
            color='white' if cm[i, j] > cm.max() / 2 else 'black',
            fontsize=14,
            fontweight='bold'
        )

plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig('outputs/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print('✓ Saved: confusion_matrix.png')

# ■■ 12. PLOT ROC CURVE ■■
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(9, 8))
ax.plot(
    fpr, tpr, color='#1f77b4', lw=3,
    label=f'ROC Curve (AUC = {roc_auc:.3f})'
)
ax.plot(
    [0, 1], [0, 1], color='red', lw=2,
    linestyle='--', label='Random Classifier'
)
ax.fill_between(fpr, tpr, alpha=0.2, color='#1f77b4')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title(
    'ROC Curve - Model Performance',
    fontsize=16,
    fontweight='bold',
    pad=20
)
ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('outputs/roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print('✓ Saved: roc_curve.png')

# ■■ 13. PLOT FEATURE DISTRIBUTION ■■
X_test_orig = pd.DataFrame(X_test, columns=features)
X_test_orig['Survived'] = y_test.values

plot_features = [
    ('Sex', 'Passenger Gender (0=Male, 1=Female)'),
    ('Age', 'Passenger Age'),
    ('Pclass', 'Passenger Class (1, 2, or 3)'),
    ('Fare', 'Ticket Fare')
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    'Feature Distribution by Survival Outcome',
    fontsize=16,
    fontweight='bold',
    y=1.00
)

for idx, (feature, label) in enumerate(plot_features):
    ax = axes[idx // 2, idx % 2]

    survived = X_test_orig[X_test_orig['Survived'] == 1][feature]
    not_survived = X_test_orig[X_test_orig['Survived'] == 0][feature]

    ax.hist(
        not_survived,
        bins=20,
        alpha=0.6,
        label='Did Not Survive',
        color='#d62728',
        edgecolor='black'
    )
    ax.hist(
        survived,
        bins=20,
        alpha=0.6,
        label='Survived',
        color='#2ca02c',
        edgecolor='black'
    )
    ax.set_xlabel(label, fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title(f'{label} Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

plt.tight_layout()
plt.savefig('outputs/feature_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print('✓ Saved: feature_distribution.png')

# ■■ 14. PLOT FEATURE IMPORTANCE ■■
fig, ax = plt.subplots(figsize=(10, 6))
coef_df = pd.DataFrame({'Feature': features, 'Coefficient': model.coef_[0]})
coef_df = coef_df.reindex(
    coef_df['Coefficient'].abs().sort_values(ascending=True).index
)
colors = ['#d62728' if c < 0 else '#2ca02c' for c in coef_df['Coefficient']]
ax.barh(
    coef_df['Feature'],
    coef_df['Coefficient'],
    color=colors,
    edgecolor='black',
    linewidth=1.2
)
ax.set_xlabel('Coefficient Value', fontsize=12, fontweight='bold')
ax.set_title(
    'Feature Importance (Model Coefficients)',
    fontsize=14,
    fontweight='bold',
    pad=15
)
ax.axvline(0, color='black', linewidth=1)
ax.grid(True, alpha=0.3, linestyle='--', axis='x')
plt.tight_layout()
plt.savefig('outputs/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print('\n✓ All visualizations generated successfully!')
