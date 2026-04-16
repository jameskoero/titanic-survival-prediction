import warnings

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

RARE_TITLES = [
    "Lady",
    "Countess",
    "Capt",
    "Col",
    "Don",
    "Dr",
    "Major",
    "Rev",
    "Sir",
    "Jonkheer",
    "Dona",
]
TITLE_MAP = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4}
FEATURES = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "IsAlone", "Title"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create title, family size and isolation features."""
    result = df.copy()
    result["Title"] = result["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    result["Title"] = result["Title"].replace(RARE_TITLES, "Rare")
    result["Title"] = result["Title"].replace("Mlle", "Miss")
    result["Title"] = result["Title"].replace("Ms", "Miss")
    result["Title"] = result["Title"].replace("Mme", "Mrs")
    result["FamilySize"] = result["SibSp"] + result["Parch"] + 1
    result["IsAlone"] = (result["FamilySize"] == 1).astype(int)
    return result


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill key missing values with standard statistics."""
    result = df.copy()
    result["Age"] = result["Age"].fillna(result["Age"].median())
    result["Embarked"] = result["Embarked"].fillna(result["Embarked"].mode()[0])
    result["Fare"] = result["Fare"].fillna(result["Fare"].median())
    return result


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Encode model categorical columns as numeric values."""
    result = df.copy()
    result["Sex"] = result["Sex"].map({"male": 0, "female": 1})
    result["Embarked"] = result["Embarked"].map({"S": 0, "C": 1, "Q": 2})
    result["Title"] = result["Title"].map(TITLE_MAP).fillna(0)
    return result


def prepare_features_target(df: pd.DataFrame):
    """Select model features and target."""
    x_values = df[FEATURES]
    y_values = df["Survived"]
    return x_values, y_values


def split_and_scale(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """Create train/test split and scale the feature matrices."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    return X_train_sc, X_test_sc, y_train, y_test, scaler


def train_model(X_train_sc, y_train) -> LogisticRegression:
    """Train logistic regression model."""
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_sc, y_train)
    return model


def evaluate_model(model: LogisticRegression, X_test_sc, y_test):
    """Calculate common binary classification metrics."""
    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:, 1]
    cm = confusion_matrix(y_test, y_pred)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "classification_report": classification_report(y_test, y_pred),
        "confusion_matrix": cm,
        "y_pred": y_pred,
    }


def create_results_plot(confusion, features, coefficients, output_path: str = "titanic_results.png", show: bool = True):
    """Create and save confusion matrix and coefficient visualizations."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(confusion, cmap="Blues")
    axes[0].set_title("Confusion Matrix", fontweight="bold")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            axes[0].text(
                j,
                i,
                str(confusion[i, j]),
                ha="center",
                va="center",
                color="white" if confusion[i, j] > confusion.max() / 2 else "black",
                fontsize=14,
            )

    coef_df = pd.DataFrame({"Feature": features, "Coefficient": coefficients})
    coef_df = coef_df.reindex(coef_df["Coefficient"].abs().sort_values(ascending=True).index)
    axes[1].barh(
        coef_df["Feature"],
        coef_df["Coefficient"],
        color=["red" if val < 0 else "green" for val in coef_df["Coefficient"]],
    )
    axes[1].set_title("Feature Coefficients", fontweight="bold")
    axes[1].axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def run_pipeline(data_path: str = "train.csv", output_plot: str = "titanic_results.png", show_plot: bool = True):
    """Run the full Titanic model training pipeline."""
    df = pd.read_csv(data_path)
    print("Dataset shape:", df.shape)
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nSurvival rate:", df["Survived"].mean().round(3))

    df = engineer_features(df)
    df = fill_missing_values(df)
    df = encode_categorical(df)

    X, y = prepare_features_target(df)
    X_train_sc, X_test_sc, y_train, y_test, _ = split_and_scale(X, y)

    model = train_model(X_train_sc, y_train)
    metrics = evaluate_model(model, X_test_sc, y_test)

    print("\n=== MODEL RESULTS ===")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"ROC-AUC : {metrics['roc_auc']:.4f}")
    print("\nClassification Report:")
    print(metrics["classification_report"])

    create_results_plot(
        metrics["confusion_matrix"],
        FEATURES,
        model.coef_[0],
        output_path=output_plot,
        show=show_plot,
    )
    print(f"Chart saved: {output_plot}")
    return metrics


if __name__ == "__main__":
    run_pipeline()
