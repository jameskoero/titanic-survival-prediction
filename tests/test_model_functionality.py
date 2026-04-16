import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def test_logistic_regression_can_fit_and_predict_binary_outcomes():
    x = np.array(
        [
            [3, 0, 22, 7.25, 0, 2, 0, 0],
            [1, 1, 38, 71.2833, 1, 2, 0, 2],
            [3, 1, 26, 7.925, 0, 1, 1, 1],
            [1, 1, 35, 53.1, 0, 2, 0, 2],
            [3, 0, 35, 8.05, 0, 1, 1, 0],
            [2, 0, 28, 13.0, 0, 1, 1, 0],
        ],
        dtype=float,
    )
    y = np.array([0, 1, 1, 1, 0, 0])

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(x_scaled, y)

    preds = model.predict(x_scaled)
    probs = model.predict_proba(x_scaled)

    assert preds.shape == (6,)
    assert set(np.unique(preds)).issubset({0, 1})
    assert probs.shape == (6, 2)
