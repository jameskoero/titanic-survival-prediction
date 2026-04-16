import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def test_required_ml_dependencies_are_importable_and_usable():
    assert pd.DataFrame({'x': [1]}).shape == (1, 1)
    assert np.array([1, 2, 3]).shape == (3,)
    assert isinstance(LogisticRegression(max_iter=1000), LogisticRegression)

    figure = plt.figure()
    try:
        assert figure is not None
    finally:
        plt.close(figure)
