# Run once on any machine with pandas+sklearn to generate 05_calibration.png
import urllib.request, os
if not os.path.exists('train.csv'):
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv',
        'train.csv')
exec(open('titanic_model.py').read().replace("if __name__", "if False"))
from titanic_model import *
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
