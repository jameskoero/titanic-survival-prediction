# Data

This project expects Titanic competition data from Kaggle.

- Download `train.csv` from: https://www.kaggle.com/c/titanic/data
- Place it at: `data/raw/train.csv`

Then run:

```bash
python src/preprocess.py --input data/raw/train.csv --output data/processed/train_processed.csv
```
