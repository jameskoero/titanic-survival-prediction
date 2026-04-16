# Raw Data

Place the original Kaggle Titanic training dataset file here as:

- `data/raw/train.csv`

Source: https://www.kaggle.com/c/titanic/data

## Optional Kaggle CLI download

1. Install Kaggle CLI: `pip install kaggle`
2. Configure `~/.kaggle/kaggle.json`
3. Run:
   ```bash
   kaggle competitions download -c titanic -p data/raw
   unzip data/raw/titanic.zip -d data/raw
   ```
