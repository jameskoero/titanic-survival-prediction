import pandas as pd


def _preprocess_like_training_script(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    data['Title'] = data['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    data['Title'] = data['Title'].replace([
        'Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'
    ], 'Rare')
    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')

    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['IsAlone'] = (data['FamilySize'] == 1).astype(int)

    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())

    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    title_map = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4}
    data['Title'] = data['Title'].map(title_map).fillna(0)

    return data


def test_data_loading_from_csv(tmp_path):
    csv_path = tmp_path / 'train.csv'
    csv_path.write_text(
        'Survived,Pclass,Name,Sex,Age,SibSp,Parch,Fare,Embarked\n'
        '0,3,"Allen, Mr. William",male,35,0,0,8.05,S\n'
        '1,1,"Cumings, Mrs. John",female,38,1,0,71.2833,C\n',
        encoding='utf-8',
    )

    loaded = pd.read_csv(csv_path)
    assert loaded.shape == (2, 9)
    assert {'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'}.issubset(loaded.columns)


def test_preprocessing_creates_expected_features_without_missing_values():
    raw = pd.DataFrame(
        {
            'Survived': [0, 1, 1],
            'Pclass': [3, 1, 2],
            'Name': ['Allen, Mr. William', 'Cumings, Mrs. John', 'Moran, Miss. Ava'],
            'Sex': ['male', 'female', 'female'],
            'Age': [35.0, None, 19.0],
            'SibSp': [0, 1, 0],
            'Parch': [0, 0, 0],
            'Fare': [8.05, 71.2833, None],
            'Embarked': ['S', 'C', None],
        }
    )

    processed = _preprocess_like_training_script(raw)

    for column in ['Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'IsAlone', 'Title']:
        assert processed[column].isna().sum() == 0

    assert processed.loc[0, 'FamilySize'] == 1
    assert processed.loc[0, 'IsAlone'] == 1
