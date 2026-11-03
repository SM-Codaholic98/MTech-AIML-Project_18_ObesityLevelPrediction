"""Data loading, duplicate removal, splitting, and encoding utilities."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

TARGET_COLUMN = "NObeyesdad"
ORDINAL_COLUMNS = [
    "Gender",
    "family_history_with_overweight",
    "FAVC",
    "SMOKE",
    "SCC",
]
ONE_HOT_COLUMNS = ["CAEC", "CALC", "MTRANS"]


@dataclass
class DataSplits:
    """Encoded train, validation, and test partitions."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    ordinal_encoder: OrdinalEncoder
    onehot_encoder: OneHotEncoder


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the CSV and remove exact duplicate rows, as in the notebook."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Place the original CSV in data/ or pass --data."
        )
    df = pd.read_csv(path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' in the dataset.")
    return df.drop_duplicates().reset_index(drop=True)


def _make_onehot_encoder() -> OneHotEncoder:
    """Support both newer and older scikit-learn OneHotEncoder APIs."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def prepare_data(
    df: pd.DataFrame,
    random_state: int = 42,
) -> DataSplits:
    """Create stratified 70/15/15 splits and fit encoders on training data only."""
    if df.isnull().any().any():
        missing = df.columns[df.isnull().any()].tolist()
        raise ValueError(
            "Missing values were found in columns: " + ", ".join(missing)
            + ". The supplied notebook reports no missing values; define an imputation "
            "strategy before continuing if your dataset differs."
        )

    missing_columns = [
        column for column in [TARGET_COLUMN, *ORDINAL_COLUMNS, *ONE_HOT_COLUMNS]
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Required dataset columns are missing: {missing_columns}")

    X = df.drop(columns=TARGET_COLUMN).copy()
    y = df[TARGET_COLUMN].copy()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=random_state, stratify=y
    )
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp
    )

    ordinal_encoder = OrdinalEncoder()
    onehot_encoder = _make_onehot_encoder()

    X_train.loc[:, ORDINAL_COLUMNS] = ordinal_encoder.fit_transform(
        X_train[ORDINAL_COLUMNS]
    )
    X_val.loc[:, ORDINAL_COLUMNS] = ordinal_encoder.transform(X_val[ORDINAL_COLUMNS])
    X_test.loc[:, ORDINAL_COLUMNS] = ordinal_encoder.transform(X_test[ORDINAL_COLUMNS])

    train_ohe = pd.DataFrame(
        onehot_encoder.fit_transform(X_train[ONE_HOT_COLUMNS]),
        columns=onehot_encoder.get_feature_names_out(ONE_HOT_COLUMNS),
        index=X_train.index,
    )
    val_ohe = pd.DataFrame(
        onehot_encoder.transform(X_val[ONE_HOT_COLUMNS]),
        columns=onehot_encoder.get_feature_names_out(ONE_HOT_COLUMNS),
        index=X_val.index,
    )
    test_ohe = pd.DataFrame(
        onehot_encoder.transform(X_test[ONE_HOT_COLUMNS]),
        columns=onehot_encoder.get_feature_names_out(ONE_HOT_COLUMNS),
        index=X_test.index,
    )

    X_train = pd.concat([X_train.drop(columns=ONE_HOT_COLUMNS), train_ohe], axis=1)
    X_val = pd.concat([X_val.drop(columns=ONE_HOT_COLUMNS), val_ohe], axis=1)
    X_test = pd.concat([X_test.drop(columns=ONE_HOT_COLUMNS), test_ohe], axis=1)

    # Keep exactly the same feature order across all partitions.
    X_val = X_val.reindex(columns=X_train.columns)
    X_test = X_test.reindex(columns=X_train.columns)

    return DataSplits(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        ordinal_encoder=ordinal_encoder,
        onehot_encoder=onehot_encoder,
    )
