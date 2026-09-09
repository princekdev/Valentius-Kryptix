"""
data_preprocessing.py
----------------------
Loads the Adult Census Income dataset, performs the train/test split, and
builds a leakage-safe preprocessing pipeline (ColumnTransformer) that is
fit ONLY on the training data.

Correct flow enforced here:
    Raw data -> train/test split -> fit preprocessor on TRAIN -> transform
    TRAIN and TEST with the already-fitted preprocessor.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.2

# The full Adult dataset has ~48,842 rows. SVM (RBF kernel) scales poorly
# with sample size (O(n^2)-O(n^3)), so a stratified subsample is used to
# keep training time reasonable while remaining well above a "trivial"
# dataset size (this is documented in the README).
SAMPLE_SIZE = 8000

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "adult.csv"

COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income",
]

TARGET_COLUMN = "income"

NUMERICAL_FEATURES = [
    "age", "fnlwgt", "education-num", "capital-gain", "capital-loss",
    "hours-per-week",
]

CATEGORICAL_FEATURES = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country",
]


def load_raw_data() -> pd.DataFrame:
    """Load the Adult dataset and standardize missing-value markers."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Make sure data/adult.csv exists."
        )
    df = pd.read_csv(DATA_PATH, header=None, names=COLUMN_NAMES,
                      skipinitialspace=True)
    # The raw file uses "?" to mark missing categorical values.
    df = df.replace("?", pd.NA)
    return df


def stratified_subsample(df: pd.DataFrame, n: int = SAMPLE_SIZE,
                          random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Take a class-stratified random subsample of the full dataset (see
    SAMPLE_SIZE comment above), preserving the proportions of the two
    income classes via sklearn's stratified train_test_split.
    """
    sampled, _ = train_test_split(
        df, train_size=n, random_state=random_state, stratify=df[TARGET_COLUMN]
    )
    return sampled.reset_index(drop=True)


def inspect_data(df: pd.DataFrame) -> None:
    """Print a quick data-quality summary (used by main.py)."""
    print(f"Dataset shape: {df.shape}")
    print(f"\nTarget distribution:\n{df[TARGET_COLUMN].value_counts()}")
    print(f"\nMissing values per column:\n{df.isnull().sum()[df.isnull().sum() > 0]}")


def split_data(df: pd.DataFrame):
    """
    Split raw features/target into train/test sets BEFORE any preprocessing
    is fit, to prevent data leakage. Stratified on the target to preserve
    class balance in both splits.
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].apply(lambda v: 1 if v.strip() == ">50K" else 0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test


def build_preprocessor() -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
      - Imputes + scales numerical features (needed by LogReg, KNN, SVM;
        harmless for Decision Tree / Naive Bayes).
      - Imputes + one-hot encodes categorical features.

    This single fitted object is reused, unchanged, for every one of the
    5 models so the comparison is fair. It is fit ONLY on X_train.
    """
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # sparse_output=False so that Gaussian Naive Bayes (which requires
        # dense arrays) can share the exact same preprocessing pipeline as
        # every other model.
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_pipeline, NUMERICAL_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])
    return preprocessor
