"""Loading and cleaning ESS Round 10."""
import numpy as np
import pandas as pd
from scipy.stats import norm

from . import config as C


def load_raw(path=C.DATA_PATH) -> pd.DataFrame:
    cols = [C.OUTCOME, C.WEIGHT, C.COUNTRY] + C.RAW_VARS
    df = pd.read_csv(path, usecols=lambda c: c in cols, low_memory=False)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"ESS file is missing columns: {missing}")
    return df


def recode_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in [C.OUTCOME] + C.RAW_VARS:
        codes = C.MISSING_CODES.get(col, C.MISSING_CODES["default"])
        df.loc[df[col].isin(codes), col] = np.nan
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Recode ESS missing codes, drop rows without the outcome, impute predictors.

    Household income has ~22% missing, so it is median-imputed and an
    `income_missing` flag is kept; the other predictors have <3% missing and
    are median-imputed. Robustness script 07 re-runs the models on complete cases.
    """
    df = recode_missing(df)
    df = df.dropna(subset=[C.OUTCOME]).reset_index(drop=True)
    df["income_missing"] = df["hinctnta"].isna().astype(int)
    df["employment_missing"] = df["emplrel"].isna().astype(int)
    for col in C.RAW_VARS:
        if col in ("gndr", "emplrel"):
            continue
        df[col] = df[col].fillna(df[col].median())
    df["female"] = (df["gndr"] == 2).astype(int)
    df["self_employed"] = (df["emplrel"] == 2).astype(int)
    df["not_employee"] = (~df["emplrel"].isin([1, 2])).astype(int)
    df["age_sq"] = (df["agea"] - df["agea"].mean()) ** 2 / 100
    df[C.WEIGHT] = df[C.WEIGHT].fillna(1.0)
    return df


def rank_normal(y: pd.Series) -> pd.Series:
    """Rank-based inverse normal transformation (Blom, 1958).

    Ties share their average rank, so every respondent with the same raw score
    gets the same latent score; the order of respondents is preserved while the
    unequal spacing of the 0-10 categories is corrected.
    """
    r = y.rank(method="average")
    n = y.notna().sum()
    return pd.Series(norm.ppf((r - 0.375) / (n + 0.25)), index=y.index, name="latent")


def load_clean() -> pd.DataFrame:
    path = C.OUTPUT_DIR / "ess10_clean.pkl"
    if path.exists():
        return pd.read_pickle(path)
    raise FileNotFoundError("Run scripts/01_data_cleaning.py first.")
