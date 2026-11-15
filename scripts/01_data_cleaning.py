"""Step 1: load ESS Round 10, recode missing values, impute, add the latent outcome."""
import _setup  # noqa: F401
from src import config as C
from src.data import load_raw, clean, rank_normal

C.OUTPUT_DIR.mkdir(exist_ok=True)
raw = load_raw()
df = clean(raw)
df["latent"] = rank_normal(df[C.OUTCOME])

missing = (raw[C.RAW_VARS].apply(lambda s: s.isin(
    C.MISSING_CODES.get(s.name, C.MISSING_CODES["default"])) | s.isna()).mean() * 100).round(2)
missing.rename("pct_missing").to_csv(C.OUTPUT_DIR / "missingness.csv")

df.to_pickle(C.OUTPUT_DIR / "ess10_clean.pkl")
print(f"Raw rows: {len(raw):,}  |  analytic sample: {len(df):,}")
print("Missing % by variable:\n", missing.sort_values(ascending=False).to_string())
