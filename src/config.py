"""Global settings for the life-satisfaction analysis (ESS Round 10)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "ESS10.csv"
FIGURES_DIR = ROOT / "figures"
OUTPUT_DIR = ROOT / "output"

SEED = 42
TEST_SIZE = 0.20

# ESS non-substantive answer codes (refusal, don't know, no answer)
MISSING_CODES = {
    "default": [77, 88, 99],
    "agea": [999],
    "eduyrs": [77, 88, 99],
    "hinctnta": [77, 88, 99],
    "health": [7, 8, 9],
    "gndr": [9],
    "emplrel": [6, 7, 8, 9, 66],
}

OUTCOME = "stflife"
WEIGHT = "pspwght"
COUNTRY = "cntry"

DEMOGRAPHIC = ["agea", "gndr", "eduyrs"]
SOCIOECONOMIC = ["hinctnta", "emplrel", "health"]
ATTITUDINAL = ["ppltrst", "trstprl", "trstprt", "pplfair", "stfdem"]
RAW_VARS = DEMOGRAPHIC + SOCIOECONOMIC + ATTITUDINAL

# model features after cleaning (age squared captures the U-shape in age)
FEATURES = [
    "agea", "age_sq", "female", "eduyrs", "hinctnta", "income_missing",
    "self_employed", "not_employee", "health",
    "ppltrst", "trstprl", "trstprt", "pplfair", "stfdem",
]

LABELS = {
    "agea": "Age", "age_sq": "Age squared", "female": "Female", "eduyrs": "Years of education",
    "hinctnta": "Income decile", "income_missing": "Income not reported",
    "self_employed": "Self-employed", "not_employee": "Not employed / other",
    "health": "Self-rated health (1 = very good)", "ppltrst": "Interpersonal trust",
    "trstprl": "Trust in parliament", "trstprt": "Trust in parties",
    "pplfair": "Fairness of others", "stfdem": "Satisfaction with democracy",
}

# clustering
CLUSTER_VARS = ["agea", "eduyrs", "hinctnta", "health", "ppltrst", "trstprl", "pplfair", "stfdem"]
K_RANGE = range(2, 9)
K_FINAL = 4
N_INIT = 25
