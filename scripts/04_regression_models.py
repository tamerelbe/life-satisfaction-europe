"""Step 4: OLS and ordinal logistic regression on the raw 0-10 outcome."""
import _setup  # noqa: F401
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from src import config as C
from src.data import load_clean

df = load_clean()
X, y = df[C.FEATURES], df[C.OUTCOME]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=C.TEST_SIZE, random_state=C.SEED)

# OLS with country-clustered standard errors on the full sample (for inference)
ols_full = sm.OLS(y, sm.add_constant(X)).fit(cov_type="cluster", cov_kwds={"groups": df[C.COUNTRY]})
coefs = pd.DataFrame({"coef": ols_full.params, "se": ols_full.bse, "p": ols_full.pvalues}).round(4)
std_beta = ols_full.params[C.FEATURES] * X.std() / y.std()
coefs.loc[C.FEATURES, "std_beta"] = std_beta.round(3)
coefs.to_csv(C.OUTPUT_DIR / "ols_coefficients.csv")
print(ols_full.summary().tables[1])

# out-of-sample performance (same split as the ML models)
ols = sm.OLS(y_tr, sm.add_constant(X_tr)).fit()
pred_ols = ols.predict(sm.add_constant(X_te))

ordinal = OrderedModel(y_tr.astype(int), X_tr.drop(columns=[]), distr="logit").fit(method="bfgs", disp=False, maxiter=500)
probs = ordinal.predict(X_te)
pred_ord = (np.asarray(probs) * np.sort(y_tr.unique())).sum(axis=1)  # expected score

rows = []
for name, pred in [("OLS", pred_ols), ("Ordinal logistic", pred_ord)]:
    rows.append({"model": name, "outcome": "raw 0-10",
                 "rmse": mean_squared_error(y_te, pred) ** 0.5,
                 "mae": mean_absolute_error(y_te, pred), "r2": r2_score(y_te, pred)})
perf = pd.DataFrame(rows).round(3)
perf.to_csv(C.OUTPUT_DIR / "regression_performance.csv", index=False)
print("\n", perf.to_string(index=False))
print("\nTop standardised effects:\n", std_beta.abs().sort_values(ascending=False).head(6).round(3).to_string())
