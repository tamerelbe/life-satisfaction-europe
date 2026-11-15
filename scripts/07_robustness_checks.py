"""Step 7: robustness checks.

1. Composite institutional trust instead of separate parliament/party trust
2. Raw 0-10 outcome instead of the latent score
3. Separate models for men and women
4. Complete cases only (no income imputation)
5. Survey-weighted OLS
6. Variance inflation factors
"""
import _setup  # noqa: F401
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from src import config as C
from src.data import load_clean

df = load_clean()


def gb_r2(data, features, target="latent"):
    X_tr, X_te, y_tr, y_te = train_test_split(data[features], data[target],
                                              test_size=C.TEST_SIZE, random_state=C.SEED)
    m = GradientBoostingRegressor(n_estimators=400, learning_rate=0.05, max_depth=3,
                                  subsample=0.8, random_state=C.SEED).fit(X_tr, y_tr)
    return r2_score(y_te, m.predict(X_te))


results = {"Baseline (Gradient Boosting, latent)": gb_r2(df, C.FEATURES)}

df["inst_trust"] = df[["trstprl", "trstprt"]].mean(axis=1)
comp = [f for f in C.FEATURES if f not in ("trstprl", "trstprt")] + ["inst_trust"]
results["Composite institutional trust"] = gb_r2(df, comp)
results["Raw 0-10 outcome"] = gb_r2(df, C.FEATURES, target=C.OUTCOME)

no_gender = [f for f in C.FEATURES if f != "female"]
results["Men only"] = gb_r2(df[df.female == 0], no_gender)
results["Women only"] = gb_r2(df[df.female == 1], no_gender)
results["Complete cases (income reported)"] = gb_r2(df[df.income_missing == 0],
                                                     [f for f in C.FEATURES if f != "income_missing"])
rob = pd.Series(results, name="test_r2").round(3)
rob.to_csv(C.OUTPUT_DIR / "robustness.csv")
print(rob.to_string())

X = sm.add_constant(df[C.FEATURES])
unweighted = sm.OLS(df[C.OUTCOME], X).fit().params
weighted = sm.WLS(df[C.OUTCOME], X, weights=df[C.WEIGHT]).fit().params
w = pd.DataFrame({"unweighted": unweighted, "weighted": weighted}).round(3)
w.to_csv(C.OUTPUT_DIR / "weighted_vs_unweighted.csv")
print("\nWeighted vs unweighted OLS:\n", w.to_string())

Xv = df[C.FEATURES].assign(const=1.0)
vif = pd.Series([variance_inflation_factor(Xv.values, i) for i in range(len(C.FEATURES))],
                index=C.FEATURES, name="vif").round(2)
vif.to_csv(C.OUTPUT_DIR / "vif.csv")
print("\nVIF:\n", vif.to_string())
