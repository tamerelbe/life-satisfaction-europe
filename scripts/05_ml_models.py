"""Step 5: Random Forest and Gradient Boosting on the latent outcome, with permutation importance."""
import _setup  # noqa: F401
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from src import config as C
from src.data import load_clean

df = load_clean()
X, y = df[C.FEATURES], df["latent"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=C.TEST_SIZE, random_state=C.SEED)

models = {
    "Random Forest": RandomForestRegressor(n_estimators=500, min_samples_leaf=20, max_features=0.5,
                                           n_jobs=-1, random_state=C.SEED),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=400, learning_rate=0.05, max_depth=3,
                                                   subsample=0.8, random_state=C.SEED),
}
rows, importances = [], {}
for name, model in models.items():
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    rows.append({"model": name, "outcome": "latent", "rmse": mean_squared_error(y_te, pred) ** 0.5,
                 "mae": mean_absolute_error(y_te, pred), "r2": r2_score(y_te, pred)})
    pi = permutation_importance(model, X_te, y_te, n_repeats=10, random_state=C.SEED, n_jobs=-1)
    importances[name] = pd.Series(pi.importances_mean, index=C.FEATURES)

perf = pd.DataFrame(rows).round(3)
perf.to_csv(C.OUTPUT_DIR / "ml_performance.csv", index=False)
imp = pd.DataFrame(importances).sort_values("Gradient Boosting", ascending=False).round(4)
imp.to_csv(C.OUTPUT_DIR / "permutation_importance.csv")
print(perf.to_string(index=False), "\n\nPermutation importance:\n", imp.to_string())

top = imp.head(8).iloc[::-1].rename(index=C.LABELS)
fig, ax = plt.subplots(figsize=(8, 5))
top.plot.barh(ax=ax, color=["#9fb7cf", "#2f5d8a"])
ax.set(xlabel="Drop in R² when the variable is shuffled", title="What predicts life satisfaction? (test set)")
fig.tight_layout(); fig.savefig(C.FIGURES_DIR / "06_feature_importance.png", dpi=150)
