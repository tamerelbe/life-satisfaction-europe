"""Step 2: descriptive statistics and the main exploratory figures."""
import _setup  # noqa: F401
import matplotlib.pyplot as plt
import seaborn as sns
from src import config as C
from src.data import load_clean

C.FIGURES_DIR.mkdir(exist_ok=True)
df = load_clean()
y = df[C.OUTCOME]
print(f"N = {len(df):,} | mean = {y.mean():.2f} | SD = {y.std():.2f} | "
      f"median = {y.median():.1f} | skew = {y.skew():.2f}")

desc = df[[C.OUTCOME] + C.RAW_VARS].describe().T.round(2)
desc.to_csv(C.OUTPUT_DIR / "descriptives.csv")

fig, ax = plt.subplots(figsize=(7, 4))
y.value_counts(normalize=True).sort_index().mul(100).plot.bar(ax=ax, color="#2f5d8a")
ax.set(xlabel="Life satisfaction (0-10)", ylabel="% of respondents", title="Distribution of life satisfaction")
fig.tight_layout(); fig.savefig(C.FIGURES_DIR / "01_outcome_distribution.png", dpi=150)

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, var in zip(axes, ["hinctnta", "health", "stfdem"]):
    df.groupby(var)[C.OUTCOME].mean().plot(ax=ax, marker="o", color="#2f5d8a")
    ax.set(xlabel=C.LABELS[var], ylabel="Mean life satisfaction")
fig.suptitle("Life satisfaction by income, health and satisfaction with democracy")
fig.tight_layout(); fig.savefig(C.FIGURES_DIR / "02_bivariate.png", dpi=150)

age = df.assign(age_band=(df["agea"] // 5) * 5).groupby("age_band")[C.OUTCOME].mean()
fig, ax = plt.subplots(figsize=(7, 4))
age.plot(ax=ax, marker="o", color="#2f5d8a")
ax.set(xlabel="Age (5-year bands)", ylabel="Mean life satisfaction", title="The U-shape in age")
fig.tight_layout(); fig.savefig(C.FIGURES_DIR / "03_age_u_shape.png", dpi=150)

corr = df[[C.OUTCOME] + C.ATTITUDINAL + ["hinctnta", "health"]].corr()
fig, ax = plt.subplots(figsize=(8, 6.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
ax.set_title("Correlation matrix")
fig.tight_layout(); fig.savefig(C.FIGURES_DIR / "04_correlations.png", dpi=150)
print("Figures saved to figures/")
