"""Step 3: inspect the rank-based inverse normal (Blom) transformation of the outcome.

The latent score is created in step 1; this script shows that it is roughly
N(0, 1) and that the implied category thresholds are unevenly spaced.
"""
import _setup  # noqa: F401
import matplotlib.pyplot as plt
from src import config as C
from src.data import load_clean

df = load_clean()
thresholds = df.groupby(C.OUTCOME)["latent"].first().sort_index()
gaps = thresholds.diff().dropna().round(3)
print("Latent score for each raw category:\n", thresholds.round(3).to_string())
print("\nGap between consecutive categories:\n", gaps.to_string())
thresholds.rename("latent_score").to_csv(C.OUTPUT_DIR / "latent_thresholds.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df["latent"].plot.hist(bins=40, ax=axes[0], color="#2f5d8a")
axes[0].set(title="A. Latent life-satisfaction scores", xlabel="Latent score")
axes[1].plot(thresholds.index, thresholds.values, marker="o", color="#2f5d8a")
axes[1].set(title="B. Raw category vs latent score", xlabel="Raw score (0-10)", ylabel="Latent score")
fig.tight_layout(); fig.savefig(C.FIGURES_DIR / "05_latent_transformation.png", dpi=150)
