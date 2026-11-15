"""Step 6: K-Means well-being profiles (elbow + silhouette to choose K)."""
import _setup  # noqa: F401
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from src import config as C
from src.data import load_clean

df = load_clean()
Z = StandardScaler().fit_transform(df[C.CLUSTER_VARS])
sample = pd.DataFrame(Z).sample(min(10_000, len(df)), random_state=C.SEED).index  # silhouette is O(n^2)

scores = []
for k in C.K_RANGE:
    km = KMeans(n_clusters=k, n_init=C.N_INIT, random_state=C.SEED).fit(Z)
    scores.append({"k": k, "inertia": km.inertia_,
                   "silhouette": silhouette_score(Z[sample], km.labels_[sample])})
scores = pd.DataFrame(scores).round(3)
scores.to_csv(C.OUTPUT_DIR / "k_selection.csv", index=False)
print(scores.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(scores.k, scores.inertia, marker="o", color="#2f5d8a"); axes[0].set(title="Elbow", xlabel="K", ylabel="Inertia")
axes[1].plot(scores.k, scores.silhouette, marker="o", color="#2f5d8a"); axes[1].set(title="Silhouette", xlabel="K")
fig.tight_layout(); fig.savefig(C.FIGURES_DIR / "07_k_selection.png", dpi=150)

km = KMeans(n_clusters=C.K_FINAL, n_init=C.N_INIT, random_state=C.SEED).fit(Z)
df["cluster"] = km.labels_
profile = df.groupby("cluster")[C.CLUSTER_VARS + [C.OUTCOME]].mean().round(2)
profile.insert(0, "share_pct", (df["cluster"].value_counts(normalize=True).sort_index() * 100).round(1))
profile = profile.sort_values(C.OUTCOME, ascending=False)
profile.to_csv(C.OUTPUT_DIR / "cluster_profiles.csv")
df[["cluster"]].to_csv(C.OUTPUT_DIR / "cluster_assignments.csv", index=False)
print("\nCluster profiles (sorted by life satisfaction):\n", profile.to_string())
