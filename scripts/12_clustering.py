"""
PHASE 7 - SCRIPT 12
Purpose : Cluster languages by phoneme inventory profile
          - Build language x phoneme binary presence matrix
          - K-Means clustering with silhouette score to find optimal K
          - Hierarchical clustering with dendrogram
          - Map clusters back to regions and families
          - Evaluate cluster purity against known macroareas
Outputs :
    data/processed/language_clusters.csv
    outputs/tables/cluster_region_crosstab.csv
    outputs/tables/cluster_family_crosstab.csv
    outputs/tables/cluster_phoneme_profile.csv
    outputs/tables/silhouette_scores.csv
    outputs/tables/phase7_report.txt
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
import os
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.join(os.path.dirname(__file__), "..")
DEDUP_PATH = os.path.join(BASE_DIR, "data", "processed", "phoible_dedup.csv")
FREQ_PATH  = os.path.join(BASE_DIR, "outputs", "tables", "phoneme_frequency.csv")
LANG_PATH  = os.path.join(BASE_DIR, "data", "processed", "language_inventory.csv")
TABLES_DIR = os.path.join(BASE_DIR, "outputs", "tables")
DATA_DIR   = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(TABLES_DIR, exist_ok=True)

lines = []
def log(text=""):
    print(text)
    lines.append(str(text))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load data
# ─────────────────────────────────────────────────────────────────────────────
log("=" * 60)
log("STEP 1 — Loading data")
log("=" * 60)

df       = pd.read_csv(DEDUP_PATH, low_memory=False)
freq     = pd.read_csv(FREQ_PATH)
lang_inv = pd.read_csv(LANG_PATH)

for col in ["has_click", "has_tone_segment"]:
    lang_inv[col] = (
        lang_inv[col].astype(str).str.strip().str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
    )

TOTAL_LANGUAGES = df["Glottocode"].nunique()
log(f"  Languages : {TOTAL_LANGUAGES:,}")
log(f"  Phonemes  : {df['Phoneme'].nunique():,}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Select features for clustering
#
# WHY: We use phonemes appearing in >= 5% of languages (66 phonemes).
# Too rare phonemes add noise — a phoneme in only 1 language gives no
# clustering signal. We want phonemes that meaningfully differentiate
# language groups from each other.
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 60)
log("STEP 2 — Selecting features (phonemes >= 5% frequency)")
log("=" * 60)

common = freq[freq["frequency_pct"] >= 5]["Phoneme"].tolist()
log(f"  Features selected: {len(common)} phonemes")

df_common = df[df["Phoneme"].isin(common)].copy()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Build binary presence matrix
#
# Rows = languages (Glottocode), Columns = phonemes
# Cell = 1 if language has phoneme, 0 otherwise
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 60)
log("STEP 3 — Building language x phoneme binary matrix")
log("=" * 60)

presence = (
    df_common.groupby(["Glottocode", "Phoneme"])
    .size()
    .unstack(fill_value=0)
    .clip(upper=1)
)

# Fill any missing phoneme columns with 0
for ph in common:
    if ph not in presence.columns:
        presence[ph] = 0
presence = presence[common]

# Add languages with no common phonemes as all-zero rows
all_langs = df["Glottocode"].unique()
missing   = [g for g in all_langs if g not in presence.index]
if missing:
    missing_df = pd.DataFrame(0, index=missing, columns=presence.columns)
    presence   = pd.concat([presence, missing_df])

presence = presence.astype(np.float32)
log(f"  Matrix shape : {presence.shape[0]} languages x {presence.shape[1]} phonemes")

# Keep only languages that have macroarea metadata
meta = lang_inv[["Glottocode", "LanguageName", "macroarea",
                  "family_name", "latitude", "longitude",
                  "total_phonemes", "has_click", "has_tone_segment"]].copy()
meta = meta[meta["macroarea"].notna()].drop_duplicates("Glottocode")

presence = presence[presence.index.isin(meta["Glottocode"])]
log(f"  After filtering to languages with metadata: {presence.shape[0]}")

X = presence.values   # numpy array for sklearn

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Find optimal K using Silhouette Score
#
# We test K = 2 to 10 and compute silhouette score for each.
# The K with the highest silhouette score is the most natural clustering.
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 60)
log("STEP 4 — Finding optimal K (Silhouette Score, K=2 to 10)")
log("=" * 60)

sil_scores = []
k_range    = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels = km.fit_predict(X)
    score  = silhouette_score(X, labels, metric="euclidean", sample_size=500,
                              random_state=42)
    sil_scores.append({"K": k, "silhouette_score": round(score, 4)})
    log(f"  K={k:2d}  Silhouette Score = {score:.4f}")

sil_df     = pd.DataFrame(sil_scores)
optimal_k  = int(sil_df.loc[sil_df["silhouette_score"].idxmax(), "K"])
log(f"\n  Optimal K = {optimal_k}  "
    f"(score = {sil_df['silhouette_score'].max():.4f})")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — K-Means with optimal K
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 60)
log(f"STEP 5 — K-Means clustering (K={optimal_k})")
log("=" * 60)

km_final = KMeans(n_clusters=optimal_k, random_state=42,
                  n_init=20, max_iter=500)
km_labels = km_final.fit_predict(X)

# Also run with K=6 (matching number of macroareas) for comparison
km6 = KMeans(n_clusters=6, random_state=42, n_init=20, max_iter=500)
km6_labels = km6.fit_predict(X)

log(f"  K-Means (K={optimal_k}) cluster sizes:")
unique, counts = np.unique(km_labels, return_counts=True)
for u, c in zip(unique, counts):
    log(f"    Cluster {u+1}: {c:,} languages")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Hierarchical clustering
#
# WHY: K-Means requires K and uses Euclidean distance.
# Hierarchical clustering is distance-agnostic and reveals
# the natural merging structure — which languages are most similar
# to each other before we impose any cluster count.
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 60)
log("STEP 6 — Hierarchical clustering (Ward linkage)")
log("=" * 60)

# Use a subsample for the dendrogram (full 2000+ is unreadable)
# Sample 200 languages stratified by macroarea for representative dendrogram
meta_indexed = meta.set_index("Glottocode")
sample_langs = []
for region in meta["macroarea"].unique():
    region_langs = meta[meta["macroarea"] == region]["Glottocode"].tolist()
    region_langs = [g for g in region_langs if g in presence.index]
    n_sample = min(30, len(region_langs))
    np.random.seed(42)
    sample_langs.extend(np.random.choice(region_langs, n_sample, replace=False))

X_sample    = presence.loc[sample_langs].values
Z           = linkage(X_sample, method="ward", metric="euclidean")
hier_labels = fcluster(Z, optimal_k, criterion="maxclust")

log(f"  Hierarchical sample size : {len(sample_langs)} languages")
log(f"  Clusters (K={optimal_k}):")
unique_h, counts_h = np.unique(hier_labels, return_counts=True)
for u, c in zip(unique_h, counts_h):
    log(f"    Cluster {u}: {c} languages")

# Full hierarchical on all languages
Z_full        = linkage(X, method="ward", metric="euclidean")
hier_full     = fcluster(Z_full, optimal_k, criterion="maxclust")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Attach cluster labels to language metadata
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 60)
log("STEP 7 — Attaching cluster labels to language metadata")
log("=" * 60)

lang_clusters = meta[meta["Glottocode"].isin(presence.index)].copy()
lang_clusters = lang_clusters.set_index("Glottocode")
lang_clusters["kmeans_cluster"]   = km_labels
lang_clusters["kmeans6_cluster"]  = km6_labels
lang_clusters["hier_cluster"]     = hier_full
lang_clusters["kmeans_cluster"]  += 1   # 1-indexed
lang_clusters["kmeans6_cluster"] += 1
lang_clusters = lang_clusters.reset_index()

log(f"  Languages with cluster labels: {len(lang_clusters):,}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Cluster vs Region crosstab
#
# WHAT: Count how many languages from each region fall in each cluster
# WHY : If a cluster = mostly one region → phoneme profiles encode geography
#       If clusters mix regions → phonology crosses geographic boundaries
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 60)
log("STEP 8 — Cluster vs Macroarea crosstab")
log("=" * 60)

crosstab_region = pd.crosstab(
    lang_clusters["kmeans_cluster"],
    lang_clusters["macroarea"],
    margins=True
)
log()
log(crosstab_region.to_string())

# Cluster purity = for each cluster, what % is the dominant region
log()
log("  Cluster purity by dominant region:")
for cluster_id in sorted(lang_clusters["kmeans_cluster"].unique()):
    sub = lang_clusters[lang_clusters["kmeans_cluster"] == cluster_id]
    dominant = sub["macroarea"].value_counts()
    top_region = dominant.index[0]
    purity = dominant.iloc[0] / len(sub) * 100
    log(f"    Cluster {cluster_id}: {top_region:<15} "
        f"({purity:.1f}% of {len(sub)} languages)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — Cluster phoneme profiles
#
# WHAT: For each cluster, what are the most and least common phonemes?
# WHY : This tells us WHAT makes each cluster distinct —
#       the linguistic interpretation of what the algorithm found.
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 60)
log("STEP 9 — Cluster phoneme profiles (centroid analysis)")
log("=" * 60)

centroids   = km_final.cluster_centers_
centroid_df = pd.DataFrame(
    centroids,
    columns=common,
    index=[f"Cluster_{i+1}" for i in range(optimal_k)]
)

log()
for i in range(optimal_k):
    row         = centroid_df.iloc[i]
    top5        = row.nlargest(5)
    bot5        = row.nsmallest(5)
    cluster_id  = i + 1
    n_lang      = (lang_clusters["kmeans_cluster"] == cluster_id).sum()
    dom_region  = (lang_clusters[lang_clusters["kmeans_cluster"] == cluster_id]
                   ["macroarea"].value_counts().index[0])
    log(f"  Cluster {cluster_id} ({n_lang} languages, dominant: {dom_region})")
    log(f"    Most common  : "
        f"{', '.join([f'{p}({v:.2f})' for p, v in top5.items()])}")
    log(f"    Least common : "
        f"{', '.join([f'{p}({v:.2f})' for p, v in bot5.items()])}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 — Save all outputs
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 60)
log("STEP 10 — Saving outputs")
log("=" * 60)

lang_clusters.to_csv(
    os.path.join(DATA_DIR, "language_clusters.csv"), index=False)
log("  Saved: data/processed/language_clusters.csv")

crosstab_region.to_csv(
    os.path.join(TABLES_DIR, "cluster_region_crosstab.csv"))
log("  Saved: cluster_region_crosstab.csv")

centroid_df.T.to_csv(
    os.path.join(TABLES_DIR, "cluster_phoneme_profile.csv"))
log("  Saved: cluster_phoneme_profile.csv")

sil_df.to_csv(
    os.path.join(TABLES_DIR, "silhouette_scores.csv"), index=False)
log("  Saved: silhouette_scores.csv")

# Save linkage matrix for dendrogram in visualization script
np.save(os.path.join(DATA_DIR, "linkage_matrix.npy"), Z)
np.save(os.path.join(DATA_DIR, "linkage_sample_langs.npy"),
        np.array(sample_langs))
log("  Saved: linkage_matrix.npy + linkage_sample_langs.npy")

with open(os.path.join(TABLES_DIR, "phase7_report.txt"),
          "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
log("  Saved: phase7_report.txt")

log()
log("[DONE] Phase 7 — Clustering complete.")
