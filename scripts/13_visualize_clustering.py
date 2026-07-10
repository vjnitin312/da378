"""
PHASE 7 - SCRIPT 13
Purpose : Visualize clustering results
Plots   :
    1.  Silhouette score vs K — line plot (matplotlib)
    2.  Cluster size bar chart (matplotlib)
    3.  Cluster vs Region stacked bar (matplotlib)
    4.  Dendrogram — hierarchical clustering sample (matplotlib)
    5.  Cluster purity heatmap (seaborn)
    6.  Cluster phoneme profile heatmap (seaborn)
    7.  Interactive world map — languages coloured by cluster (Plotly)
    8.  Interactive cluster vs region bar (Plotly)
    9.  Interactive silhouette score plot (Plotly)
Outputs : outputs/figures/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import dendrogram
import os
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.join(os.path.dirname(__file__), "..")
TABLES_DIR  = os.path.join(BASE_DIR, "outputs", "tables")
DATA_DIR    = os.path.join(BASE_DIR, "data", "processed")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Load outputs ──────────────────────────────────────────────────────────────
lang_cl   = pd.read_csv(os.path.join(DATA_DIR,   "language_clusters.csv"))
sil_df    = pd.read_csv(os.path.join(TABLES_DIR, "silhouette_scores.csv"))
crosstab  = pd.read_csv(os.path.join(TABLES_DIR, "cluster_region_crosstab.csv"),
                         index_col=0)
profile   = pd.read_csv(os.path.join(TABLES_DIR, "cluster_phoneme_profile.csv"),
                         index_col=0)
Z         = np.load(os.path.join(DATA_DIR, "linkage_matrix.npy"))
sample_langs = np.load(os.path.join(DATA_DIR, "linkage_sample_langs.npy"),
                        allow_pickle=True).tolist()

OPTIMAL_K = int(sil_df.loc[sil_df["silhouette_score"].idxmax(), "K"])

# ── Color palettes ────────────────────────────────────────────────────────────
REGION_COLORS = {
    "Africa"        : "#E66101",
    "Eurasia"       : "#5E3C99",
    "South America" : "#008837",
    "Australia"     : "#FDB863",
    "North America" : "#B2ABD2",
    "Papunesia"     : "#35978F",
}
CLUSTER_PALETTE = px.colors.qualitative.Bold

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — Silhouette score vs K (matplotlib)
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Plot 1 — Silhouette score vs K")

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(sil_df["K"], sil_df["silhouette_score"],
        marker="o", color="#2C7BB6", linewidth=2.2,
        markersize=8, markerfacecolor="white", markeredgewidth=2)
ax.axvline(OPTIMAL_K, color="#D7191C", linestyle="--", linewidth=1.5,
           label=f"Optimal K = {OPTIMAL_K}")
ax.scatter([OPTIMAL_K],
           [sil_df.loc[sil_df["K"] == OPTIMAL_K, "silhouette_score"].values[0]],
           color="#D7191C", s=120, zorder=5)
for _, row in sil_df.iterrows():
    ax.annotate(f"{row['silhouette_score']:.3f}",
                (row["K"], row["silhouette_score"]),
                textcoords="offset points", xytext=(0, 8),
                fontsize=8, ha="center", color="#333")
ax.set_xlabel("Number of Clusters (K)", fontsize=11)
ax.set_ylabel("Silhouette Score", fontsize=11)
ax.set_title("Silhouette Score vs Number of Clusters\n"
             "(Higher = more natural clustering)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.set_xticks(sil_df["K"].tolist())
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "plot28_silhouette_vs_k.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: plot28_silhouette_vs_k.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — Cluster size bar chart (matplotlib)
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Plot 2 — Cluster size bar chart")

cluster_sizes = (lang_cl.groupby("kmeans_cluster")
                 .agg(n_languages=("Glottocode", "count"),
                      dominant_region=("macroarea",
                                       lambda x: x.value_counts().index[0]))
                 .reset_index())

fig, ax = plt.subplots(figsize=(10, 5))
colors_c = [REGION_COLORS.get(r, "#999")
            for r in cluster_sizes["dominant_region"]]
bars = ax.bar(
    [f"C{int(c)}\n({r[:3]})" for c, r in
     zip(cluster_sizes["kmeans_cluster"],
         cluster_sizes["dominant_region"])],
    cluster_sizes["n_languages"],
    color=colors_c, edgecolor="white", width=0.65
)
for bar, val in zip(bars, cluster_sizes["n_languages"]):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 4,
            str(val), ha="center", va="bottom", fontsize=10)

ax.set_xlabel("Cluster (dominant region abbreviated)", fontsize=11)
ax.set_ylabel("Number of Languages", fontsize=11)
ax.set_title(f"K-Means Cluster Sizes (K={OPTIMAL_K})\n"
             "Bar colour = dominant geographic region",
             fontsize=12, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.4)

# Legend
from matplotlib.patches import Patch
legend_patches = [Patch(facecolor=c, label=r)
                  for r, c in REGION_COLORS.items()]
ax.legend(handles=legend_patches, fontsize=9,
          loc="upper right", ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "plot29_cluster_sizes.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: plot29_cluster_sizes.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 — Cluster vs Region stacked bar (matplotlib)
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Plot 3 — Cluster vs region stacked bar")

ct = crosstab.drop("All", errors="ignore").drop("All", axis=1, errors="ignore")
ct = ct.astype(float)

# Normalize to % per cluster
ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(12, 5))
bottom = np.zeros(len(ct_pct))
regions = ct_pct.columns.tolist()

for region in regions:
    color = REGION_COLORS.get(region, "#999999")
    vals  = ct_pct[region].values
    bars  = ax.bar(
        [f"C{int(i)}" for i in ct_pct.index],
        vals, bottom=bottom,
        label=region, color=color, edgecolor="white"
    )
    for i, (bar, val) in enumerate(zip(bars, vals)):
        if val > 8:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bottom[i] + val/2,
                    f"{val:.0f}%",
                    ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")
    bottom += vals

ax.set_xlabel("K-Means Cluster", fontsize=11)
ax.set_ylabel("% of languages", fontsize=11)
ax.set_title(f"Geographic Composition of Each Cluster (K={OPTIMAL_K})\n"
             "Colour = Macroarea",
             fontsize=12, fontweight="bold")
ax.legend(loc="upper right", fontsize=9, ncol=2)
ax.set_ylim(0, 110)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "plot30_cluster_region_stacked.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: plot30_cluster_region_stacked.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4 — Dendrogram (matplotlib)
#
# WHAT: Shows the hierarchical merging structure of 180 sampled languages
# WHY : Reveals at what similarity level language groups merge —
#       i.e. which languages are phonologically closest to each other
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Plot 4 — Dendrogram (hierarchical clustering)")

# Build region labels for the sample
lang_cl_indexed = lang_cl.set_index("Glottocode")
sample_regions  = []
for g in sample_langs:
    if g in lang_cl_indexed.index:
        sample_regions.append(lang_cl_indexed.loc[g, "macroarea"])
    else:
        sample_regions.append("Unknown")

# Color leaves by region
leaf_colors = [REGION_COLORS.get(r, "#999999") for r in sample_regions]

fig, ax = plt.subplots(figsize=(18, 7))
dend = dendrogram(
    Z,
    ax=ax,
    no_labels=True,
    color_threshold=0.6 * max(Z[:, 2]),
    above_threshold_color="#aaaaaa",
    leaf_rotation=90,
)

ax.set_title(
    "Hierarchical Clustering Dendrogram (Ward Linkage)\n"
    f"180 sampled languages — coloured by macroarea",
    fontsize=13, fontweight="bold"
)
ax.set_xlabel("Language samples", fontsize=11)
ax.set_ylabel("Distance (Ward linkage)", fontsize=11)
ax.spines[["top", "right"]].set_visible(False)

# Add region legend
from matplotlib.patches import Patch
legend_patches = [Patch(facecolor=c, label=r)
                  for r, c in REGION_COLORS.items()]
ax.legend(handles=legend_patches, fontsize=9,
          loc="upper right", title="Macroarea")

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "plot31_dendrogram.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: plot31_dendrogram.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 5 — Cluster purity heatmap (seaborn)
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Plot 5 — Cluster purity heatmap")

ct_pct_heat = ct_pct.copy()
ct_pct_heat.index = [f"Cluster {int(i)}" for i in ct_pct_heat.index]

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(
    ct_pct_heat,
    ax=ax,
    cmap="YlOrRd",
    annot=True,
    fmt=".1f",
    linewidths=0.5,
    linecolor="#dddddd",
    cbar_kws={"label": "% of cluster languages", "shrink": 0.75},
)
ax.set_title(
    "Cluster Purity Heatmap\n"
    "(% of each cluster belonging to each macroarea)",
    fontsize=13, fontweight="bold", pad=15
)
ax.set_xlabel("Macroarea", fontsize=11)
ax.set_ylabel("K-Means Cluster", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "plot32_cluster_purity_heatmap.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: plot32_cluster_purity_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 6 — Cluster phoneme profile heatmap (seaborn)
#
# WHAT: Each row = cluster centroid, each column = phoneme
#       Cell value = avg presence (0–1) of that phoneme in the cluster
# WHY : Reveals WHAT phonological features define each cluster
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Plot 6 — Cluster phoneme profile heatmap")

# Show top 25 most variable phonemes across clusters (most informative)
profile_t  = profile.T.copy()   # now shape: clusters x phonemes
variability = profile_t.std(axis=0).nlargest(25).index.tolist()
profile_sub = profile_t[variability]
profile_sub.index = [f"C{i+1}" for i in range(len(profile_sub))]

fig, ax = plt.subplots(figsize=(16, 5))
sns.heatmap(
    profile_sub,
    ax=ax,
    cmap="Blues",
    vmin=0, vmax=1,
    linewidths=0.4,
    linecolor="#dddddd",
    annot=True,
    fmt=".2f",
    annot_kws={"size": 8},
    cbar_kws={"label": "Avg presence (0=absent, 1=universal)",
               "shrink": 0.6},
)
ax.set_title(
    "Cluster Phoneme Profiles — Top 25 Most Discriminating Phonemes\n"
    "(Cluster centroids: avg phoneme presence per cluster)",
    fontsize=12, fontweight="bold", pad=15
)
ax.set_xlabel("Phoneme", fontsize=11)
ax.set_ylabel("Cluster", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "plot33_cluster_phoneme_profile.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: plot33_cluster_phoneme_profile.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 7 — Interactive world map coloured by cluster (Plotly)
#
# WHAT: Each language = dot on world map, coloured by K-Means cluster
# WHY : Visually tests if clusters are geographically coherent —
#       do same-cluster languages live in the same region?
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Plot 7 — Interactive world map by cluster (Plotly)")

map_df = lang_cl[
    lang_cl["latitude"].notna() &
    lang_cl["longitude"].notna()
].copy()
map_df["Cluster"] = "Cluster " + map_df["kmeans_cluster"].astype(str)

fig_map = px.scatter_geo(
    map_df,
    lat="latitude",
    lon="longitude",
    color="Cluster",
    hover_name="LanguageName",
    hover_data={
        "macroarea"      : True,
        "family_name"    : True,
        "total_phonemes" : True,
        "kmeans_cluster" : False,
        "latitude"       : False,
        "longitude"      : False,
    },
    color_discrete_sequence=CLUSTER_PALETTE,
    projection="natural earth",
    labels={
        "Cluster"        : "K-Means Cluster",
        "macroarea"      : "Region",
        "family_name"    : "Family",
        "total_phonemes" : "Inventory Size",
    },
    title=f"Language Clusters on World Map (K-Means, K={OPTIMAL_K})",
)
fig_map.update_traces(marker=dict(size=5, opacity=0.8))
fig_map.update_layout(
    geo=dict(
        showland=True, landcolor="#f5f5f0",
        showocean=True, oceancolor="#cce5f0",
        showcoastlines=True, coastlinecolor="#aaaaaa",
        showframe=False,
    ),
    title_font_size=15,
    height=560,
    margin=dict(l=0, r=0, t=50, b=0),
    legend=dict(title="Cluster", orientation="v"),
)
fig_map.write_html(os.path.join(FIGURES_DIR, "plot34_cluster_world_map.html"))
print("  Saved: plot34_cluster_world_map.html")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 8 — Interactive cluster vs region bar (Plotly)
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Plot 8 — Interactive cluster vs region (Plotly)")

ct_long = ct_pct.reset_index().melt(
    id_vars="kmeans_cluster",
    var_name="macroarea",
    value_name="pct"
)
ct_long["Cluster"] = "Cluster " + ct_long["kmeans_cluster"].astype(str)
ct_long["pct"]     = ct_long["pct"].round(1)

fig_bar = px.bar(
    ct_long,
    x="Cluster",
    y="pct",
    color="macroarea",
    color_discrete_map=REGION_COLORS,
    text="pct",
    labels={
        "pct"      : "% of cluster",
        "Cluster"  : "K-Means Cluster",
        "macroarea": "Macroarea",
    },
    title=f"Geographic Composition of Each Cluster (K={OPTIMAL_K})",
)
fig_bar.update_traces(
    texttemplate="%{text:.0f}%",
    textposition="inside",
)
fig_bar.update_layout(
    barmode="stack",
    plot_bgcolor="white",
    yaxis=dict(gridcolor="#eeeeee", title="% of cluster languages"),
    legend_title="Macroarea",
    height=480,
    title_font_size=15,
    margin=dict(l=10, r=10, t=50, b=10),
)
fig_bar.write_html(os.path.join(FIGURES_DIR, "plot35_cluster_region_interactive.html"))
print("  Saved: plot35_cluster_region_interactive.html")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 9 — Interactive silhouette score (Plotly)
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Plot 9 — Interactive silhouette score (Plotly)")

fig_sil = go.Figure()
fig_sil.add_trace(go.Scatter(
    x=sil_df["K"],
    y=sil_df["silhouette_score"],
    mode="lines+markers+text",
    text=[f"{v:.3f}" for v in sil_df["silhouette_score"]],
    textposition="top center",
    line=dict(color="#2C7BB6", width=2.5),
    marker=dict(size=10, color="#2C7BB6",
                line=dict(color="white", width=2)),
    name="Silhouette Score",
))
fig_sil.add_vline(
    x=OPTIMAL_K,
    line_dash="dash",
    line_color="#D7191C",
    annotation_text=f"Optimal K={OPTIMAL_K}",
    annotation_position="top right",
    annotation_font_color="#D7191C",
)
fig_sil.update_layout(
    title="Silhouette Score vs Number of Clusters",
    title_font_size=15,
    plot_bgcolor="white",
    xaxis=dict(title="Number of Clusters (K)",
               gridcolor="#eeeeee", dtick=1),
    yaxis=dict(title="Silhouette Score",
               gridcolor="#eeeeee"),
    height=420,
    margin=dict(l=10, r=10, t=50, b=10),
    showlegend=False,
)
fig_sil.write_html(os.path.join(FIGURES_DIR, "plot36_silhouette_interactive.html"))
print("  Saved: plot36_silhouette_interactive.html")

print()
print("[DONE] All Phase 7 visualizations saved to outputs/figures/")
