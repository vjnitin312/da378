"""
PHASE 5 - SCRIPT 9
Purpose : Visualize pairwise phoneme statistics
Plots   :
    1.  Co-occurrence heatmap — top 30 phonemes (seaborn)
    2.  Jaccard similarity heatmap — top 30 phonemes (seaborn)
    3.  Phi coefficient heatmap — top 30 phonemes (seaborn)
    4.  Top 20 associated pairs — bar chart (matplotlib)
    5.  Top 20 exclusive pairs — bar chart (matplotlib)
    6.  Conditional frequency heatmap P(B|A) — top 20 (seaborn)
    7.  Interactive Jaccard heatmap (Plotly)
    8.  Interactive Phi heatmap (Plotly)
    9.  Interactive conditional frequency heatmap (Plotly)
    10. Implicational universals network chart (Plotly)
Outputs : outputs/figures/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.join(os.path.dirname(__file__), "..")
TABLES_DIR  = os.path.join(BASE_DIR, "outputs", "tables")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Load outputs from script 8 ────────────────────────────────────────────────
cooc_df    = pd.read_csv(os.path.join(TABLES_DIR, "cooccurrence_matrix.csv"),
                          index_col=0)
jaccard_df = pd.read_csv(os.path.join(TABLES_DIR, "jaccard_matrix.csv"),
                          index_col=0)
phi_df     = pd.read_csv(os.path.join(TABLES_DIR, "phi_matrix.csv"),
                          index_col=0)
top_assoc  = pd.read_csv(os.path.join(TABLES_DIR, "top_associated_pairs.csv"))
top_excl   = pd.read_csv(os.path.join(TABLES_DIR, "top_exclusive_pairs.csv"))
cond_pivot = pd.read_csv(os.path.join(TABLES_DIR,
                          "conditional_frequency_pivot.csv"), index_col=0)
impl_df    = pd.read_csv(os.path.join(TABLES_DIR, "implicational_universals.csv"))
freq       = pd.read_csv(os.path.join(TABLES_DIR, "phoneme_frequency.csv"))

# ── Select top 30 phonemes (by frequency) for static heatmaps ─────────────────
top30 = freq.nlargest(30, "language_count")["Phoneme"].tolist()
top30 = [p for p in top30 if p in cooc_df.index]

CLASS_COLORS = {
    "consonant" : "#2C7BB6",
    "vowel"     : "#D7191C",
    "tone"      : "#1A9641",
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPER — axis label colors by phoneme class
# ─────────────────────────────────────────────────────────────────────────────
def label_colors(phonemes):
    class_map = freq.set_index("Phoneme")["segment_class"].to_dict()
    return [CLASS_COLORS.get(class_map.get(p, "consonant"), "#333") for p in phonemes]

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — Co-occurrence heatmap (seaborn) — top 30
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Plot 1 — Co-occurrence heatmap (seaborn)")

arr_cooc = np.array(cooc_df.loc[top30, top30].values, dtype=float)
np.fill_diagonal(arr_cooc, 0)
sub_cooc = pd.DataFrame(arr_cooc, index=top30, columns=top30)   # zero diagonal so self never dominates

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(
    sub_cooc,
    ax=ax,
    cmap="YlOrRd",
    linewidths=0.3,
    linecolor="#dddddd",
    annot=False,
    cbar_kws={"label": "Number of languages containing both phonemes",
               "shrink": 0.75},
)
ax.set_title(
    "Phoneme Co-occurrence Matrix\n"
    "(Top 30 phonemes — diagonal set to 0)",
    fontsize=14, fontweight="bold", pad=15
)
ax.set_xlabel("Phoneme B", fontsize=11)
ax.set_ylabel("Phoneme A", fontsize=11)

# Color tick labels by class
for tick, ph in zip(ax.get_xticklabels(), top30):
    class_map = freq.set_index("Phoneme")["segment_class"].to_dict()
    tick.set_color(CLASS_COLORS.get(class_map.get(ph, "consonant"), "#333"))
for tick, ph in zip(ax.get_yticklabels(), top30):
    tick.set_color(CLASS_COLORS.get(class_map.get(ph, "consonant"), "#333"))

# Legend for label colors
from matplotlib.patches import Patch
legend_patches = [
    Patch(color=CLASS_COLORS["consonant"], label="Consonant"),
    Patch(color=CLASS_COLORS["vowel"],     label="Vowel"),
    Patch(color=CLASS_COLORS["tone"],      label="Tone"),
]
ax.legend(handles=legend_patches, loc="upper right",
          bbox_to_anchor=(1.18, 1.12), title="IPA label color")

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "plot16_cooccurrence_heatmap.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: plot16_cooccurrence_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — Jaccard similarity heatmap (seaborn) — top 30
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Plot 2 — Jaccard similarity heatmap (seaborn)")

arr_jacc = np.array(jaccard_df.loc[top30, top30].values, dtype=float)
np.fill_diagonal(arr_jacc, np.nan)
sub_jacc = pd.DataFrame(arr_jacc, index=top30, columns=top30)  # mask diagonal

fig, ax = plt.subplots(figsize=(14, 12))
mask = np.eye(len(top30), dtype=bool)
sns.heatmap(
    sub_jacc,
    ax=ax,
    cmap="Blues",
    mask=mask,
    linewidths=0.3,
    linecolor="#dddddd",
    vmin=0, vmax=1,
    annot=False,
    cbar_kws={"label": "Jaccard Similarity (0=never together, 1=always together)",
               "shrink": 0.75},
)
ax.set_title(
    "Jaccard Similarity Matrix\n"
    "(Top 30 phonemes — how often they co-occur relative to their union)",
    fontsize=13, fontweight="bold", pad=15
)
ax.set_xlabel("Phoneme B", fontsize=11)
ax.set_ylabel("Phoneme A", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "plot17_jaccard_heatmap.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: plot17_jaccard_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 — Phi coefficient heatmap (seaborn) — top 30
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Plot 3 — Phi coefficient heatmap (seaborn)")

arr_phi = np.array(phi_df.loc[top30, top30].values, dtype=float)
np.fill_diagonal(arr_phi, np.nan)
sub_phi = pd.DataFrame(arr_phi, index=top30, columns=top30)

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(
    sub_phi,
    ax=ax,
    cmap="RdBu",
    center=0,
    mask=mask,
    linewidths=0.3,
    linecolor="#dddddd",
    vmin=-0.15, vmax=0.15,
    annot=False,
    cbar_kws={"label": "Phi coefficient (blue=positive, red=negative association)",
               "shrink": 0.75},
)
ax.set_title(
    "Phi Coefficient Matrix\n"
    "(Blue = tend to co-occur, Red = tend to avoid each other)",
    fontsize=13, fontweight="bold", pad=15
)
ax.set_xlabel("Phoneme B", fontsize=11)
ax.set_ylabel("Phoneme A", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "plot18_phi_heatmap.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: plot18_phi_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4 — Top 20 associated pairs bar chart (matplotlib)
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Plot 4 — Top associated pairs bar chart")

top_assoc["pair"] = top_assoc["phoneme_A"] + " & " + top_assoc["phoneme_B"]
top_assoc_plot = top_assoc.head(20).sort_values("jaccard", ascending=True)
colors = [CLASS_COLORS.get(t.split("-")[0], "#2C7BB6")
          for t in top_assoc_plot["pair_type"]]

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(top_assoc_plot["pair"], top_assoc_plot["jaccard"],
               color=colors, edgecolor="white", height=0.65)
for bar, val in zip(bars, top_assoc_plot["jaccard"]):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", ha="left", fontsize=9)

ax.set_xlabel("Jaccard Similarity", fontsize=11)
ax.set_title("Top 20 Most Co-occurring Phoneme Pairs\n(by Jaccard Similarity)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlim(0, 1.05)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "plot19_top_associated_pairs.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: plot19_top_associated_pairs.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 5 — Top 20 exclusive pairs bar chart (matplotlib)
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Plot 5 — Top exclusive pairs bar chart")

top_excl["pair"] = top_excl["phoneme_A"] + " & " + top_excl["phoneme_B"]
top_excl_plot = top_excl.head(20).sort_values("phi", ascending=False)

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(top_excl_plot["pair"], top_excl_plot["phi"],
               color="#D7191C", edgecolor="white", height=0.65)
for bar, val in zip(bars, top_excl_plot["phi"]):
    ax.text(bar.get_width() - 0.001,
            bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", ha="right", fontsize=9, color="white")

ax.set_xlabel("Phi Coefficient", fontsize=11)
ax.set_title("Top 20 Most Mutually Exclusive Phoneme Pairs\n(by Phi Coefficient)",
             fontsize=13, fontweight="bold", pad=12)
ax.axvline(0, color="#333", linewidth=0.8, linestyle="--")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "plot20_top_exclusive_pairs.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: plot20_top_exclusive_pairs.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 6 — Conditional frequency heatmap P(B|A) — seaborn
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Plot 6 — Conditional frequency heatmap (seaborn)")

top20 = freq.nlargest(20, "language_count")["Phoneme"].tolist()
top20 = [p for p in top20 if p in cond_pivot.index and p in cond_pivot.columns]
arr_cond = np.array(cond_pivot.loc[top20, top20].values, dtype=float)
np.fill_diagonal(arr_cond, np.nan)
cond_sub = pd.DataFrame(arr_cond, index=top20, columns=top20)

fig, ax = plt.subplots(figsize=(13, 11))
sns.heatmap(
    cond_sub,
    ax=ax,
    cmap="Greens",
    vmin=0, vmax=1,
    linewidths=0.4,
    linecolor="#dddddd",
    annot=True,
    fmt=".2f",
    annot_kws={"size": 8},
    cbar_kws={"label": "P(B | A) — probability of B given A",
               "shrink": 0.75},
    mask=np.eye(len(top20), dtype=bool),
)
ax.set_title(
    "Conditional Frequency P(B | A)\n"
    "Row = Given phoneme A, Column = Probability of B",
    fontsize=13, fontweight="bold", pad=15
)
ax.set_xlabel("Phoneme B", fontsize=11)
ax.set_ylabel("Given Phoneme A", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "plot21_conditional_freq_heatmap.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: plot21_conditional_freq_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 7 — Interactive Jaccard heatmap (Plotly)
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Plot 7 — Interactive Jaccard heatmap (Plotly)")

top25 = freq.nlargest(25, "language_count")["Phoneme"].tolist()
top25 = [p for p in top25 if p in jaccard_df.index]
arr_j = np.array(jaccard_df.loc[top25, top25].values, dtype=float)
np.fill_diagonal(arr_j, np.nan)
sub_j = pd.DataFrame(arr_j, index=top25, columns=top25)

fig7 = go.Figure(data=go.Heatmap(
    z            = sub_j.values,
    x            = top25,
    y            = top25,
    colorscale   = "Blues",
    zmin=0, zmax=1,
    text         = np.round(sub_j.values, 3),
    texttemplate = "%{text}",
    textfont     = {"size": 9},
    hoverongaps  = False,
    colorbar     = dict(title="Jaccard"),
))
fig7.update_layout(
    title      = "Jaccard Similarity — Top 25 Phonemes",
    title_font_size = 15,
    xaxis_title = "Phoneme B",
    yaxis_title = "Phoneme A",
    height     = 650,
    margin     = dict(l=60, r=20, t=60, b=60),
)
fig7.write_html(os.path.join(FIGURES_DIR, "plot22_jaccard_interactive.html"))
print("  Saved: plot22_jaccard_interactive.html")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 8 — Interactive Phi heatmap (Plotly)
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Plot 8 — Interactive Phi heatmap (Plotly)")

arr_p = np.array(phi_df.loc[top25, top25].values, dtype=float)
np.fill_diagonal(arr_p, np.nan)
sub_p = pd.DataFrame(arr_p, index=top25, columns=top25)

fig8 = go.Figure(data=go.Heatmap(
    z            = sub_p.values,
    x            = top25,
    y            = top25,
    colorscale   = "RdBu",
    zmid         = 0,
    text         = np.round(sub_p.values, 3),
    texttemplate = "%{text}",
    textfont     = {"size": 9},
    hoverongaps  = False,
    colorbar     = dict(title="Phi (φ)"),
))
fig8.update_layout(
    title      = "Phi Coefficient Matrix — Top 25 Phonemes",
    title_font_size = 15,
    xaxis_title = "Phoneme B",
    yaxis_title = "Phoneme A",
    height     = 650,
    margin     = dict(l=60, r=20, t=60, b=60),
)
fig8.write_html(os.path.join(FIGURES_DIR, "plot23_phi_interactive.html"))
print("  Saved: plot23_phi_interactive.html")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 9 — Interactive conditional frequency heatmap (Plotly)
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Plot 9 — Interactive conditional frequency (Plotly)")

arr_csp = np.array(cond_pivot.loc[top20, top20].values, dtype=float)
np.fill_diagonal(arr_csp, np.nan)
cond_sub_plot = pd.DataFrame(arr_csp, index=top20, columns=top20)

fig9 = go.Figure(data=go.Heatmap(
    z            = cond_sub_plot.values,
    x            = top20,
    y            = top20,
    colorscale   = "Greens",
    zmin=0, zmax=1,
    text         = np.round(cond_sub_plot.values, 3),
    texttemplate = "%{text}",
    textfont     = {"size": 9},
    hoverongaps  = False,
    colorbar     = dict(title="P(B|A)"),
))
fig9.update_layout(
    title      = "Conditional Frequency P(B | A) — Top 20 Phonemes",
    title_font_size = 15,
    xaxis_title = "Phoneme B (given A in row)",
    yaxis_title = "Given Phoneme A",
    height     = 600,
    margin     = dict(l=60, r=20, t=60, b=60),
)
fig9.write_html(os.path.join(FIGURES_DIR, "plot24_conditional_interactive.html"))
print("  Saved: plot24_conditional_interactive.html")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 10 — Implicational universals bubble chart (Plotly)
#
# WHAT : Top 30 implicational pairs plotted as bubbles
#        x = n(A) (how common is the given phoneme)
#        y = P(B|A) (how strongly does A imply B)
#        size = n(A,B) (how many languages have both)
#        color = phoneme B
# WHY  : Shows which phoneme implications are both strong AND cover many
#        languages — the most typologically significant universals
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Plot 10 — Implicational universals bubble chart (Plotly)")

impl_top = impl_df.head(30).copy()
impl_top["label"] = impl_top["given_A"] + " → " + impl_top["phoneme_B"]

fig10 = px.scatter(
    impl_top,
    x         = "n_A",
    y         = "P(B|A)",
    size      = "n_AB",
    color     = "phoneme_B",
    text      = "label",
    hover_data= {"given_A": True, "phoneme_B": True,
                 "P(B|A)": True, "n_A": True, "n_AB": True},
    labels    = {
        "n_A"      : "Languages with phoneme A",
        "P(B|A)"   : "P(B | A)",
        "n_AB"     : "Languages with both",
        "phoneme_B": "Phoneme B",
        "given_A"  : "Given A",
    },
    title     = "Implicational Universals — P(B|A) ≥ 0.90",
    size_max  = 30,
)
fig10.update_traces(textposition="top center", textfont_size=9)
fig10.update_layout(
    plot_bgcolor = "white",
    yaxis = dict(gridcolor="#eeeeee", title="P(B | A)",
                 range=[0.88, 1.01]),
    xaxis = dict(gridcolor="#eeeeee",
                 title="Number of languages containing A"),
    height       = 550,
    title_font_size = 15,
    showlegend   = False,
    margin       = dict(l=10, r=10, t=60, b=10),
)
fig10.write_html(os.path.join(FIGURES_DIR,
                               "plot25_implicational_universals.html"))
print("  Saved: plot25_implicational_universals.html")

print()
print("[DONE] All Phase 5 visualizations saved to outputs/figures/")
