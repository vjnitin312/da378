"""
PHASE 6 - SCRIPT 11
Purpose : Generate a single comprehensive summary figure combining
          all key findings from Phases 2-5
Plots   :
    1.  Master summary figure (6 subplots, static PNG)
    2.  Interactive project summary dashboard panel (Plotly HTML)
Outputs : outputs/figures/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import os
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.join(os.path.dirname(__file__), "..")
TABLES_DIR  = os.path.join(BASE_DIR, "outputs", "tables")
DATA_DIR    = os.path.join(BASE_DIR, "data", "processed")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

freq       = pd.read_csv(os.path.join(TABLES_DIR, "phoneme_frequency.csv"))
region_inv = pd.read_csv(os.path.join(TABLES_DIR, "region_inventory_stats.csv"))
top_assoc  = pd.read_csv(os.path.join(TABLES_DIR, "top_associated_pairs.csv"))
top_excl   = pd.read_csv(os.path.join(TABLES_DIR, "top_exclusive_pairs.csv"))
impl_df    = pd.read_csv(os.path.join(TABLES_DIR, "implicational_universals.csv"))
lang_inv   = pd.read_csv(os.path.join(DATA_DIR,   "language_inventory.csv"))
region_cls = pd.read_csv(os.path.join(TABLES_DIR, "region_class_distribution.csv"))
summary    = pd.read_csv(os.path.join(TABLES_DIR, "project_summary_stats.csv"))

for col in ["has_click", "has_tone_segment"]:
    lang_inv[col] = (lang_inv[col].astype(str).str.strip().str.lower()
                     .map({"true": True, "false": False, "1": True, "0": False})
                     .fillna(False))

CLASS_COLORS = {
    "consonant" : "#2C7BB6",
    "vowel"     : "#D7191C",
    "tone"      : "#1A9641",
}
REGION_COLORS = {
    "Africa"        : "#E66101",
    "Eurasia"       : "#5E3C99",
    "South America" : "#008837",
    "Australia"     : "#FDB863",
    "North America" : "#B2ABD2",
    "Papunesia"     : "#35978F",
}

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — Master Summary Figure (6 subplots)
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Plot 1 — Master summary figure (6 subplots)")

fig = plt.figure(figsize=(20, 22))
fig.patch.set_facecolor("#FAFAFA")
gs  = gridspec.GridSpec(
    3, 3,
    figure=fig,
    hspace=0.42,
    wspace=0.38,
    left=0.07, right=0.97,
    top=0.93,  bottom=0.05,
)

# ── SUB 1: Top 10 phonemes ───────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])   # spans first 2 columns
top10 = freq.head(10).copy()
colors10 = [CLASS_COLORS[c] for c in top10["segment_class"]]
bars = ax1.barh(
    top10["Phoneme"][::-1],
    top10["frequency_pct"][::-1],
    color=colors10[::-1],
    edgecolor="white", height=0.65
)
for bar, pct in zip(bars, top10["frequency_pct"][::-1]):
    ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
             f"{pct:.1f}%", va="center", ha="left", fontsize=9, color="#333")
ax1.set_xlabel("Frequency (% of 2,176 languages)", fontsize=10)
ax1.set_title("Top 10 Most Common Phonemes Globally",
              fontsize=12, fontweight="bold")
ax1.set_xlim(0, 108)
ax1.xaxis.set_major_formatter(mtick.PercentFormatter())
ax1.spines[["top", "right"]].set_visible(False)
ax1.grid(axis="x", linestyle="--", alpha=0.3)
legend_patches = [
    mpatches.Patch(color=CLASS_COLORS["consonant"], label="Consonant"),
    mpatches.Patch(color=CLASS_COLORS["vowel"],     label="Vowel"),
]
ax1.legend(handles=legend_patches, fontsize=9, loc="lower right")

# ── SUB 2: Zipfian distribution ───────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
ax2.hist(freq["frequency_pct"], bins=60,
         color="#2C7BB6", edgecolor="white", alpha=0.85)
ax2.set_xscale("log")
ax2.set_xlabel("Frequency % (log scale)", fontsize=10)
ax2.set_ylabel("Number of phonemes", fontsize=10)
ax2.set_title("Zipfian Frequency Distribution\n(2,721 phonemes)",
              fontsize=11, fontweight="bold")
ax2.spines[["top", "right"]].set_visible(False)
ax2.grid(axis="y", linestyle="--", alpha=0.3)
ax2.annotate("18 phonemes\nin >50% languages",
             xy=(50, 2), xytext=(8, 80),
             arrowprops=dict(arrowstyle="->", color="#333"),
             fontsize=8, color="#333")

# ── SUB 3: Avg inventory by region ───────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
reg_sorted = region_inv.sort_values("avg_inventory", ascending=True)
colors_reg = [REGION_COLORS[r] for r in reg_sorted["macroarea"]]
bars3 = ax3.barh(reg_sorted["macroarea"], reg_sorted["avg_inventory"],
                 color=colors_reg, edgecolor="white", height=0.6)
for bar, val in zip(bars3, reg_sorted["avg_inventory"]):
    ax3.text(bar.get_width() + 0.2,
             bar.get_y() + bar.get_height()/2,
             f"{val:.1f}", va="center", ha="left", fontsize=9)
ax3.set_xlabel("Avg Inventory Size", fontsize=10)
ax3.set_title("Avg Inventory\nby Region", fontsize=11, fontweight="bold")
ax3.spines[["top", "right"]].set_visible(False)
ax3.grid(axis="x", linestyle="--", alpha=0.3)
ax3.set_xlim(0, 55)

# ── SUB 4: Tone & click % by region ──────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
reg_tc = region_inv.sort_values("pct_with_tones", ascending=False)
x4 = np.arange(len(reg_tc))
w  = 0.35
b1 = ax4.bar(x4 - w/2, reg_tc["pct_with_tones"],  w,
             label="% Tonal", color="#1A9641", edgecolor="white")
b2 = ax4.bar(x4 + w/2, reg_tc["pct_with_clicks"], w,
             label="% Clicks", color="#8B0000", edgecolor="white")
ax4.set_xticks(x4)
ax4.set_xticklabels(reg_tc["macroarea"], rotation=30,
                    ha="right", fontsize=8)
ax4.set_ylabel("% of languages", fontsize=10)
ax4.set_title("Tone & Click\nby Region", fontsize=11, fontweight="bold")
ax4.legend(fontsize=8)
ax4.spines[["top", "right"]].set_visible(False)
ax4.grid(axis="y", linestyle="--", alpha=0.3)

# ── SUB 5: Inventory box by region ───────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
lang_clean = lang_inv[lang_inv["macroarea"].notna()].copy()
order5 = (lang_clean.groupby("macroarea")["total_phonemes"]
          .median().sort_values(ascending=False).index.tolist())
palette5 = [REGION_COLORS[r] for r in order5]
parts = ax5.violinplot(
    [lang_clean[lang_clean["macroarea"] == r]["total_phonemes"].values
     for r in order5],
    positions=range(len(order5)),
    showmedians=True,
    showextrema=False,
)
for i, (pc, color) in enumerate(zip(parts["bodies"], palette5)):
    pc.set_facecolor(color)
    pc.set_alpha(0.7)
parts["cmedians"].set_color("#333")
ax5.set_xticks(range(len(order5)))
ax5.set_xticklabels(order5, rotation=30, ha="right", fontsize=8)
ax5.set_ylabel("Total phonemes", fontsize=10)
ax5.set_title("Inventory Distribution\n(Violin plot by region)",
              fontsize=11, fontweight="bold")
ax5.spines[["top", "right"]].set_visible(False)
ax5.grid(axis="y", linestyle="--", alpha=0.3)

# ── SUB 6: Top 15 associated pairs ───────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, :2])
top15_assoc = top_assoc.head(15).copy()
top15_assoc["pair"] = (top15_assoc["phoneme_A"] + " & "
                       + top15_assoc["phoneme_B"])
top15_sorted = top15_assoc.sort_values("jaccard", ascending=True)
col6 = [CLASS_COLORS.get(t.split("-")[0], "#2C7BB6")
        for t in top15_sorted["pair_type"]]
bars6 = ax6.barh(top15_sorted["pair"], top15_sorted["jaccard"],
                 color=col6, edgecolor="white", height=0.65)
for bar, val in zip(bars6, top15_sorted["jaccard"]):
    ax6.text(bar.get_width() + 0.003,
             bar.get_y() + bar.get_height()/2,
             f"{val:.3f}", va="center", ha="left", fontsize=9)
ax6.set_xlabel("Jaccard Similarity", fontsize=10)
ax6.set_title("Top 15 Co-occurring Phoneme Pairs (Jaccard Similarity)",
              fontsize=12, fontweight="bold")
ax6.set_xlim(0, 1.05)
ax6.spines[["top", "right"]].set_visible(False)
ax6.grid(axis="x", linestyle="--", alpha=0.3)

# ── SUB 7: Top 10 implicational universals ────────────────────────────────────
ax7 = fig.add_subplot(gs[2, 2])
impl_top10 = impl_df.head(10).copy()
impl_top10["label"] = impl_top10["given_A"] + "→" + impl_top10["phoneme_B"]
impl_sorted = impl_top10.sort_values("P(B|A)", ascending=True)
bars7 = ax7.barh(impl_sorted["label"], impl_sorted["P(B|A)"],
                 color="#5E3C99", edgecolor="white", height=0.65)
for bar, val in zip(bars7, impl_sorted["P(B|A)"]):
    ax7.text(bar.get_width() - 0.002,
             bar.get_y() + bar.get_height()/2,
             f"{val:.3f}", va="center", ha="right",
             fontsize=8, color="white", fontweight="bold")
ax7.set_xlabel("P(B | A)", fontsize=10)
ax7.set_xlim(0.87, 1.01)
ax7.set_title("Top 10 Implicational\nUniversals P(B|A) ≥ 0.90",
              fontsize=11, fontweight="bold")
ax7.spines[["top", "right"]].set_visible(False)
ax7.grid(axis="x", linestyle="--", alpha=0.3)

# ── Master title ──────────────────────────────────────────────────────────────
fig.suptitle(
    "Phoneme Distribution Across Languages — Key Findings Summary\n"
    "PHOIBLE v2.0  ·  2,176 Languages  ·  2,721 Phonemes  ·  177 Families",
    fontsize=15, fontweight="bold", y=0.97
)

path_master = os.path.join(FIGURES_DIR, "plot26_master_summary.png")
plt.savefig(path_master, dpi=150, bbox_inches="tight",
            facecolor="#FAFAFA")
plt.close()
print(f"  Saved: {path_master}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — Interactive project summary (Plotly subplots)
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Plot 2 — Interactive summary dashboard (Plotly)")

fig_int = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        "Top 10 Phonemes Globally",
        "Avg Inventory by Region",
        "Tone & Click % by Region",
        "Top 10 Implicational Universals",
    ],
    horizontal_spacing=0.12,
    vertical_spacing=0.18,
)

# Panel 1: Top 10 phonemes
top10_p = freq.head(10).copy()
fig_int.add_trace(
    go.Bar(
        x=top10_p["frequency_pct"],
        y=top10_p["Phoneme"],
        orientation="h",
        marker_color=[CLASS_COLORS[c] for c in top10_p["segment_class"]],
        text=[f"{v:.1f}%" for v in top10_p["frequency_pct"]],
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>Frequency: %{x:.1f}%<extra></extra>"
        ),
        showlegend=False,
    ),
    row=1, col=1
)

# Panel 2: Avg inventory by region
reg_s = region_inv.sort_values("avg_inventory", ascending=False)
fig_int.add_trace(
    go.Bar(
        x=reg_s["macroarea"],
        y=reg_s["avg_inventory"],
        marker_color=[REGION_COLORS[r] for r in reg_s["macroarea"]],
        text=[f"{v:.1f}" for v in reg_s["avg_inventory"]],
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>Avg Inventory: %{y:.1f}<extra></extra>"
        ),
        showlegend=False,
    ),
    row=1, col=2
)

# Panel 3: Tone & Click % by region
reg_tc2 = region_inv.sort_values("pct_with_tones", ascending=False)
fig_int.add_trace(
    go.Bar(
        name="% Tonal",
        x=reg_tc2["macroarea"],
        y=reg_tc2["pct_with_tones"],
        marker_color="#1A9641",
        offsetgroup=0,
        hovertemplate="<b>%{x}</b><br>Tonal: %{y:.1f}%<extra></extra>",
    ),
    row=2, col=1
)
fig_int.add_trace(
    go.Bar(
        name="% Clicks",
        x=reg_tc2["macroarea"],
        y=reg_tc2["pct_with_clicks"],
        marker_color="#8B0000",
        offsetgroup=1,
        hovertemplate="<b>%{x}</b><br>Clicks: %{y:.1f}%<extra></extra>",
    ),
    row=2, col=1
)

# Panel 4: Implicational universals
impl_10 = impl_df.head(10).copy()
impl_10["label"] = impl_10["given_A"] + " → " + impl_10["phoneme_B"]
fig_int.add_trace(
    go.Bar(
        x=impl_10["P(B|A)"],
        y=impl_10["label"],
        orientation="h",
        marker_color="#5E3C99",
        text=[f"{v:.3f}" for v in impl_10["P(B|A)"]],
        textposition="inside",
        insidetextanchor="end",
        hovertemplate=(
            "<b>%{y}</b><br>P(B|A): %{x:.3f}<extra></extra>"
        ),
        showlegend=False,
    ),
    row=2, col=2
)

fig_int.update_layout(
    title=dict(
        text="Phoneme Distribution Across Languages — Interactive Summary",
        font=dict(size=16),
        x=0.5,
    ),
    plot_bgcolor="white",
    paper_bgcolor="#FAFAFA",
    height=750,
    barmode="group",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
    ),
    font=dict(size=11),
)

# Update all axes
for row in [1, 2]:
    for col in [1, 2]:
        fig_int.update_xaxes(gridcolor="#eeeeee", row=row, col=col)
        fig_int.update_yaxes(gridcolor="#eeeeee", row=row, col=col)

path_int = os.path.join(FIGURES_DIR, "plot27_interactive_summary.html")
fig_int.write_html(path_int)
print(f"  Saved: {path_int}")

print()
print("[DONE] All Phase 6 visualizations saved to outputs/figures/")
