"""
PHASE 2 - SCRIPT 5
Purpose : Visualize phoneme frequency results
Plots   :
    1. Top 10 phonemes globally — horizontal bar chart (seaborn)
    2. Top 5 per class — grouped bar chart (matplotlib)
    3. Frequency distribution — histogram + KDE (seaborn)
    4. Top 10 phonemes — interactive bar chart (Plotly)
    5. Frequency distribution by class — Plotly box plot
Outputs : outputs/figures/
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.join(os.path.dirname(__file__), "..")
TABLES_DIR  = os.path.join(BASE_DIR, "outputs", "tables")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

freq   = pd.read_csv(os.path.join(TABLES_DIR, "phoneme_frequency.csv"))
top10  = pd.read_csv(os.path.join(TABLES_DIR, "top10_phonemes.csv"))

# ── Color palette per class ───────────────────────────────────────────────────
CLASS_COLORS = {
    "consonant" : "#2C7BB6",
    "vowel"     : "#D7191C",
    "tone"      : "#1A9641",
}

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — Top 10 phonemes globally (seaborn horizontal bar)
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Plot 1 — Top 10 phonemes globally")

fig, ax = plt.subplots(figsize=(10, 6))

colors = [CLASS_COLORS[c] for c in top10["segment_class"]]
bars = ax.barh(
    top10["Phoneme"][::-1],
    top10["frequency_pct"][::-1],
    color=colors[::-1],
    edgecolor="white",
    height=0.65,
)

# Annotate bars with exact %
for bar, pct in zip(bars, top10["frequency_pct"][::-1]):
    ax.text(
        bar.get_width() + 0.5,
        bar.get_y() + bar.get_height() / 2,
        f"{pct:.1f}%",
        va="center", ha="left", fontsize=10, color="#333333"
    )

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=CLASS_COLORS["consonant"], label="Consonant"),
    Patch(facecolor=CLASS_COLORS["vowel"],     label="Vowel"),
    Patch(facecolor=CLASS_COLORS["tone"],      label="Tone"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

ax.set_xlabel("Frequency (% of languages)", fontsize=11)
ax.set_title(
    "Top 10 Most Common Phonemes Globally\n"
    "(% of 2,176 languages containing the phoneme)",
    fontsize=13, fontweight="bold", pad=15
)
ax.set_xlim(0, 105)
ax.xaxis.set_major_formatter(mtick.PercentFormatter())
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", linestyle="--", alpha=0.4)

plt.tight_layout()
path1 = os.path.join(FIGURES_DIR, "plot1_top10_phonemes.png")
plt.savefig(path1, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {path1}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — Top 5 per SegmentClass (matplotlib grouped)
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Plot 2 — Top 5 per segment class")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
classes = ["consonant", "vowel", "tone"]
titles  = ["Top 5 Consonants", "Top 5 Vowels", "Top 5 Tones"]

for ax, cls, title in zip(axes, classes, titles):
    subset = freq[freq["segment_class"] == cls].head(5)
    color  = CLASS_COLORS[cls]

    bars = ax.bar(
        subset["Phoneme"],
        subset["frequency_pct"],
        color=color,
        edgecolor="white",
        width=0.55,
    )
    for bar, pct in zip(bars, subset["frequency_pct"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{pct:.1f}%",
            ha="center", va="bottom", fontsize=10
        )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency (%)" if ax == axes[0] else "")
    ax.set_ylim(0, max(subset["frequency_pct"]) * 1.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.grid(axis="y", linestyle="--", alpha=0.4)

fig.suptitle(
    "Top 5 Phonemes by Segment Class",
    fontsize=14, fontweight="bold", y=1.02
)
plt.tight_layout()
path2 = os.path.join(FIGURES_DIR, "plot2_top5_per_class.png")
plt.savefig(path2, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {path2}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 — Frequency distribution histogram + KDE (seaborn)
#
# WHAT  : Shows how frequency % is distributed across all 2,721 phonemes
# WHY   : Reveals the Zipfian / long-tail nature — a few phonemes are
#         universal while the vast majority appear in only 1–2 languages
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Plot 3 — Frequency distribution histogram")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: full distribution (log scale x-axis to reveal long tail)
ax = axes[0]
ax.hist(
    freq["frequency_pct"],
    bins=80, color="#2C7BB6", edgecolor="white", alpha=0.85
)
ax.set_xlabel("Frequency % (log scale)", fontsize=11)
ax.set_ylabel("Number of phonemes", fontsize=11)
ax.set_title(
    "Phoneme Frequency Distribution\n(All 2,721 phonemes — log x-axis)",
    fontsize=12, fontweight="bold"
)
ax.set_xscale("log")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.4)

# Right: zoomed to phonemes above 5% frequency (the meaningful range)
ax = axes[1]
freq_above5 = freq[freq["frequency_pct"] >= 5]
colors_above5 = [CLASS_COLORS[c] for c in freq_above5["segment_class"]]
ax.scatter(
    freq_above5["rank"],
    freq_above5["frequency_pct"],
    c=colors_above5,
    s=60, alpha=0.8, edgecolors="white", linewidths=0.5
)
# Label top 10
for _, row in freq_above5.head(10).iterrows():
    ax.annotate(
        row["Phoneme"],
        (row["rank"], row["frequency_pct"]),
        textcoords="offset points", xytext=(5, 2),
        fontsize=9, color="#333333"
    )

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=CLASS_COLORS["consonant"], label="Consonant"),
    Patch(facecolor=CLASS_COLORS["vowel"],     label="Vowel"),
]
ax.legend(handles=legend_elements, fontsize=9)
ax.set_xlabel("Phoneme rank", fontsize=11)
ax.set_ylabel("Frequency %", fontsize=11)
ax.set_title(
    "Phonemes Appearing in ≥5% of Languages\n(Rank vs Frequency)",
    fontsize=12, fontweight="bold"
)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(linestyle="--", alpha=0.4)

plt.tight_layout()
path3 = os.path.join(FIGURES_DIR, "plot3_frequency_distribution.png")
plt.savefig(path3, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {path3}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4 — Interactive Top 10 bar chart (Plotly)
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Plot 4 — Interactive Top 10 (Plotly)")

top10_plot = top10.copy()
top10_plot["color"] = top10_plot["segment_class"].map(CLASS_COLORS)

fig4 = px.bar(
    top10_plot,
    x="Phoneme",
    y="frequency_pct",
    color="segment_class",
    color_discrete_map=CLASS_COLORS,
    text="frequency_pct",
    hover_data={"language_count": True, "frequency_pct": True,
                "segment_class": True},
    labels={
        "frequency_pct"  : "Frequency (%)",
        "Phoneme"        : "Phoneme (IPA)",
        "segment_class"  : "Class",
        "language_count" : "Language Count"
    },
    title="Top 10 Most Common Phonemes Globally",
)
fig4.update_traces(
    texttemplate="%{text:.1f}%",
    textposition="outside"
)
fig4.update_layout(
    plot_bgcolor="white",
    yaxis=dict(
        title="Frequency (% of 2,176 languages)",
        gridcolor="#eeeeee",
        range=[0, 105]
    ),
    xaxis=dict(title="Phoneme (IPA symbol)"),
    legend_title="Segment Class",
    font=dict(size=13),
    title_font_size=16,
    height=500,
)
path4 = os.path.join(FIGURES_DIR, "plot4_top10_interactive.html")
fig4.write_html(path4)
print(f"  Saved: {path4}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 5 — Frequency distribution by class — Plotly box plot
#
# WHAT  : Box plot showing spread of frequency values per class
# WHY   : Consonants have many more phonemes but most are rare.
#         Vowels have fewer but slightly higher median frequency.
#         Tones are almost all rare (low frequency globally).
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Plot 5 — Frequency by class box plot (Plotly)")

fig5 = px.box(
    freq,
    x="segment_class",
    y="frequency_pct",
    color="segment_class",
    color_discrete_map=CLASS_COLORS,
    points="outliers",
    hover_data=["Phoneme", "language_count"],
    labels={
        "segment_class"  : "Segment Class",
        "frequency_pct"  : "Frequency (%)",
    },
    title="Phoneme Frequency Distribution by Segment Class",
)
fig5.update_layout(
    plot_bgcolor="white",
    yaxis=dict(gridcolor="#eeeeee"),
    showlegend=False,
    font=dict(size=13),
    title_font_size=16,
    height=500,
)
path5 = os.path.join(FIGURES_DIR, "plot5_frequency_by_class_boxplot.html")
fig5.write_html(path5)
print(f"  Saved: {path5}")

print()
print("[DONE] All Phase 2 visualizations saved to outputs/figures/")
