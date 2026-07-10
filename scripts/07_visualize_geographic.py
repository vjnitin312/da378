"""
PHASE 3 - SCRIPT 7
Purpose : Visualize geographic and language family grouping results
Plots   :
    1.  Avg inventory size by macroarea — horizontal bar (seaborn)
    2.  Phoneme class % by region — stacked bar (matplotlib)
    3.  Click & tone % by region — grouped bar (matplotlib)
    4.  Top 15 families by avg inventory — bar chart (seaborn)
    5.  Inventory size distribution by region — box plot (seaborn)
    6.  Interactive: avg inventory by region (Plotly bar)
    7.  Interactive: family inventory heatmap (Plotly)
    8.  Interactive: world map — inventory size per language (Plotly choropleth)
    9.  Interactive: tone concentration map (Folium)
    10. Interactive: click language map (Folium)
Outputs : outputs/figures/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import MarkerCluster
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.join(os.path.dirname(__file__), "..")
TABLES_DIR  = os.path.join(BASE_DIR, "outputs", "tables")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

region_inv    = pd.read_csv(os.path.join(TABLES_DIR, "region_inventory_stats.csv"))
region_class  = pd.read_csv(os.path.join(TABLES_DIR, "region_class_distribution.csv"))
family_inv    = pd.read_csv(os.path.join(TABLES_DIR, "family_inventory_stats.csv"))
lang_inv      = pd.read_csv(os.path.join(BASE_DIR, "data", "processed",
                                          "language_inventory.csv"))

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
# PLOT 1 — Avg inventory size by macroarea (seaborn horizontal bar)
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Plot 1 — Avg inventory by macroarea")

region_sorted = region_inv.sort_values("avg_inventory", ascending=True)
colors = [REGION_COLORS[r] for r in region_sorted["macroarea"]]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(
    region_sorted["macroarea"],
    region_sorted["avg_inventory"],
    color=colors, edgecolor="white", height=0.6
)
for bar, val in zip(bars, region_sorted["avg_inventory"]):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}", va="center", ha="left", fontsize=10)

ax.set_xlabel("Average Phoneme Inventory Size", fontsize=11)
ax.set_title("Average Phoneme Inventory Size by Macroarea",
             fontsize=13, fontweight="bold", pad=12)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", linestyle="--", alpha=0.4)
ax.set_xlim(0, 52)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "plot6_region_inventory.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: plot6_region_inventory.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — Phoneme class % by region (stacked bar)
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Plot 2 — Phoneme class % by region")

pivot = region_class.pivot(
    index="macroarea", columns="SegmentClass", values="pct"
).fillna(0)

# Reorder regions by consonant %
pivot = pivot.sort_values("consonant", ascending=False)
regions = pivot.index.tolist()
x = range(len(regions))
width = 0.55

fig, ax = plt.subplots(figsize=(11, 6))
bottom = np.zeros(len(regions))

for cls, color in CLASS_COLORS.items():
    if cls in pivot.columns:
        vals = pivot[cls].values
        bars = ax.bar(x, vals, width, bottom=bottom,
                      label=cls.capitalize(), color=color,
                      edgecolor="white")
        for i, (bar, val) in enumerate(zip(bars, vals)):
            if val > 1.5:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bottom[i] + val / 2,
                        f"{val:.0f}%", ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold")
        bottom += vals

ax.set_xticks(list(x))
ax.set_xticklabels(regions, rotation=15, ha="right", fontsize=10)
ax.set_ylabel("% of phoneme tokens", fontsize=11)
ax.set_title("Phoneme Class Distribution by Macroarea",
             fontsize=13, fontweight="bold", pad=12)
ax.legend(loc="upper right", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
ax.set_ylim(0, 108)
ax.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "plot7_class_by_region_stacked.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: plot7_class_by_region_stacked.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 — Click & tone % by region (grouped bar)
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Plot 3 — Click & tone % by region")

region_sorted2 = region_inv.sort_values("pct_with_tones", ascending=False)
x = np.arange(len(region_sorted2))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 5))
bars1 = ax.bar(x - width/2, region_sorted2["pct_with_tones"],
               width, label="% with Tone segments",
               color="#1A9641", edgecolor="white")
bars2 = ax.bar(x + width/2, region_sorted2["pct_with_clicks"],
               width, label="% with Click phonemes",
               color="#8B0000", edgecolor="white")

for bar in bars1:
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                f"{h:.0f}%", ha="center", va="bottom", fontsize=9)
for bar in bars2:
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(region_sorted2["macroarea"], rotation=15,
                   ha="right", fontsize=10)
ax.set_ylabel("% of languages", fontsize=11)
ax.set_title("Tone and Click Phoneme Concentration by Macroarea",
             fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "plot8_click_tone_by_region.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: plot8_click_tone_by_region.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4 — Top 15 families by avg inventory (seaborn)
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Plot 4 — Top 15 families by inventory")

fam5 = family_inv[family_inv["n_languages"] >= 5].copy()
top15 = fam5.nlargest(15, "avg_inventory").sort_values(
    "avg_inventory", ascending=True)

fig, ax = plt.subplots(figsize=(11, 7))
bars = ax.barh(top15["family_name"], top15["avg_inventory"],
               color="#5E3C99", edgecolor="white", height=0.65)
for bar, val in zip(bars, top15["avg_inventory"]):
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}", va="center", ha="left", fontsize=9)

ax.set_xlabel("Average Inventory Size", fontsize=11)
ax.set_title("Top 15 Language Families by Average Phoneme Inventory\n"
             "(families with ≥5 languages)",
             fontsize=12, fontweight="bold", pad=12)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", linestyle="--", alpha=0.4)
ax.set_xlim(0, 75)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "plot9_top15_families.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: plot9_top15_families.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 5 — Inventory size distribution by region (seaborn boxplot)
#
# WHAT : Shows the spread of inventory sizes within each region
# WHY  : Averages hide variance. Africa has both click languages with 150+
#        phonemes and small-inventory languages — the box plot reveals this.
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Plot 5 — Inventory distribution by region (box)")

lang_clean = lang_inv[lang_inv["macroarea"].notna()].copy()
order = (lang_clean.groupby("macroarea")["total_phonemes"]
         .median().sort_values(ascending=False).index.tolist())

fig, ax = plt.subplots(figsize=(12, 6))
palette = [REGION_COLORS[r] for r in order]
sns.boxplot(
    data=lang_clean, x="macroarea", y="total_phonemes",
    order=order, palette=palette,
    width=0.55, fliersize=3, linewidth=1.2, ax=ax
)
ax.set_xlabel("Macroarea", fontsize=11)
ax.set_ylabel("Total Phonemes in Inventory", fontsize=11)
ax.set_title("Distribution of Phoneme Inventory Sizes by Macroarea",
             fontsize=13, fontweight="bold", pad=12)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "plot10_inventory_boxplot.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: plot10_inventory_boxplot.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 6 — Interactive: avg inventory by region (Plotly)
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Plot 6 — Interactive region inventory (Plotly)")

fig6 = px.bar(
    region_inv.sort_values("avg_inventory", ascending=False),
    x="macroarea", y="avg_inventory",
    color="macroarea",
    color_discrete_map=REGION_COLORS,
    text="avg_inventory",
    hover_data={"n_languages": True, "avg_consonants": True,
                "avg_vowels": True, "pct_with_tones": True,
                "pct_with_clicks": True},
    labels={"avg_inventory": "Avg Inventory Size",
            "macroarea": "Macroarea",
            "n_languages": "No. of Languages"},
    title="Average Phoneme Inventory Size by Macroarea"
)
fig6.update_traces(texttemplate="%{text:.1f}", textposition="outside")
fig6.update_layout(
    plot_bgcolor="white",
    yaxis=dict(gridcolor="#eeeeee", title="Avg Inventory Size"),
    showlegend=False, font=dict(size=13),
    title_font_size=16, height=480
)
fig6.write_html(os.path.join(FIGURES_DIR, "plot11_region_inventory_interactive.html"))
print("  Saved: plot11_region_inventory_interactive.html")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 7 — Interactive: family inventory heatmap (Plotly)
# Top 20 families × avg consonants, vowels, tones
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Plot 7 — Family heatmap (Plotly)")

top20_fam = fam5.nlargest(20, "avg_inventory")
heatmap_data = top20_fam.set_index("family_name")[
    ["avg_consonants", "avg_vowels", "pct_with_tones"]
].rename(columns={
    "avg_consonants"  : "Avg Consonants",
    "avg_vowels"      : "Avg Vowels",
    "pct_with_tones"  : "% with Tones",
})

fig7 = go.Figure(data=go.Heatmap(
    z=heatmap_data.values,
    x=heatmap_data.columns.tolist(),
    y=heatmap_data.index.tolist(),
    colorscale="RdBu",
    text=np.round(heatmap_data.values, 1),
    texttemplate="%{text}",
    textfont={"size": 10},
    hoverongaps=False,
))
fig7.update_layout(
    title="Top 20 Language Families — Phoneme Profile Heatmap",
    title_font_size=15,
    xaxis_title="Feature",
    yaxis_title="Language Family",
    height=600,
    font=dict(size=11),
)
fig7.write_html(os.path.join(FIGURES_DIR, "plot12_family_heatmap.html"))
print("  Saved: plot12_family_heatmap.html")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 8 — Interactive world map: inventory size per language (Plotly scatter)
#
# WHAT : Each language plotted at its lat/lon, colored by inventory size
# WHY  : Reveals geographic clustering — large inventories in Africa/Eurasia,
#        small inventories in Papunesia/Australia
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Plot 8 — World map: inventory size (Plotly)")

map_df = lang_clean[
    lang_clean["latitude"].notna() & lang_clean["longitude"].notna()
].copy()

fig8 = px.scatter_geo(
    map_df,
    lat="latitude",
    lon="longitude",
    color="total_phonemes",
    hover_name="LanguageName",
    hover_data={
        "macroarea"     : True,
        "family_name"   : True,
        "total_phonemes": True,
        "n_consonants"  : True,
        "n_vowels"      : True,
        "latitude"      : False,
        "longitude"     : False,
    },
    color_continuous_scale="Viridis",
    size="total_phonemes",
    size_max=14,
    projection="natural earth",
    labels={
        "total_phonemes": "Inventory Size",
        "macroarea"     : "Region",
        "family_name"   : "Family",
    },
    title="World Map — Phoneme Inventory Size per Language",
)
fig8.update_layout(
    title_font_size=16,
    coloraxis_colorbar=dict(title="Inventory Size"),
    geo=dict(showland=True, landcolor="#f0f0f0",
             showocean=True, oceancolor="#d0e8f0",
             showcoastlines=True, coastlinecolor="#aaaaaa"),
    height=550,
    margin=dict(l=0, r=0, t=50, b=0),
)
fig8.write_html(os.path.join(FIGURES_DIR, "plot13_world_map_inventory.html"))
print("  Saved: plot13_world_map_inventory.html")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 9 — Folium map: tone languages highlighted
#
# WHAT : Interactive map marking all tonal languages in red, non-tonal in blue
# WHY  : Lets the viewer see the geographic clustering of tone —
#        heavy in sub-Saharan Africa and East/Southeast Asia
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Plot 9 — Folium map: tone languages")

tone_map = folium.Map(location=[15, 30], zoom_start=2,
                      tiles="CartoDB positron")

for _, row in map_df.iterrows():
    color  = "#D7191C" if row["has_tone_segment"] else "#2C7BB6"
    radius = 4
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=radius,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        popup=folium.Popup(
            f"<b>{row['LanguageName']}</b><br>"
            f"Region: {row['macroarea']}<br>"
            f"Family: {row['family_name']}<br>"
            f"Tonal: {'Yes' if row['has_tone_segment'] else 'No'}<br>"
            f"Inventory: {int(row['total_phonemes'])} phonemes",
            max_width=200
        ),
        tooltip=row["LanguageName"],
    ).add_to(tone_map)

# Legend
legend_html = """
<div style="position:fixed; bottom:30px; left:30px; z-index:1000;
     background:white; padding:10px 15px; border-radius:8px;
     border:1px solid #ccc; font-size:13px; line-height:1.8;">
  <b>Tone Languages</b><br>
  <span style="color:#D7191C;">&#9679;</span> Tonal language<br>
  <span style="color:#2C7BB6;">&#9679;</span> Non-tonal language
</div>
"""
tone_map.get_root().html.add_child(folium.Element(legend_html))
tone_map.save(os.path.join(FIGURES_DIR, "plot14_tone_map_folium.html"))
print("  Saved: plot14_tone_map_folium.html")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 10 — Folium map: click languages
#
# WHAT : Only languages with click phonemes marked (with language name popup)
# WHY  : Dramatically shows geographic restriction of clicks —
#        almost entirely southern/eastern Africa
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Plot 10 — Folium map: click languages")

click_map = folium.Map(location=[0, 25], zoom_start=3,
                       tiles="CartoDB positron")

click_langs = map_df[map_df["has_click"] == True]
non_click   = map_df[map_df["has_click"] == False]

# Non-click: small grey dots
for _, row in non_click.iterrows():
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=2,
        color="#cccccc",
        fill=True, fill_color="#cccccc", fill_opacity=0.4,
        tooltip=row["LanguageName"],
    ).add_to(click_map)

# Click languages: prominent red markers
for _, row in click_langs.iterrows():
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=9,
        color="#8B0000",
        fill=True, fill_color="#D7191C", fill_opacity=0.9,
        popup=folium.Popup(
            f"<b>{row['LanguageName']}</b><br>"
            f"Family: {row['family_name']}<br>"
            f"Inventory: {int(row['total_phonemes'])} phonemes<br>"
            f"Clicks: Yes",
            max_width=200
        ),
        tooltip=f"[CLICK] {row['LanguageName']}",
    ).add_to(click_map)

legend_html = """
<div style="position:fixed; bottom:30px; left:30px; z-index:1000;
     background:white; padding:10px 15px; border-radius:8px;
     border:1px solid #ccc; font-size:13px; line-height:1.8;">
  <b>Click Languages</b><br>
  <span style="color:#D7191C; font-size:16px;">&#9679;</span>
    Language with clicks<br>
  <span style="color:#cccccc; font-size:12px;">&#9679;</span>
    No click phonemes
</div>
"""
click_map.get_root().html.add_child(folium.Element(legend_html))
click_map.save(os.path.join(FIGURES_DIR, "plot15_click_map_folium.html"))
print("  Saved: plot15_click_map_folium.html")

print()
print("[DONE] All Phase 3 visualizations saved to outputs/figures/")
