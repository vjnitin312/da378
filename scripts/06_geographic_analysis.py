"""
PHASE 3 - SCRIPT 6
Purpose : Group phoneme data by macroarea and language family
          - Phoneme class distribution per region
          - Inventory size statistics per region and family
          - Click, tone, nasal concentration by region
          - Top/bottom language families by inventory size
Outputs :
    outputs/tables/region_class_distribution.csv
    outputs/tables/region_inventory_stats.csv
    outputs/tables/family_inventory_stats.csv
    outputs/tables/region_click_tone_nasal.csv
    outputs/tables/phase3_report.txt
"""

import pandas as pd
import numpy as np
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.join(os.path.dirname(__file__), "..")
DEDUP_PATH = os.path.join(BASE_DIR, "data", "processed", "phoible_dedup.csv")
TABLES_DIR = os.path.join(BASE_DIR, "outputs", "tables")
os.makedirs(TABLES_DIR, exist_ok=True)

lines = []
def log(text=""):
    print(text)
    lines.append(str(text))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load data
# ─────────────────────────────────────────────────────────────────────────────
log("=" * 60)
log("STEP 1 — Loading phoible_dedup.csv")
log("=" * 60)

df = pd.read_csv(DEDUP_PATH, low_memory=False)
df["is_click"] = df["is_click"].astype(str).str.strip().str.lower() == "true"
df["is_tone"]  = df["is_tone"].astype(str).str.strip().str.lower()  == "true"
df["is_nasal"] = df["is_nasal"].astype(str).str.strip().str.lower() == "true"

log(f"  Rows    : {len(df):,}")
log(f"  Languages : {df['Glottocode'].nunique():,}")
log(f"  Macroareas: {df['macroarea'].nunique()}")
log(f"  Families  : {df['family_name'].nunique()}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Inventory size per language
#
# WHAT : Count how many phonemes each language has
# WHY  : Inventory size is a key typological variable.
#        It is the basis for comparing richness across regions and families.
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 60)
log("STEP 2 — Computing inventory size per language")
log("=" * 60)

inv = (
    df.groupby(["Glottocode", "LanguageName", "macroarea", "family_name",
                "latitude", "longitude"])
    .agg(
        total_phonemes     = ("Phoneme",       "nunique"),
        n_consonants       = ("SegmentClass",  lambda x: (x == "consonant").sum()),
        n_vowels           = ("SegmentClass",  lambda x: (x == "vowel").sum()),
        n_tones            = ("SegmentClass",  lambda x: (x == "tone").sum()),
        has_click          = ("is_click",      "any"),
        has_tone_segment   = ("is_tone",       "any"),
        n_nasal_phonemes   = ("is_nasal",      "sum"),
    )
    .reset_index()
)

inv["consonant_ratio"] = (inv["n_consonants"] / inv["total_phonemes"]).round(4)
inv["vowel_ratio"]     = (inv["n_vowels"]     / inv["total_phonemes"]).round(4)

log(f"  Languages with inventory data: {len(inv):,}")
log(f"  Global avg inventory size    : {inv['total_phonemes'].mean():.1f}")
log(f"  Global min inventory size    : {inv['total_phonemes'].min()} "
    f"({inv.loc[inv['total_phonemes'].idxmin(), 'LanguageName']})")
log(f"  Global max inventory size    : {inv['total_phonemes'].max()} "
    f"({inv.loc[inv['total_phonemes'].idxmax(), 'LanguageName']})")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Inventory statistics by macroarea
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 60)
log("STEP 3 — Inventory statistics by macroarea")
log("=" * 60)

region_inv = (
    inv.groupby("macroarea")
    .agg(
        n_languages        = ("Glottocode",      "count"),
        avg_inventory      = ("total_phonemes",  "mean"),
        median_inventory   = ("total_phonemes",  "median"),
        std_inventory      = ("total_phonemes",  "std"),
        avg_consonants     = ("n_consonants",    "mean"),
        avg_vowels         = ("n_vowels",        "mean"),
        avg_tones          = ("n_tones",         "mean"),
        avg_consonant_ratio= ("consonant_ratio", "mean"),
        avg_vowel_ratio    = ("vowel_ratio",     "mean"),
        pct_with_clicks    = ("has_click",       "mean"),
        pct_with_tones     = ("has_tone_segment","mean"),
    )
    .reset_index()
    .sort_values("avg_inventory", ascending=False)
)

for col in ["avg_inventory", "median_inventory", "std_inventory",
            "avg_consonants", "avg_vowels", "avg_tones",
            "avg_consonant_ratio", "avg_vowel_ratio"]:
    region_inv[col] = region_inv[col].round(2)

region_inv["pct_with_clicks"] = (region_inv["pct_with_clicks"] * 100).round(1)
region_inv["pct_with_tones"]  = (region_inv["pct_with_tones"]  * 100).round(1)

log()
log(region_inv[[
    "macroarea", "n_languages", "avg_inventory",
    "avg_consonants", "avg_vowels", "avg_tones",
    "pct_with_clicks", "pct_with_tones"
]].to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Phoneme class distribution per region
#
# WHAT : For each region, what % of phoneme tokens are consonants vs vowels
# WHY  : Some regions (e.g. Australia) are known for very high consonant ratios
#        while others (e.g. Polynesian/Papunesia) have many vowels
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 60)
log("STEP 4 — Phoneme class distribution per region")
log("=" * 60)

region_class = (
    df.groupby(["macroarea", "SegmentClass"])
    .agg(phoneme_count=("Phoneme", "count"))
    .reset_index()
)
region_total = (
    df.groupby("macroarea")
    .agg(total=("Phoneme", "count"))
    .reset_index()
)
region_class = region_class.merge(region_total, on="macroarea")
region_class["pct"] = (
    region_class["phoneme_count"] / region_class["total"] * 100
).round(2)

log()
log(region_class.pivot(
    index="macroarea", columns="SegmentClass", values="pct"
).round(2).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Click phoneme concentration by region
#
# WHAT : Count languages with at least one click phoneme per region
# WHY  : Clicks are geographically restricted — almost exclusively in
#        southern African Khoisan and Bantu languages.
#        This is a classic case of areal feature concentration.
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 60)
log("STEP 5 — Click concentration by region")
log("=" * 60)

click_region = (
    inv.groupby("macroarea")
    .agg(
        total_languages  = ("Glottocode", "count"),
        languages_w_click= ("has_click",  "sum"),
    )
    .reset_index()
)
click_region["click_pct"] = (
    click_region["languages_w_click"] / click_region["total_languages"] * 100
).round(2)
click_region = click_region.sort_values("click_pct", ascending=False)

log()
log(click_region.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Tone segment concentration by region
#
# WHAT : Count languages with tone phoneme segments per region
# WHY  : Tonal languages are concentrated in Africa (Niger-Congo, Nilo-Saharan)
#        and East/Southeast Asia (Sino-Tibetan, Tai-Kadai, Hmong-Mien).
#        This is a well-documented areal typological pattern.
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 60)
log("STEP 6 — Tone segment concentration by region")
log("=" * 60)

tone_region = (
    inv.groupby("macroarea")
    .agg(
        total_languages = ("Glottocode",       "count"),
        languages_w_tone= ("has_tone_segment", "sum"),
    )
    .reset_index()
)
tone_region["tone_pct"] = (
    tone_region["languages_w_tone"] / tone_region["total_languages"] * 100
).round(2)
tone_region = tone_region.sort_values("tone_pct", ascending=False)

log()
log(tone_region.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Language family inventory statistics
# Top 15 families by avg inventory, bottom 10
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 60)
log("STEP 7 — Language family inventory statistics")
log("=" * 60)

family_inv = (
    inv[inv["family_name"].notna()]
    .groupby("family_name")
    .agg(
        n_languages      = ("Glottocode",     "count"),
        avg_inventory    = ("total_phonemes", "mean"),
        median_inventory = ("total_phonemes", "median"),
        avg_consonants   = ("n_consonants",   "mean"),
        avg_vowels       = ("n_vowels",       "mean"),
        pct_with_clicks  = ("has_click",      "mean"),
        pct_with_tones   = ("has_tone_segment","mean"),
    )
    .reset_index()
)
family_inv["avg_inventory"]    = family_inv["avg_inventory"].round(1)
family_inv["median_inventory"] = family_inv["median_inventory"].round(1)
family_inv["avg_consonants"]   = family_inv["avg_consonants"].round(1)
family_inv["avg_vowels"]       = family_inv["avg_vowels"].round(1)
family_inv["pct_with_clicks"]  = (family_inv["pct_with_clicks"] * 100).round(1)
family_inv["pct_with_tones"]   = (family_inv["pct_with_tones"]  * 100).round(1)

# Filter: families with at least 5 languages for reliable statistics
family_inv_5 = family_inv[family_inv["n_languages"] >= 5].copy()

log(f"\n  Families with ≥5 languages: {len(family_inv_5)}")

log()
log("  TOP 15 FAMILIES BY AVG INVENTORY SIZE:")
top15_fam = family_inv_5.sort_values("avg_inventory", ascending=False).head(15)
log(top15_fam[[
    "family_name", "n_languages", "avg_inventory",
    "avg_consonants", "avg_vowels"
]].to_string(index=False))

log()
log("  BOTTOM 10 FAMILIES BY AVG INVENTORY SIZE:")
bot10_fam = family_inv_5.sort_values("avg_inventory").head(10)
log(bot10_fam[[
    "family_name", "n_languages", "avg_inventory",
    "avg_consonants", "avg_vowels"
]].to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Top and bottom individual languages
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 60)
log("STEP 8 — Extreme languages (largest and smallest inventories)")
log("=" * 60)

log()
log("  TOP 10 LANGUAGES — LARGEST PHONEME INVENTORY:")
top10_lang = inv.nlargest(10, "total_phonemes")[[
    "LanguageName", "macroarea", "family_name", "total_phonemes",
    "n_consonants", "n_vowels", "has_click"
]]
log(top10_lang.to_string(index=False))

log()
log("  BOTTOM 10 LANGUAGES — SMALLEST PHONEME INVENTORY:")
bot10_lang = inv.nsmallest(10, "total_phonemes")[[
    "LanguageName", "macroarea", "family_name", "total_phonemes",
    "n_consonants", "n_vowels"
]]
log(bot10_lang.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — Save all tables
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 60)
log("STEP 9 — Saving tables")
log("=" * 60)

region_inv.to_csv(
    os.path.join(TABLES_DIR, "region_inventory_stats.csv"), index=False)
log("  Saved: region_inventory_stats.csv")

region_class.to_csv(
    os.path.join(TABLES_DIR, "region_class_distribution.csv"), index=False)
log("  Saved: region_class_distribution.csv")

family_inv.to_csv(
    os.path.join(TABLES_DIR, "family_inventory_stats.csv"), index=False)
log("  Saved: family_inventory_stats.csv")

click_region.to_csv(
    os.path.join(TABLES_DIR, "region_click_tone_nasal.csv"), index=False)
log("  Saved: region_click_tone_nasal.csv")

inv.to_csv(
    os.path.join(BASE_DIR, "data", "processed", "language_inventory.csv"),
    index=False)
log("  Saved: data/processed/language_inventory.csv")

with open(os.path.join(TABLES_DIR, "phase3_report.txt"), "w",
          encoding="utf-8") as f:
    f.write("\n".join(lines))
log("  Saved: phase3_report.txt")

log()
log("[DONE] Phase 3 — Geographic & Family Analysis complete.")
