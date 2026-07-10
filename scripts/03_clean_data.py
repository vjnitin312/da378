"""
PHASE 1 - SCRIPT 3
Purpose : Clean PHOIBLE raw data and merge with Glottolog language metadata
          (macroarea, lat/lon, family name) from CLDF PHOIBLE languages.csv
Outputs :
    data/processed/phoible_clean.csv   — full cleaned dataset
    data/processed/phoible_dedup.csv   — one inventory per language (deduplicated)
    outputs/tables/cleaning_report.txt — cleaning summary
"""

import pandas as pd
import numpy as np
import requests
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.join(os.path.dirname(__file__), "..")
RAW_PATH       = os.path.join(BASE_DIR, "data", "raw", "phoible.csv")
PROCESSED_DIR  = os.path.join(BASE_DIR, "data", "processed")
TABLES_DIR     = os.path.join(BASE_DIR, "outputs", "tables")
CLEAN_PATH     = os.path.join(PROCESSED_DIR, "phoible_clean.csv")
DEDUP_PATH     = os.path.join(PROCESSED_DIR, "phoible_dedup.csv")
REPORT_PATH    = os.path.join(TABLES_DIR, "cleaning_report.txt")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

lines = []
def log(text=""):
    print(text)
    lines.append(str(text))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load raw PHOIBLE
# ─────────────────────────────────────────────────────────────────────────────
log("=" * 60)
log("STEP 1 — Loading raw PHOIBLE")
log("=" * 60)

df = pd.read_csv(RAW_PATH, low_memory=False)
log(f"  Raw shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Drop columns we do not need for this analysis
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 60)
log("STEP 2 — Dropping unused columns")
log("=" * 60)

# We keep:
#   Identity : InventoryID, Glottocode, ISO6393, LanguageName
#   Phoneme  : Phoneme, SegmentClass, Marginal
#   Features : tone, click, nasal, lateral, labial, coronal, dorsal,
#              sonorant, continuant, approximant, trill, tap,
#              periodicGlottalSource (voiced), spreadGlottis, constrictedGlottis
#   Source   : Source

KEEP_COLS = [
    "InventoryID", "Glottocode", "ISO6393", "LanguageName",
    "Phoneme", "SegmentClass", "Marginal", "Source",
    "tone", "click",
    "nasal", "lateral", "labial", "coronal", "dorsal",
    "sonorant", "continuant", "approximant", "trill", "tap",
    "periodicGlottalSource", "spreadGlottis", "constrictedGlottis",
]

dropped = [c for c in df.columns if c not in KEEP_COLS]
log(f"  Dropping {len(dropped)} columns: {dropped}")
df = df[KEEP_COLS].copy()
log(f"  Shape after drop: {df.shape[0]:,} × {df.shape[1]}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Handle Marginal phonemes
# Marginal phonemes are sounds that appear in a language only in loanwords
# or very rarely. Including them inflates frequency counts unfairly.
# We flag them but exclude them from the main cleaned set.
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 60)
log("STEP 3 — Handling Marginal phonemes")
log("=" * 60)

marginal_mask = df["Marginal"].astype(str).str.strip().str.upper() == "TRUE"
log(f"  Marginal phoneme rows: {marginal_mask.sum():,}")
log(f"  Non-marginal rows    : {(~marginal_mask).sum():,}")

# Exclude marginal from main dataset
df = df[~marginal_mask].copy()
df.drop(columns=["Marginal"], inplace=True)
log(f"  Shape after removing marginals: {df.shape[0]:,} × {df.shape[1]}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Strip whitespace from string columns
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 60)
log("STEP 4 — Stripping whitespace from string columns")
log("=" * 60)

str_cols = df.select_dtypes(include="object").columns.tolist()
for col in str_cols:
    df[col] = df[col].astype(str).str.strip()
log(f"  Cleaned {len(str_cols)} string columns")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Fix NaN represented as string 'nan'
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 60)
log("STEP 5 — Replacing string 'nan' with actual NaN")
log("=" * 60)

df.replace("nan", np.nan, inplace=True)
log(f"  Done. Remaining nulls per column:")
null_summary = df.isnull().sum()
for col, cnt in null_summary[null_summary > 0].items():
    log(f"    {col:<30} {cnt:,}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Download and merge Glottolog metadata
# (macroarea, latitude, longitude, family_name)
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 60)
log("STEP 6 — Downloading Glottolog metadata (CLDF PHOIBLE)")
log("=" * 60)

LANG_URL = (
    "https://raw.githubusercontent.com/cldf-datasets/phoible"
    "/master/cldf/languages.csv"
)
LANG_CACHE = os.path.join(BASE_DIR, "data", "raw", "languages.csv")

if not os.path.exists(LANG_CACHE):
    r = requests.get(LANG_URL, timeout=60)
    with open(LANG_CACHE, "wb") as f:
        f.write(r.content)
    log(f"  Downloaded languages.csv → {LANG_CACHE}")
else:
    log(f"  languages.csv already cached at {LANG_CACHE}")

lang_df = pd.read_csv(LANG_CACHE, low_memory=False)
log(f"  languages.csv shape: {lang_df.shape}")
log(f"  Columns: {lang_df.columns.tolist()}")

# Keep only what we need
lang_df = lang_df[[
    "Glottocode", "Macroarea", "Latitude", "Longitude", "Family_Name"
]].copy()
lang_df.rename(columns={
    "Macroarea"   : "macroarea",
    "Latitude"    : "latitude",
    "Longitude"   : "longitude",
    "Family_Name" : "family_name"
}, inplace=True)

# Drop duplicate glottocodes (keep first)
lang_df.drop_duplicates(subset="Glottocode", inplace=True)
log(f"  Unique Glottocodes in metadata: {lang_df['Glottocode'].nunique():,}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Merge metadata into main DataFrame
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 60)
log("STEP 7 — Merging metadata into PHOIBLE")
log("=" * 60)

before = len(df)
df = df.merge(lang_df, on="Glottocode", how="left")
log(f"  Rows before merge : {before:,}")
log(f"  Rows after merge  : {len(df):,}")
log(f"  macroarea nulls   : {df['macroarea'].isnull().sum():,}")
log(f"  family_name nulls : {df['family_name'].isnull().sum():,}")
log(f"  Macroarea values  : {df['macroarea'].value_counts().to_dict()}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Standardize SegmentClass
# Confirm only 3 valid classes: consonant, vowel, tone
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 60)
log("STEP 8 — Standardizing SegmentClass")
log("=" * 60)

log(f"  SegmentClass values: {df['SegmentClass'].value_counts().to_dict()}")
valid_classes = {"consonant", "vowel", "tone"}
invalid_mask = ~df["SegmentClass"].isin(valid_classes)
log(f"  Rows with invalid SegmentClass: {invalid_mask.sum()}")
df = df[~invalid_mask].copy()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — Add derived columns useful for later analysis
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 60)
log("STEP 9 — Adding derived columns")
log("=" * 60)

# is_click: phoneme is a click consonant
df["is_click"] = df["click"].astype(str).str.strip() == "+"
log(f"  Click phoneme rows  : {df['is_click'].sum():,}")

# is_tone: tone segment
df["is_tone"] = df["SegmentClass"] == "tone"
log(f"  Tone segment rows   : {df['is_tone'].sum():,}")

# is_nasal
df["is_nasal"] = df["nasal"].astype(str).str.strip() == "+"
log(f"  Nasal phoneme rows  : {df['is_nasal'].sum():,}")

# broad_class: consonant / vowel / tone (same as SegmentClass, kept explicit)
df["broad_class"] = df["SegmentClass"]

# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 — Save full cleaned dataset
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 60)
log("STEP 10 — Saving phoible_clean.csv")
log("=" * 60)

df.to_csv(CLEAN_PATH, index=False)
log(f"  Saved: {CLEAN_PATH}")
log(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 11 — Deduplicate: one inventory per Glottocode
#
# WHY: PHOIBLE has multiple sources (UPSID, SAPHON, AA, etc.) for the same
# language. If we count /m/ in English from 3 sources, we inflate its
# frequency 3×. We keep the source with the largest inventory (most complete)
# per language, following standard PHOIBLE practice.
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 60)
log("STEP 11 — Deduplication (one inventory per Glottocode)")
log("=" * 60)

# Count phonemes per InventoryID
inv_sizes = (
    df.groupby("InventoryID")["Phoneme"]
    .count()
    .reset_index()
    .rename(columns={"Phoneme": "inv_size"})
)
df = df.merge(inv_sizes, on="InventoryID", how="left")

# For each Glottocode, keep only the InventoryID with the largest inv_size
# (ties broken by first occurrence)
best_inv = (
    df[["Glottocode", "InventoryID", "inv_size"]]
    .drop_duplicates()
    .sort_values("inv_size", ascending=False)
    .groupby("Glottocode", as_index=False)
    .first()[["Glottocode", "InventoryID"]]
)
best_inv_set = set(best_inv["InventoryID"].tolist())

df_dedup = df[df["InventoryID"].isin(best_inv_set)].copy()
df_dedup.drop(columns=["inv_size"], inplace=True)

log(f"  Unique InventoryIDs before dedup: {df['InventoryID'].nunique():,}")
log(f"  Unique Glottocodes before dedup : {df['Glottocode'].nunique():,}")
log(f"  Rows after dedup                : {len(df_dedup):,}")
log(f"  Unique Glottocodes after dedup  : {df_dedup['Glottocode'].nunique():,}")

df_dedup.to_csv(DEDUP_PATH, index=False)
log(f"  Saved: {DEDUP_PATH}")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 60)
log("CLEANING SUMMARY")
log("=" * 60)
log(f"  Raw rows                    : 105,484")
log(f"  After marginal removal      : {len(df):,}")
log(f"  After deduplication         : {len(df_dedup):,}")
log(f"  Unique languages (dedup)    : {df_dedup['Glottocode'].nunique():,}")
log(f"  Unique phonemes (dedup)     : {df_dedup['Phoneme'].nunique():,}")
log(f"  Macroareas covered          : {df_dedup['macroarea'].nunique()}")
log(f"  Language families covered   : {df_dedup['family_name'].nunique()}")
log()

# Save report
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
log(f"[INFO] Cleaning report saved → {REPORT_PATH}")
