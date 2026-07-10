"""
PHASE 2 - SCRIPT 4
Purpose : Compute cross-linguistic phoneme frequencies
          - How many languages contain each phoneme
          - Global frequency % per phoneme
          - Top 10 most common phonemes globally
          - Frequency breakdown by SegmentClass
Outputs :
    outputs/tables/phoneme_frequency.csv
    outputs/tables/top10_phonemes.csv
    outputs/tables/frequency_by_class.csv
    outputs/tables/frequency_report.txt
"""

import pandas as pd
import numpy as np
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.join(os.path.dirname(__file__), "..")
DEDUP_PATH   = os.path.join(BASE_DIR, "data", "processed", "phoible_dedup.csv")
TABLES_DIR   = os.path.join(BASE_DIR, "outputs", "tables")
os.makedirs(TABLES_DIR, exist_ok=True)

lines = []
def log(text=""):
    print(text)
    lines.append(str(text))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load deduplicated data
# ─────────────────────────────────────────────────────────────────────────────
log("=" * 60)
log("STEP 1 — Loading phoible_dedup.csv")
log("=" * 60)

df = pd.read_csv(DEDUP_PATH, low_memory=False)
log(f"  Rows    : {len(df):,}")
log(f"  Columns : {df.shape[1]}")

TOTAL_LANGUAGES = df["Glottocode"].nunique()
log(f"  Total unique languages: {TOTAL_LANGUAGES:,}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Compute cross-linguistic frequency per phoneme
#
# For each phoneme, count how many distinct Glottocodes contain it.
# Then compute percentage out of total languages.
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 60)
log("STEP 2 — Computing cross-linguistic frequency per phoneme")
log("=" * 60)

freq = (
    df.groupby("Phoneme")
    .agg(
        language_count  = ("Glottocode",    "nunique"),
        segment_class   = ("SegmentClass",  "first"),
        is_click        = ("is_click",       "first"),
        is_tone         = ("is_tone",        "first"),
        is_nasal        = ("is_nasal",       "first"),
    )
    .reset_index()
)

freq["frequency_pct"] = (freq["language_count"] / TOTAL_LANGUAGES * 100).round(2)
freq = freq.sort_values("language_count", ascending=False).reset_index(drop=True)
freq["rank"] = freq.index + 1

log(f"  Total unique phonemes analysed: {len(freq):,}")
log(f"  Phonemes appearing in 50%+ languages : "
    f"{(freq['frequency_pct'] >= 50).sum()}")
log(f"  Phonemes appearing in 10%+ languages : "
    f"{(freq['frequency_pct'] >= 10).sum()}")
log(f"  Phonemes appearing in only 1 language: "
    f"{(freq['language_count'] == 1).sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Top 10 most common phonemes globally
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 60)
log("STEP 3 — Top 10 phonemes globally")
log("=" * 60)

top10 = freq.head(10).copy()

log(f"\n  {'Rank':<6} {'Phoneme':<12} {'Class':<12} "
    f"{'Languages':>10} {'Frequency %':>12}")
log(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*10} {'-'*12}")
for _, row in top10.iterrows():
    log(f"  {int(row['rank']):<6} {row['Phoneme']:<12} "
        f"{row['segment_class']:<12} "
        f"{int(row['language_count']):>10,} "
        f"{row['frequency_pct']:>11.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Frequency breakdown by SegmentClass
#
# How do consonants, vowels, and tones compare in terms of:
# - total unique phonemes
# - average frequency
# - max frequency
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 60)
log("STEP 4 — Frequency breakdown by SegmentClass")
log("=" * 60)

class_stats = (
    freq.groupby("segment_class")
    .agg(
        total_unique_phonemes = ("Phoneme",        "count"),
        avg_frequency_pct     = ("frequency_pct",  "mean"),
        median_frequency_pct  = ("frequency_pct",  "median"),
        max_frequency_pct     = ("frequency_pct",  "max"),
        min_frequency_pct     = ("frequency_pct",  "min"),
        phonemes_above_50pct  = ("frequency_pct",  lambda x: (x >= 50).sum()),
    )
    .reset_index()
)
class_stats["avg_frequency_pct"]    = class_stats["avg_frequency_pct"].round(2)
class_stats["median_frequency_pct"] = class_stats["median_frequency_pct"].round(2)
class_stats["max_frequency_pct"]    = class_stats["max_frequency_pct"].round(2)

log()
log(class_stats.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Top 5 per class
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 60)
log("STEP 5 — Top 5 phonemes per SegmentClass")
log("=" * 60)

for cls in ["consonant", "vowel", "tone"]:
    subset = freq[freq["segment_class"] == cls].head(5)
    log(f"\n  [{cls.upper()}]")
    log(f"  {'Phoneme':<12} {'Languages':>10} {'Frequency %':>12}")
    log(f"  {'-'*12} {'-'*10} {'-'*12}")
    for _, row in subset.iterrows():
        log(f"  {row['Phoneme']:<12} "
            f"{int(row['language_count']):>10,} "
            f"{row['frequency_pct']:>11.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Click and Tone phoneme summary
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 60)
log("STEP 6 — Click phoneme summary")
log("=" * 60)

click_phonemes = freq[freq["is_click"] == True].sort_values(
    "language_count", ascending=False
)
log(f"  Total click phonemes: {len(click_phonemes)}")
log(f"\n  {'Phoneme':<12} {'Languages':>10} {'Frequency %':>12}")
log(f"  {'-'*12} {'-'*10} {'-'*12}")
for _, row in click_phonemes.head(10).iterrows():
    log(f"  {row['Phoneme']:<12} "
        f"{int(row['language_count']):>10,} "
        f"{row['frequency_pct']:>11.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Frequency distribution statistics
#
# WHAT  : Descriptive statistics on the frequency distribution
# WHY   : Most phonemes are rare (appear in very few languages).
#         A small number are universal. This is a typical Zipfian distribution.
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 60)
log("STEP 7 — Frequency distribution statistics")
log("=" * 60)

desc = freq["frequency_pct"].describe(percentiles=[0.25, 0.5, 0.75, 0.90, 0.95])
log()
for stat, val in desc.items():
    log(f"  {stat:<10} {val:.2f}%")

log()
log("  Interpretation:")
log("  → Most phonemes appear in very few languages (long tail)")
log("  → A small core set appears in nearly all languages")
log("  → This is consistent with a Zipfian / power-law distribution")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Save outputs
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 60)
log("STEP 8 — Saving output tables")
log("=" * 60)

freq.to_csv(
    os.path.join(TABLES_DIR, "phoneme_frequency.csv"), index=False
)
log("  Saved: outputs/tables/phoneme_frequency.csv")

top10.to_csv(
    os.path.join(TABLES_DIR, "top10_phonemes.csv"), index=False
)
log("  Saved: outputs/tables/top10_phonemes.csv")

class_stats.to_csv(
    os.path.join(TABLES_DIR, "frequency_by_class.csv"), index=False
)
log("  Saved: outputs/tables/frequency_by_class.csv")

with open(os.path.join(TABLES_DIR, "frequency_report.txt"), "w",
          encoding="utf-8") as f:
    f.write("\n".join(lines))
log("  Saved: outputs/tables/frequency_report.txt")

log()
log("[DONE] Phase 2 — Frequency Analysis complete.")
