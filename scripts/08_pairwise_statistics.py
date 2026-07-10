"""
PHASE 5 - SCRIPT 8
Purpose : Compute pairwise phoneme statistics
          - Co-occurrence matrix (languages containing both phonemes)
          - Jaccard similarity matrix
          - Phi coefficient matrix
          - Conditional frequency table
          - Top strongly associated and mutually exclusive pairs
Outputs :
    outputs/tables/cooccurrence_matrix.csv
    outputs/tables/jaccard_matrix.csv
    outputs/tables/phi_matrix.csv
    outputs/tables/top_associated_pairs.csv
    outputs/tables/top_exclusive_pairs.csv
    outputs/tables/conditional_frequency.csv
    outputs/tables/phase5_report.txt

NOTE : Computing full 2721 × 2721 matrix is memory-intensive.
       We restrict to phonemes appearing in >= 5% of languages (66 phonemes)
       for the matrix, and compute conditional frequency for top 30.
"""

import pandas as pd
import numpy as np
from itertools import combinations
import os
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.join(os.path.dirname(__file__), "..")
DEDUP_PATH = os.path.join(BASE_DIR, "data", "processed", "phoible_dedup.csv")
FREQ_PATH  = os.path.join(BASE_DIR, "outputs", "tables", "phoneme_frequency.csv")
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
log("STEP 1 — Loading data")
log("=" * 60)

df   = pd.read_csv(DEDUP_PATH, low_memory=False)
freq = pd.read_csv(FREQ_PATH)

TOTAL_LANGUAGES = df["Glottocode"].nunique()
log(f"  Total languages : {TOTAL_LANGUAGES:,}")
log(f"  Total phonemes  : {df['Phoneme'].nunique():,}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Select phonemes for matrix analysis
#
# WHY RESTRICT: A 2721×2721 matrix = 7.4 million pairs.
# Most phonemes are extremely rare (appear in 1-2 languages),
# making their pairwise stats statistically meaningless.
# We use phonemes appearing in >= 5% of languages (≈ 103 languages).
# This gives a meaningful, computationally feasible set.
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 60)
log("STEP 2 — Selecting phonemes for matrix (frequency >= 5%)")
log("=" * 60)

common_phonemes = freq[freq["frequency_pct"] >= 5]["Phoneme"].tolist()
log(f"  Phonemes with freq >= 5%: {len(common_phonemes)}")

# Filter main dataframe to only these phonemes
df_common = df[df["Phoneme"].isin(common_phonemes)].copy()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Build language × phoneme binary presence matrix
#
# WHAT : A matrix where rows = languages, columns = phonemes
#        Cell value = 1 if language has that phoneme, 0 if not
# WHY  : This is the foundation for all pairwise calculations.
#        Every co-occurrence, Jaccard, and phi calculation
#        is derived from this binary matrix.
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 60)
log("STEP 3 — Building language × phoneme binary matrix")
log("=" * 60)

presence = (
    df_common.groupby(["Glottocode", "Phoneme"])
    .size()
    .unstack(fill_value=0)
    .clip(upper=1)                # ensure binary (0 or 1)
)

# Add languages that have none of these common phonemes as all-zero rows
all_langs = df["Glottocode"].unique()
missing   = [g for g in all_langs if g not in presence.index]
if missing:
    missing_df = pd.DataFrame(
        0, index=missing, columns=presence.columns
    )
    presence = pd.concat([presence, missing_df])

presence = presence.reindex(columns=common_phonemes, fill_value=0)
presence = presence.astype(np.int8)

log(f"  Matrix shape: {presence.shape[0]} languages × {presence.shape[1]} phonemes")
log(f"  Memory usage: {presence.memory_usage(deep=True).sum() / 1024:.1f} KB")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Co-occurrence matrix
#
# WHAT : M[i][j] = number of languages containing both phoneme i and phoneme j
# HOW  : Matrix multiplication: P^T × P
#        Each cell (i,j) = dot product of columns i and j
#        = sum of (1×1) products = count of languages with both
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 60)
log("STEP 4 — Computing co-occurrence matrix (P^T × P)")
log("=" * 60)

P          = presence.values.astype(np.int16)
cooc_matrix = P.T @ P                          # shape: (n_phonemes, n_phonemes)
cooc_df     = pd.DataFrame(
    cooc_matrix,
    index   = common_phonemes,
    columns = common_phonemes,
)
log(f"  Co-occurrence matrix shape: {cooc_df.shape}")
log(f"  Diagonal (self-count) max : {np.diag(cooc_matrix).max()} (= /m/ count)")
log(f"  Diagonal (self-count) min : {np.diag(cooc_matrix).min()}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Jaccard similarity matrix
#
# Jaccard(A,B) = |A∩B| / |A∪B|
#              = cooc(A,B) / (count(A) + count(B) - cooc(A,B))
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 60)
log("STEP 5 — Computing Jaccard similarity matrix")
log("=" * 60)

counts        = np.diag(cooc_matrix)           # individual phoneme counts
union_matrix  = (
    counts[:, None] + counts[None, :] - cooc_matrix
)                                              # |A| + |B| - |A∩B| = |A∪B|
# Avoid division by zero
with np.errstate(invalid="ignore", divide="ignore"):
    jaccard_matrix = np.where(
        union_matrix == 0,
        0.0,
        cooc_matrix / union_matrix
    )
np.fill_diagonal(jaccard_matrix, 1.0)          # self-similarity = 1

jaccard_df = pd.DataFrame(
    jaccard_matrix,
    index   = common_phonemes,
    columns = common_phonemes,
)
log(f"  Jaccard matrix shape  : {jaccard_df.shape}")
log(f"  Max off-diagonal value: {jaccard_matrix[jaccard_matrix < 1].max():.4f}")
log(f"  Min value             : {jaccard_matrix.min():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Phi coefficient matrix
#
# For each pair (A, B):
#   n11 = languages with both A and B
#   n10 = languages with A but not B
#   n01 = languages with B but not A
#   n00 = languages with neither
#   phi = (n11*n00 - n10*n01) / sqrt(n1. * n0. * n.1 * n.0)
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 60)
log("STEP 6 — Computing Phi coefficient matrix")
log("=" * 60)

N       = TOTAL_LANGUAGES
n11     = cooc_matrix                          # both present
n1_dot  = counts[:, None]                      # A present (any B)
n_dot1  = counts[None, :]                      # B present (any A)
n10     = n1_dot  - n11                        # A present, B absent
n01     = n_dot1  - n11                        # B present, A absent
n00     = N - n11 - n10 - n01                  # neither present
n0_dot  = N - n1_dot                           # A absent
n_dot0  = N - n_dot1                           # B absent

numerator   = (n11 * n00) - (n10 * n01)
denominator = np.sqrt(
    n1_dot.astype(float) *
    n0_dot.astype(float) *
    n_dot1.astype(float) *
    n_dot0.astype(float)
)
with np.errstate(invalid="ignore", divide="ignore"):
    phi_matrix = np.where(denominator == 0, 0.0, numerator / denominator)
np.fill_diagonal(phi_matrix, 1.0)

phi_df = pd.DataFrame(
    phi_matrix,
    index   = common_phonemes,
    columns = common_phonemes,
)
log(f"  Phi matrix shape      : {phi_df.shape}")
log(f"  Max phi (off-diagonal): {phi_matrix[phi_matrix < 1].max():.4f}")
log(f"  Min phi               : {phi_matrix.min():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Extract top strongly associated pairs (high Jaccard + high phi)
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 60)
log("STEP 7 — Top strongly associated phoneme pairs")
log("=" * 60)

pairs = []
phoneme_list = common_phonemes
for i, a in enumerate(phoneme_list):
    for j, b in enumerate(phoneme_list):
        if j <= i:
            continue
        pairs.append({
            "phoneme_A"    : a,
            "phoneme_B"    : b,
            "cooccurrence" : int(cooc_matrix[i, j]),
            "jaccard"      : round(float(jaccard_matrix[i, j]), 4),
            "phi"          : round(float(phi_matrix[i, j]), 4),
            "freq_A"       : int(counts[i]),
            "freq_B"       : int(counts[j]),
        })

pairs_df = pd.DataFrame(pairs)

# Get segment class for each phoneme
class_map = freq.set_index("Phoneme")["segment_class"].to_dict()
pairs_df["class_A"] = pairs_df["phoneme_A"].map(class_map)
pairs_df["class_B"] = pairs_df["phoneme_B"].map(class_map)
pairs_df["pair_type"] = pairs_df.apply(
    lambda r: f"{r['class_A']}-{r['class_B']}", axis=1
)

# Top 20 by Jaccard (most similar)
top_assoc = pairs_df.nlargest(20, "jaccard")
log()
log(f"  {'Phoneme A':<12} {'Phoneme B':<12} {'Jaccard':>8} "
    f"{'Phi':>8} {'Co-occur':>10} {'Type':<20}")
log(f"  {'-'*12} {'-'*12} {'-'*8} {'-'*8} {'-'*10} {'-'*20}")
for _, row in top_assoc.iterrows():
    log(f"  {row['phoneme_A']:<12} {row['phoneme_B']:<12} "
        f"{row['jaccard']:>8.4f} {row['phi']:>8.4f} "
        f"{row['cooccurrence']:>10,} {row['pair_type']:<20}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Top mutually exclusive pairs (lowest phi)
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 60)
log("STEP 8 — Top mutually exclusive phoneme pairs (lowest phi)")
log("=" * 60)

top_excl = pairs_df.nsmallest(20, "phi")
log()
log(f"  {'Phoneme A':<12} {'Phoneme B':<12} {'Phi':>8} "
    f"{'Jaccard':>8} {'Co-occur':>10} {'Type':<20}")
log(f"  {'-'*12} {'-'*12} {'-'*8} {'-'*8} {'-'*10} {'-'*20}")
for _, row in top_excl.iterrows():
    log(f"  {row['phoneme_A']:<12} {row['phoneme_B']:<12} "
        f"{row['phi']:>8.4f} {row['jaccard']:>8.4f} "
        f"{row['cooccurrence']:>10,} {row['pair_type']:<20}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — Conditional frequency table
#
# WHAT : P(B|A) = P(language has B given it has A)
# WHY  : Tells us implicational universals —
#        e.g. "if a language has /ŋ/, does it always have /n/?"
#        This is a core question in Greenbergian typology.
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 60)
log("STEP 9 — Conditional frequency P(B|A) for top 30 phonemes")
log("=" * 60)

top30 = freq.nlargest(30, "language_count")["Phoneme"].tolist()
top30 = [p for p in top30 if p in common_phonemes]

cond_rows = []
for a in top30:
    if a not in presence.columns:
        continue
    langs_with_a = presence[presence[a] == 1].index
    n_a = len(langs_with_a)
    for b in top30:
        if a == b or b not in presence.columns:
            continue
        n_ab = int(presence.loc[langs_with_a, b].sum())
        cond_rows.append({
            "given_A"  : a,
            "phoneme_B": b,
            "P(B|A)"   : round(n_ab / n_a, 4) if n_a > 0 else 0.0,
            "n_A"      : n_a,
            "n_AB"     : n_ab,
        })

cond_df = pd.DataFrame(cond_rows)

# Pivot for display
cond_pivot = cond_df.pivot(
    index="given_A", columns="phoneme_B", values="P(B|A)"
).round(3)

log()
log("  Sample — P(B | /m/) for top phonemes:")
if "m" in cond_pivot.index:
    m_row = cond_pivot.loc["m"].sort_values(ascending=False).head(10)
    for ph, val in m_row.items():
        log(f"    P({ph} | m) = {val:.3f}")

log()
log("  Sample — P(B | /a/) for top phonemes:")
if "a" in cond_pivot.index:
    a_row = cond_pivot.loc["a"].sort_values(ascending=False).head(10)
    for ph, val in a_row.items():
        log(f"    P({ph} | a) = {val:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 — Implicational universals check
#
# WHAT : Find pairs where P(B|A) >= 0.90
#        These are near-universal implications:
#        "Nearly every language with A also has B"
# WHY  : This is the empirical basis for Greenberg's universals.
#        e.g. if a language has voiced stops, it tends to have voiceless stops.
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 60)
log("STEP 10 — Implicational universals (P(B|A) >= 0.90)")
log("=" * 60)

strong_impl = cond_df[cond_df["P(B|A)"] >= 0.90].sort_values(
    "P(B|A)", ascending=False
)
log(f"  Pairs with P(B|A) >= 0.90: {len(strong_impl)}")
log()
log(f"  {'Given A':<12} {'→  B':<12} {'P(B|A)':>8} "
    f"{'n(A)':>8} {'n(A,B)':>8}")
log(f"  {'-'*12} {'-'*12} {'-'*8} {'-'*8} {'-'*8}")
for _, row in strong_impl.head(25).iterrows():
    log(f"  {row['given_A']:<12} → {row['phoneme_B']:<12} "
        f"{row['P(B|A)']:>8.3f} {row['n_A']:>8,} {row['n_AB']:>8,}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 11 — Save all outputs
# ─────────────────────────────────────────────────────────────────────────────
log()
log("=" * 60)
log("STEP 11 — Saving outputs")
log("=" * 60)

cooc_df.to_csv(
    os.path.join(TABLES_DIR, "cooccurrence_matrix.csv"))
log("  Saved: cooccurrence_matrix.csv")

jaccard_df.to_csv(
    os.path.join(TABLES_DIR, "jaccard_matrix.csv"))
log("  Saved: jaccard_matrix.csv")

phi_df.to_csv(
    os.path.join(TABLES_DIR, "phi_matrix.csv"))
log("  Saved: phi_matrix.csv")

top_assoc.to_csv(
    os.path.join(TABLES_DIR, "top_associated_pairs.csv"), index=False)
log("  Saved: top_associated_pairs.csv")

top_excl.to_csv(
    os.path.join(TABLES_DIR, "top_exclusive_pairs.csv"), index=False)
log("  Saved: top_exclusive_pairs.csv")

cond_df.to_csv(
    os.path.join(TABLES_DIR, "conditional_frequency.csv"), index=False)
log("  Saved: conditional_frequency.csv")

cond_pivot.to_csv(
    os.path.join(TABLES_DIR, "conditional_frequency_pivot.csv"))
log("  Saved: conditional_frequency_pivot.csv")

strong_impl.to_csv(
    os.path.join(TABLES_DIR, "implicational_universals.csv"), index=False)
log("  Saved: implicational_universals.csv")

with open(os.path.join(TABLES_DIR, "phase5_report.txt"),
          "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
log("  Saved: phase5_report.txt")

log()
log("[DONE] Phase 5 — Pairwise Statistics complete.")
