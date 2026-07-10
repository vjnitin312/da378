"""
PHASE 1 - SCRIPT 2
Purpose : Inspect raw PHOIBLE data — shape, columns, dtypes, nulls, samples
Output  : Printed report + outputs/tables/inspection_report.txt
"""

import pandas as pd
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.join(os.path.dirname(__file__), "..")
RAW_PATH    = os.path.join(BASE_DIR, "data", "raw", "phoible.csv")
TABLES_DIR  = os.path.join(BASE_DIR, "outputs", "tables")
REPORT_PATH = os.path.join(TABLES_DIR, "inspection_report.txt")

os.makedirs(TABLES_DIR, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
print("[INFO] Loading phoible.csv ...")
df = pd.read_csv(RAW_PATH, low_memory=False)

lines = []   # we collect every output line here → save to file too

def log(text=""):
    print(text)
    lines.append(text)

# ── 1. Shape ──────────────────────────────────────────────────────────────────
log("=" * 60)
log("1. SHAPE")
log("=" * 60)
log(f"   Rows    : {df.shape[0]:,}")
log(f"   Columns : {df.shape[1]}")

# ── 2. Column names and dtypes ────────────────────────────────────────────────
log()
log("=" * 60)
log("2. COLUMNS AND DTYPES")
log("=" * 60)
for col in df.columns:
    log(f"   {col:<30} {str(df[col].dtype):<12}")

# ── 3. Missing values ─────────────────────────────────────────────────────────
log()
log("=" * 60)
log("3. MISSING VALUES (columns with any nulls)")
log("=" * 60)
null_counts = df.isnull().sum()
null_counts = null_counts[null_counts > 0]
if null_counts.empty:
    log("   No missing values found.")
else:
    for col, count in null_counts.items():
        pct = count / len(df) * 100
        log(f"   {col:<30} {count:>7,}  ({pct:.1f}%)")

# ── 4. Unique value counts for key columns ────────────────────────────────────
log()
log("=" * 60)
log("4. UNIQUE VALUE COUNTS (key columns)")
log("=" * 60)
key_cols = [
    "InventoryID", "Glottocode", "ISO6393",
    "LanguageName", "Phoneme", "SegmentClass",
    "macroarea", "family_id"
]
for col in key_cols:
    if col in df.columns:
        log(f"   {col:<30} {df[col].nunique():>6,} unique values")

# ── 5. SegmentClass distribution ──────────────────────────────────────────────
log()
log("=" * 60)
log("5. SEGMENTCLASS DISTRIBUTION")
log("=" * 60)
if "SegmentClass" in df.columns:
    for val, cnt in df["SegmentClass"].value_counts().items():
        pct = cnt / len(df) * 100
        log(f"   {str(val):<20} {cnt:>8,}  ({pct:.1f}%)")

# ── 6. Macroarea distribution ─────────────────────────────────────────────────
log()
log("=" * 60)
log("6. MACROAREA DISTRIBUTION")
log("=" * 60)
if "macroarea" in df.columns:
    for val, cnt in df["macroarea"].value_counts().items():
        pct = cnt / len(df) * 100
        log(f"   {str(val):<20} {cnt:>8,}  ({pct:.1f}%)")

# ── 7. Sample rows ────────────────────────────────────────────────────────────
log()
log("=" * 60)
log("7. FIRST 5 ROWS (selected columns)")
log("=" * 60)
preview_cols = [
    c for c in ["LanguageName", "Phoneme", "SegmentClass",
                 "macroarea", "family_id", "tone"]
    if c in df.columns
]
log(df[preview_cols].head(5).to_string(index=False))

# ── 8. Top 10 most frequent phonemes (raw, before cleaning) ───────────────────
log()
log("=" * 60)
log("8. TOP 10 PHONEMES (raw count, before cleaning)")
log("=" * 60)
if "Phoneme" in df.columns:
    for phoneme, cnt in df["Phoneme"].value_counts().head(10).items():
        log(f"   {str(phoneme):<15} {cnt:>6,} rows")

# ── Save report ───────────────────────────────────────────────────────────────
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

log()
log(f"[INFO] Report saved → {REPORT_PATH}")
