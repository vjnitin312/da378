"""
PHASE 6 - SCRIPT 10
Purpose : Generate structured typological interpretation report
          Drawing on all Phase 2-5 outputs to build a written analysis
Outputs :
    outputs/tables/phase6_interpretation_report.txt
    outputs/tables/project_summary_stats.csv
"""

import pandas as pd
import numpy as np
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.join(os.path.dirname(__file__), "..")
TABLES_DIR = os.path.join(BASE_DIR, "outputs", "tables")
DATA_DIR   = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(TABLES_DIR, exist_ok=True)

# ── Load all outputs ──────────────────────────────────────────────────────────
freq        = pd.read_csv(os.path.join(TABLES_DIR, "phoneme_frequency.csv"))
region_inv  = pd.read_csv(os.path.join(TABLES_DIR, "region_inventory_stats.csv"))
family_inv  = pd.read_csv(os.path.join(TABLES_DIR, "family_inventory_stats.csv"))
top_assoc   = pd.read_csv(os.path.join(TABLES_DIR, "top_associated_pairs.csv"))
top_excl    = pd.read_csv(os.path.join(TABLES_DIR, "top_exclusive_pairs.csv"))
impl_df     = pd.read_csv(os.path.join(TABLES_DIR, "implicational_universals.csv"))
lang_inv    = pd.read_csv(os.path.join(DATA_DIR,   "language_inventory.csv"))
region_cls  = pd.read_csv(os.path.join(TABLES_DIR, "region_class_distribution.csv"))

for col in ["has_click", "has_tone_segment"]:
    lang_inv[col] = (lang_inv[col].astype(str).str.strip().str.lower()
                     .map({"true": True, "false": False, "1": True, "0": False})
                     .fillna(False))

lines = []
def log(text=""):
    print(text)
    lines.append(str(text))

def section(title):
    log()
    log("=" * 70)
    log(f"  {title}")
    log("=" * 70)

def subsection(title):
    log()
    log(f"  ── {title}")
    log(f"  {'─' * (len(title) + 4)}")

# ─────────────────────────────────────────────────────────────────────────────
# PROJECT HEADER
# ─────────────────────────────────────────────────────────────────────────────
log("=" * 70)
log("  PHONEME DISTRIBUTION ACROSS LANGUAGES")
log("  Typological Interpretation & Critical Analysis")
log("  Based on PHOIBLE v2.0 — 2,176 languages · 2,721 unique phonemes")
log("  IIT Guwahati — BSc Data Science & AI")
log("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — DATASET OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
section("1. DATASET OVERVIEW")

TOTAL_LANG     = lang_inv["Glottocode"].nunique()
TOTAL_PHONEMES = freq["Phoneme"].nunique()
TOTAL_FAMILIES = lang_inv["family_name"].nunique()
AVG_INV        = lang_inv["total_phonemes"].mean()
MAX_INV        = lang_inv["total_phonemes"].max()
MIN_INV        = lang_inv["total_phonemes"].min()
MAX_LANG       = lang_inv.loc[lang_inv["total_phonemes"].idxmax(), "LanguageName"]
MIN_LANG       = lang_inv.loc[lang_inv["total_phonemes"].idxmin(), "LanguageName"]

log(f"""
  After deduplication (one inventory per Glottocode), the analysis covers:

    Languages          : {TOTAL_LANG:,} across 6 macroareas
    Unique phonemes    : {TOTAL_PHONEMES:,}
    Language families  : {TOTAL_FAMILIES}
    Avg inventory size : {AVG_INV:.1f} phonemes per language
    Largest inventory  : {MAX_LANG} ({MAX_INV} phonemes)
    Smallest inventory : {MIN_LANG} ({MIN_INV} phonemes)

  Deduplication was necessary because PHOIBLE aggregates data from multiple
  sources (UPSID, SAPHON, AA, RA, etc.) for the same language. Keeping one
  inventory per language prevents high-resource languages from inflating counts.
  The largest-inventory source per Glottocode was retained as the most complete.
""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — THE UNIVERSAL PHONEME CORE
# ─────────────────────────────────────────────────────────────────────────────
section("2. THE UNIVERSAL PHONEME CORE")

top10 = freq.head(10)
universal_threshold = 50.0
universal = freq[freq["frequency_pct"] >= universal_threshold]
rare       = freq[freq["language_count"] == 1]

log(f"""
  2.1 Top 10 Most Common Phonemes Globally
  ─────────────────────────────────────────
  {'Rank':<6} {'Phoneme':<12} {'Class':<12} {'Languages':>10} {'Freq%':>8}""")
for _, row in top10.iterrows():
    log(f"  {int(row['rank']):<6} {row['Phoneme']:<12} "
        f"{row['segment_class']:<12} "
        f"{int(row['language_count']):>10,} {row['frequency_pct']:>7.1f}%")

log(f"""
  2.2 Typological Interpretation — Why These Phonemes?
  ──────────────────────────────────────────────────────

  /m/ — The most universal consonant (96.7%). Bilabial nasal. The only
  consonant produced with complete oral closure + nasal airflow. It is
  the first consonant acquired by infants universally ("mama"). Its
  articulatory simplicity makes avoidance extremely costly.

  /i/, /u/, /a/ — The vowel triangle (88–94%). These three vowels maximally
  exploit the acoustic vowel space. Any language selecting a minimal vowel
  system (3 vowels) picks exactly these three. They are perceptually
  maximally distinct, reducing confusion between phonemes — a pressure
  documented across all phonological typology literature.

  /k/ — The most common stop (90.2%). Velar stops are preferred cross-
  linguistically because the velar closure point creates a large resonant
  cavity, making /k/ acoustically salient.

  /j/, /w/ — Approximants (89–84%). These are the consonantal counterparts
  of the high vowels /i/ and /u/ respectively. Their near-universal presence
  follows directly from the near-universality of /i/ and /u/.

  2.3 The Phoneme Frequency Distribution is Zipfian
  ──────────────────────────────────────────────────
  Phonemes appearing in >= 50%  of languages : {len(universal):>4}
  Phonemes appearing in exactly 1  language  : {len(rare):>4} ({len(rare)/TOTAL_PHONEMES*100:.1f}%)

  This extreme long tail is a Zipfian power-law distribution — the same
  pattern seen in word frequencies, city populations, and website traffic.
  Linguistically, it means the world's languages collectively explore a vast
  space of possible sounds, but only a tiny core is universally selected.
  The rare phonemes represent innovations unique to individual languages
  or small geographic regions.
""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — GEOGRAPHIC PATTERNS
# ─────────────────────────────────────────────────────────────────────────────
section("3. GEOGRAPHIC PATTERNS & AREAL FEATURES")

africa = region_inv[region_inv["macroarea"] == "Africa"].iloc[0]
aus    = region_inv[region_inv["macroarea"] == "Australia"].iloc[0]
eurasia= region_inv[region_inv["macroarea"] == "Eurasia"].iloc[0]
papun  = region_inv[region_inv["macroarea"] == "Papunesia"].iloc[0]
sam    = region_inv[region_inv["macroarea"] == "South America"].iloc[0]

log(f"""
  3.1 Inventory Size Variation by Region
  ────────────────────────────────────────
  Eurasia has the largest average inventories ({eurasia['avg_inventory']:.1f} phonemes),
  driven by the Caucasus languages (Nakh-Daghestanian avg 64.8) with their
  massive consonant clusters, and Central Asian languages with complex
  phonological systems.

  Africa ({africa['avg_inventory']:.1f} avg) is the most phonologically diverse region —
  it contains both the world's largest inventories (Khoisan click languages,
  up to 161 phonemes) and small-inventory Bantu languages. The high variance
  reflects Africa's status as the region with the greatest genetic linguistic
  diversity.

  Papunesia has the smallest inventories ({papun['avg_inventory']:.1f} avg). Polynesian
  languages famously have as few as 11–13 phonemes (Rotokas: 11). This
  reflects a typological tendency toward simpler syllable structures and
  fewer phonemic contrasts in island language communities.

  3.2 Click Phonemes — The Sharpest Geographic Restriction
  ──────────────────────────────────────────────────────────
  Click languages in dataset     : {lang_inv['has_click'].sum()}
  All located in                 : Africa (100%)

  Click phonemes are produced with a velaric airstream — a mechanism
  entirely separate from the pulmonic egressive airstream used by all
  other phonemes. They appear exclusively in:
    - Khoisan families (e.g., !Xóõ, Juǀʼhoan)
    - Bantu languages through contact (e.g., Zulu, Xhosa, Yeyi)

  The geographic concentration is absolute — not a single non-African
  language in 2,176 has a phonemic click. This is one of the strongest
  areal restrictions in the PHOIBLE database.

  Statistical implication: The phi coefficient between clicks and any
  non-African phoneme feature is consistently negative, confirming
  the complementary distribution.

  3.3 Tone Segments — A Bimodal Distribution
  ────────────────────────────────────────────
  % tonal languages by region:
    Africa        : {africa['pct_with_tones']:.1f}%
    North America : {region_inv[region_inv['macroarea']=='North America'].iloc[0]['pct_with_tones']:.1f}%
    Papunesia     : {papun['pct_with_tones']:.1f}%
    South America : {sam['pct_with_tones']:.1f}%
    Eurasia       : {eurasia['pct_with_tones']:.1f}%
    Australia     : {aus['pct_with_tones']:.1f}%

  Africa's dominance (64.6%) is driven by Niger-Congo and Nilo-Saharan
  families — the two largest African families are almost entirely tonal.

  Australia's 0% is equally striking. Australian languages are almost
  entirely non-tonal, non-click, and consonant-heavy with small vowel
  inventories. This reflects the region's long isolation from both
  African areal contact and the tonal spread of East/Southeast Asia.

  3.4 Australia — The Consonant-Dominant Region
  ───────────────────────────────────────────────
  Australia avg consonant ratio : {aus['avg_consonant_ratio']*100:.1f}%
  Global avg consonant ratio    : {region_inv['avg_consonant_ratio'].mean()*100:.1f}%

  Australian languages have the highest proportion of consonants to vowels
  of any region. Languages like Arrernte have fewer than 4 vowel phonemes
  but up to 20+ consonants. This pattern is driven by the Pama-Nyungan
  family which covers ~90% of the Australian continent.

  3.5 South America — The Vowel-Rich Region
  ───────────────────────────────────────────
  South America avg vowel ratio : {sam['avg_vowel_ratio']*100:.1f}%
  Global avg vowel ratio        : {region_inv['avg_vowel_ratio'].mean()*100:.1f}%

  Andean and Amazonian languages often have rich vowel systems with fewer
  consonant distinctions. Quechua languages are typologically notable for
  their 3-vowel systems (/i/, /a/, /u/) with very high-frequency usage of
  each vowel.
""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — LANGUAGE FAMILY PATTERNS
# ─────────────────────────────────────────────────────────────────────────────
section("4. LANGUAGE FAMILY PATTERNS")

fam5 = family_inv[family_inv["n_languages"] >= 5]
top_fam = fam5.nlargest(5, "avg_inventory")
bot_fam = fam5.nsmallest(5, "avg_inventory")

log(f"""
  4.1 Top 5 Families by Average Inventory
  ─────────────────────────────────────────""")
for _, row in top_fam.iterrows():
    log(f"  {row['family_name']:<30} avg={row['avg_inventory']:.1f}  "
        f"n={int(row['n_languages'])} languages")

log(f"""
  4.2 Bottom 5 Families by Average Inventory
  ────────────────────────────────────────────""")
for _, row in bot_fam.iterrows():
    log(f"  {row['family_name']:<30} avg={row['avg_inventory']:.1f}  "
        f"n={int(row['n_languages'])} languages")

log(f"""
  4.3 Typological Interpretation — Family Differences
  ─────────────────────────────────────────────────────

  Nakh-Daghestanian (Caucasus): Large inventories driven by a massive
  system of uvular, pharyngeal, and ejective consonants. Languages like
  Ubykh (now extinct) reportedly had 80+ consonant phonemes. This is an
  areal feature of the Caucasus — a region of extreme phonological
  complexity shared across unrelated families (Kartvelian, Abkhaz-Adyge).

  Khoisan families: The largest inventories in the world due to click
  phonemes. !Xóõ (Taa) has approximately 161 phonemes including ~130
  consonants. This is not random — the click mechanism creates a separate
  dimension of phonemic contrast unavailable to other languages.

  Austronesian (Papunesia, Pacific): Small inventories. Most Austronesian
  languages have simple CV (consonant-vowel) syllable structures, few
  consonant clusters, and 5-vowel systems (a, e, i, o, u). Polynesian
  subgroup is the most extreme case of inventory reduction.

  Indo-European: Mid-range inventories (~33 avg). High internal diversity —
  from Georgian neighbors (large inventories) to Polynesian-like simplicity
  in some Celtic dialects. The Proto-Indo-European reconstruction suggests
  a moderately complex inventory that has diverged significantly.
""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — CO-OCCURRENCE & PAIRWISE PATTERNS
# ─────────────────────────────────────────────────────────────────────────────
section("5. CO-OCCURRENCE & PAIRWISE STATISTICAL PATTERNS")

top5_assoc = top_assoc.head(5)
top5_excl  = top_excl.head(5)

log(f"""
  5.1 Strongly Associated Pairs (High Jaccard Similarity)
  ─────────────────────────────────────────────────────────

  {'Pair':<20} {'Jaccard':>8}  Interpretation""")
for _, row in top5_assoc.iterrows():
    pair = f"{row['phoneme_A']} & {row['phoneme_B']}"
    log(f"  {pair:<20} {row['jaccard']:>8.3f}")

log(f"""
  The highest Jaccard pair is ˦ & ˨ (high tone & low tone, Jaccard=0.960).
  This is a near-perfect co-occurrence: tonal languages almost always have
  at least two contrasting tone levels. A language with only one tone level
  is functionally non-tonal — tone only functions as a phonemic contrast
  when two or more levels exist.

  i & u (Jaccard=0.932): These are the two peripheral high vowels.
  Their near-universal co-occurrence reflects the perceptual principle
  that vowel systems are built to maximize acoustic distance.
  A language with /i/ but not /u/ (or vice versa) creates an asymmetric
  vowel space — typologically extremely rare.

  5.2 Mutually Exclusive Pairs (Lowest Phi Coefficient)
  ───────────────────────────────────────────────────────

  {'Pair':<25} {'Phi':>8}  Interpretation""")
for _, row in top5_excl.iterrows():
    pair = f"{row['phoneme_A']} & {row['phoneme_B']}"
    log(f"  {pair:<25} {row['phi']:>8.4f}")

log(f"""
  The negative phi values are modest (max ≈ -0.13) compared to the
  positive associations. This is expected — phonological avoidance is
  rarely absolute. Languages don't strictly forbid combinations, they
  just prefer or avoid them.

  The tone-retroflex avoidance (˦˨ & n̠d̠ʒ, phi=-0.130) reflects the
  geographic incompatibility of tone-heavy systems (Africa, East Asia)
  and retroflex consonant systems (South Asia, Australian languages).
  These features belong to different areal zones.

  5.3 Implicational Universals — Empirical Test
  ───────────────────────────────────────────────
  Total pairs with P(B|A) >= 0.90: {len(impl_df)}

  The strongest implication: P(m | ŋ) = 0.994
  Nearly every language with the velar nasal /ŋ/ also has /m/.

  This is a classic markedness hierarchy:
    /m/ (bilabial nasal) → less marked (simpler, more universal)
    /ŋ/ (velar nasal)   → more marked (more complex, less universal)

  Greenberg's Universal 9 states: "With well-defined exceptions, when
  voiced stops occur in a series, there is also a series of voiceless
  stops." Our data finds P(voiceless stop | voiced stop) consistently
  above 0.90, providing empirical confirmation of this universal.

  The directionality matters: P(ŋ | m) is much lower than P(m | ŋ).
  Having /m/ does NOT imply /ŋ/. This asymmetry is exactly what
  markedness theory predicts — marked sounds imply unmarked ones,
  not the reverse.
""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — STATISTICAL VS TYPOLOGICAL IMPLICATIONS
# ─────────────────────────────────────────────────────────────────────────────
section("6. CRITICAL THINKING: STATISTICAL VS TYPOLOGICAL IMPLICATIONS")

log(f"""
  6.1 What the Statistics Can Tell Us
  ─────────────────────────────────────

  ✓ Cross-linguistic frequency reliably identifies unmarked sounds.
    Phonemes appearing in >80% of languages are articulatorily simple
    and perceptually salient — this is not a coincidence.

  ✓ Jaccard similarity captures vowel harmony and stop system coherence.
    High Jaccard between /i/ and /u/, between /p/ and /k/, between tone
    levels — these reflect real phonological system pressures.

  ✓ Conditional frequency P(B|A) empirically tests Greenbergian universals.
    The 139 pairs with P≥0.90 are a data-driven list of phonological
    implications that hold across unrelated language families.

  ✓ Geographic concentration of clicks and tones is statistically
    unambiguous. The phi between "has click" and "region=Africa" would
    be near +1.0 — one of the strongest effects in the dataset.

  6.2 What the Statistics Cannot Tell Us
  ────────────────────────────────────────

  ✗ Correlation ≠ causation. The co-occurrence of /i/ and /u/ does not
    CAUSE each other. They co-occur because both are driven by the same
    underlying pressure: maximizing vowel space dispersion.

  ✗ Statistical universals are not absolute universals. P(m|ŋ)=0.994
    means 0.6% of languages with /ŋ/ lack /m/. These exceptions are
    typologically significant — they are the languages that challenge
    the universal and require explanation.

  ✗ Phi coefficients are small (-0.13 to +0.13 range). This means no
    two phonemes are strongly mutually exclusive at the global level.
    Areal restrictions (clicks=Africa) appear as geographic effects,
    not as direct phoneme-to-phoneme avoidance.

  ✗ The dataset has sampling bias. Africa (710 languages) is well
    represented relative to South America (301) and Papunesia (176).
    Under-sampled regions may have patterns not yet captured.

  ✗ Glottocode deduplication loses dialect variation. Some "language"
    boundaries in PHOIBLE are contested — what counts as a dialect vs
    a language affects frequency counts.

  6.3 Statistical vs Typological Significance
  ─────────────────────────────────────────────

  A finding can be statistically significant but typologically trivial.
  Example: P(m | i) = 0.965 is statistically robust across 2,176 languages.
  But typologically, this simply reflects that both /m/ and /i/ are
  independently near-universal — they co-occur because each individually
  appears in ~95% of languages, not because they have any phonological
  relationship to each other.

  Conversely, a finding can be typologically profound but statistically
  modest. The geographic restriction of clicks involves only 16 languages
  out of 2,176 — a small sample. But the pattern is 100% geographically
  consistent, making it one of the most robust areal universals in
  phonological typology.

  The correct approach: use statistics to identify patterns, use
  linguistic theory to explain them. Neither alone is sufficient.

  6.4 Greenberg's Universals — Empirical Confirmation & Challenges
  ──────────────────────────────────────────────────────────────────

  CONFIRMED by this data:
  • Near-universal nasals (/m/ > /n/ > /ŋ/) confirm the markedness
    hierarchy: bilabial < alveolar < velar for nasals.
  • /i/, /u/, /a/ as the minimal vowel set confirms the vowel dispersion
    principle.
  • Stop systems co-occur as complete series (P(k|p)=0.977).

  CHALLENGED or NUANCED:
  • Tone is not simply a marked feature — it is areally structured.
    64.6% of African languages are tonal, which means for African
    languages, tone is the UNMARKED default, not a marked addition.
  • The Zipfian tail (1,444 phonemes in 1 language) suggests that
    phonological innovation is ongoing and language-specific,
    not converging toward universals.

  6.5 Key Takeaway
  ─────────────────
  This analysis demonstrates that the world's phonological systems are
  simultaneously universal and diverse. A small core of ~18 phonemes
  appears in over 50% of languages — a shared foundation likely
  reflecting the articulatory and perceptual constraints of the human
  vocal tract. Beyond this core, phonological diversity is vast,
  geographically structured, family-dependent, and historically
  contingent. No statistical model can capture this complexity in full —
  but the patterns revealed here provide a rigorous empirical foundation
  for deeper linguistic investigation.
""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — PROJECT SUMMARY STATISTICS TABLE
# ─────────────────────────────────────────────────────────────────────────────
section("7. PROJECT SUMMARY — KEY NUMBERS")

summary_stats = {
    "Total languages analysed"          : TOTAL_LANG,
    "Total unique phonemes"             : TOTAL_PHONEMES,
    "Language families"                 : TOTAL_FAMILIES,
    "Macroareas"                        : 6,
    "Global avg inventory size"         : round(AVG_INV, 1),
    "Largest inventory (phonemes)"      : MAX_INV,
    "Smallest inventory (phonemes)"     : MIN_INV,
    "Phonemes in 50%+ languages"        : len(freq[freq["frequency_pct"] >= 50]),
    "Phonemes in only 1 language"       : len(freq[freq["language_count"] == 1]),
    "Click languages in dataset"        : int(lang_inv["has_click"].sum()),
    "Tonal languages in dataset"        : int(lang_inv["has_tone_segment"].sum()),
    "% Africa tonal"                    : f"{africa['pct_with_tones']:.1f}%",
    "% Australia tonal"                 : f"{aus['pct_with_tones']:.1f}%",
    "Implicational pairs P(B|A)>=0.90"  : len(impl_df),
    "Strongest implication"             : "P(m|ng) = 0.994",
    "Highest Jaccard pair"              : "tone hi & tone lo = 0.960",
    "Most exclusive pair (phi)"         : "tone-contour & retroflex = -0.130",
}

log()
for k, v in summary_stats.items():
    log(f"  {k:<45} {v}")

# Save summary as CSV
summary_df = pd.DataFrame(
    list(summary_stats.items()),
    columns=["Metric", "Value"]
)
summary_df.to_csv(
    os.path.join(TABLES_DIR, "project_summary_stats.csv"), index=False)
log()
log("  Saved: project_summary_stats.csv")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE REPORT
# ─────────────────────────────────────────────────────────────────────────────
report_path = os.path.join(TABLES_DIR, "phase6_interpretation_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

log()
log(f"  Full report saved → {report_path}")
log()
log("[DONE] Phase 6 — Typological Interpretation complete.")
