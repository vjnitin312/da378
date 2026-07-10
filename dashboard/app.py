"""
PHASE 4 — STREAMLIT DASHBOARD
File    : dashboard/app.py
Purpose : Interactive dashboard for PHOIBLE Phoneme Distribution Analysis
Run     : streamlit run dashboard/app.py
Tabs    :
    1 — Overview        : dataset summary cards + top 10 bar chart
    2 — Frequency       : phoneme frequency explorer with filters
    3 — Regional        : macroarea comparisons + class distribution
    4 — Language Family : family rankings + heatmap
    5 — World Maps      : Plotly world map + Folium tone/click maps
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import os
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "Phoneme Distribution Across Languages",
    page_icon   = "🌍",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.join(os.path.dirname(__file__), "..")
TABLES_DIR = os.path.join(BASE_DIR, "outputs", "tables")
DATA_DIR   = os.path.join(BASE_DIR, "data", "processed")

# ── Color constants ───────────────────────────────────────────────────────────
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
# DATA LOADING  (cached so it loads once per session)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    freq = pd.read_csv(
        os.path.join(TABLES_DIR, "phoneme_frequency.csv"))
    region_inv = pd.read_csv(
        os.path.join(TABLES_DIR, "region_inventory_stats.csv"))
    region_class = pd.read_csv(
        os.path.join(TABLES_DIR, "region_class_distribution.csv"))
    family_inv = pd.read_csv(
        os.path.join(TABLES_DIR, "family_inventory_stats.csv"))
    lang_inv = pd.read_csv(
        os.path.join(DATA_DIR, "language_inventory.csv"))

    # Fix bool columns
    for col in ["has_click", "has_tone_segment"]:
        lang_inv[col] = (
            lang_inv[col].astype(str).str.strip().str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
            .fillna(False)
        )
    for col in ["is_click", "is_tone", "is_nasal"]:
        freq[col] = (
            freq[col].astype(str).str.strip().str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
            .fillna(False)
        )
    lang_cl = pd.read_csv(os.path.join(DATA_DIR, "language_clusters.csv"))
    sil_df  = pd.read_csv(os.path.join(TABLES_DIR, "silhouette_scores.csv"))
    return freq, region_inv, region_class, family_inv, lang_inv, lang_cl, sil_df

freq, region_inv, region_class, family_inv, lang_inv, lang_cl, sil_df = load_data()

TOTAL_LANGUAGES = int(lang_inv["Glottocode"].nunique())
TOTAL_PHONEMES  = int(freq["Phoneme"].nunique())
TOTAL_FAMILIES  = int(lang_inv["family_name"].nunique())

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://phoible.org/images/phoible-logo.png", width=160)
    st.markdown("## 🌍 Phoneme Explorer")
    st.markdown("---")

    st.markdown("### Filters")

    # Region filter
    all_regions = sorted(lang_inv["macroarea"].dropna().unique().tolist())
    selected_regions = st.multiselect(
        "Macroarea",
        options=all_regions,
        default=all_regions,
        help="Filter by geographic region"
    )

    # Segment class filter
    selected_classes = st.multiselect(
        "Phoneme Class",
        options=["consonant", "vowel", "tone"],
        default=["consonant", "vowel", "tone"],
        help="Filter by segment class"
    )

    # Min frequency slider
    min_freq = st.slider(
        "Min. Frequency % (phoneme explorer)",
        min_value=0.0,
        max_value=50.0,
        value=1.0,
        step=0.5,
        help="Show only phonemes appearing in at least this % of languages"
    )

    # Family min language count
    min_lang_count = st.slider(
        "Min. languages per family",
        min_value=1,
        max_value=30,
        value=5,
        help="Filter language families by minimum number of languages"
    )

    st.markdown("---")
    st.markdown(
        "**Dataset:** PHOIBLE v2.0  \n"
        "**Source:** [phoible.org](https://phoible.org)  \n"
        "**Languages:** 2,176  \n"
        "**Phonemes:** 3,020 inventories"
    )

# ─────────────────────────────────────────────────────────────────────────────
# FILTERED DATA (responds to sidebar)
# ─────────────────────────────────────────────────────────────────────────────
lang_filtered = lang_inv[
    lang_inv["macroarea"].isin(selected_regions)
].copy()

freq_filtered = freq[
    (freq["segment_class"].isin(selected_classes)) &
    (freq["frequency_pct"] >= min_freq)
].copy()

family_filtered = family_inv[
    family_inv["n_languages"] >= min_lang_count
].copy()

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center; color:#2C3E50;'>"
    "🌍 Phoneme Distribution Across Languages</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; color:#7F8C8D; font-size:16px;'>"
    "Exploring PHOIBLE v2.0 — 2,176 languages · 2,721 unique phonemes"
    " · 177 language families · 6 macroareas</p>",
    unsafe_allow_html=True
)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "🔠 Frequency Explorer",
    "🌐 Regional Analysis",
    "👨‍👩‍👧‍👦 Language Families",
    "🗺️ World Maps",
    "🤖 Clustering",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Dataset at a Glance")

    # Metric cards
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Languages",       f"{TOTAL_LANGUAGES:,}")
    c2.metric("Unique Phonemes", f"{TOTAL_PHONEMES:,}")
    c3.metric("Language Families", f"{TOTAL_FAMILIES:,}")
    c4.metric("Macroareas",      "6")
    c5.metric("Avg Inventory",   f"{lang_inv['total_phonemes'].mean():.1f}")

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("#### Top 10 Most Common Phonemes Globally")
        top10 = freq.head(10).copy()
        top10["color"] = top10["segment_class"].map(CLASS_COLORS)

        fig_top10 = px.bar(
            top10,
            x="frequency_pct",
            y="Phoneme",
            orientation="h",
            color="segment_class",
            color_discrete_map=CLASS_COLORS,
            text="frequency_pct",
            hover_data={"language_count": True, "segment_class": True},
            labels={
                "frequency_pct"  : "Frequency (%)",
                "Phoneme"        : "Phoneme (IPA)",
                "segment_class"  : "Class",
                "language_count" : "Languages",
            },
        )
        fig_top10.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside"
        )
        fig_top10.update_layout(
            plot_bgcolor="white",
            xaxis=dict(title="% of 2,176 languages",
                       gridcolor="#eeeeee", range=[0, 108]),
            yaxis=dict(title="", categoryorder="total ascending"),
            legend_title="Class",
            height=420,
            margin=dict(l=10, r=60, t=10, b=10),
        )
        st.plotly_chart(fig_top10, use_container_width=True)

    with col_right:
        st.markdown("#### Phoneme Class Breakdown")

        class_counts = freq.groupby("segment_class").agg(
            total_phonemes = ("Phoneme", "count"),
            avg_freq       = ("frequency_pct", "mean"),
        ).reset_index()

        fig_pie = px.pie(
            class_counts,
            names="segment_class",
            values="total_phonemes",
            color="segment_class",
            color_discrete_map=CLASS_COLORS,
            hole=0.45,
        )
        fig_pie.update_traces(
            textinfo="label+percent",
            textfont_size=13,
        )
        fig_pie.update_layout(
            showlegend=False,
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("#### Class Statistics")
        class_display = class_counts.rename(columns={
            "segment_class"  : "Class",
            "total_phonemes" : "Unique Phonemes",
            "avg_freq"       : "Avg Freq %",
        })
        class_display["Avg Freq %"] = class_display["Avg Freq %"].round(2)
        st.dataframe(class_display, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Frequency Distribution — All 2,721 Phonemes")
    st.caption(
        "Most phonemes appear in very few languages (long tail). "
        "A tiny universal core appears in 80–97% of all languages."
    )

    fig_dist = px.histogram(
        freq,
        x="frequency_pct",
        color="segment_class",
        color_discrete_map=CLASS_COLORS,
        nbins=60,
        log_y=True,
        barmode="overlay",
        opacity=0.75,
        labels={
            "frequency_pct"  : "Frequency (% of languages)",
            "segment_class"  : "Class",
            "count"          : "Number of Phonemes",
        },
    )
    fig_dist.update_layout(
        plot_bgcolor="white",
        yaxis=dict(title="Number of phonemes (log)", gridcolor="#eeeeee"),
        xaxis=dict(title="Frequency % across languages", gridcolor="#eeeeee"),
        legend_title="Class",
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_dist, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — FREQUENCY EXPLORER
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Phoneme Frequency Explorer")
    st.caption(
        f"Showing phonemes with frequency ≥ {min_freq}% | "
        f"Classes: {', '.join(selected_classes)} | "
        f"Total shown: {len(freq_filtered):,} phonemes"
    )

    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown("#### Top Phonemes by Frequency")
        top_n = st.slider("Show top N phonemes", 5, 50, 20, key="topn")

        top_n_df = freq_filtered.head(top_n).copy()

        fig_freq = px.bar(
            top_n_df,
            x="Phoneme",
            y="frequency_pct",
            color="segment_class",
            color_discrete_map=CLASS_COLORS,
            text="frequency_pct",
            hover_data={
                "language_count"  : True,
                "segment_class"   : True,
                "frequency_pct"   : True,
            },
            labels={
                "frequency_pct"  : "Frequency (%)",
                "Phoneme"        : "Phoneme (IPA)",
                "segment_class"  : "Class",
                "language_count" : "Languages",
            },
        )
        fig_freq.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside"
        )
        fig_freq.update_layout(
            plot_bgcolor="white",
            yaxis=dict(
                title="Frequency (% of languages)",
                gridcolor="#eeeeee",
                range=[0, max(top_n_df["frequency_pct"]) * 1.15]
            ),
            xaxis=dict(title="Phoneme (IPA symbol)"),
            legend_title="Class",
            height=430,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_freq, use_container_width=True)

    with col_b:
        st.markdown("#### Top 5 Per Class")

        for cls in ["consonant", "vowel", "tone"]:
            if cls not in selected_classes:
                continue
            subset = freq[freq["segment_class"] == cls].head(5)
            color  = CLASS_COLORS[cls]

            fig_cls = px.bar(
                subset,
                x="Phoneme",
                y="frequency_pct",
                text="frequency_pct",
                labels={"frequency_pct": "Freq %", "Phoneme": ""},
                title=f"{cls.capitalize()}s",
            )
            fig_cls.update_traces(
                marker_color=color,
                texttemplate="%{text:.1f}%",
                textposition="outside"
            )
            fig_cls.update_layout(
                plot_bgcolor="white",
                yaxis=dict(gridcolor="#eeeeee",
                           range=[0, subset["frequency_pct"].max() * 1.2]),
                height=210,
                margin=dict(l=5, r=5, t=30, b=5),
                title_font_size=13,
                showlegend=False,
            )
            st.plotly_chart(fig_cls, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Rank vs Frequency Scatter")
    st.caption(
        "Each dot is a phoneme. "
        "The steep drop-off confirms a Zipfian distribution — "
        "a few phonemes are near-universal, the vast majority are rare."
    )

    freq_above = freq[freq["frequency_pct"] >= min_freq].copy()
    fig_scatter = px.scatter(
        freq_above,
        x="rank",
        y="frequency_pct",
        color="segment_class",
        color_discrete_map=CLASS_COLORS,
        hover_data={"Phoneme": True, "language_count": True},
        labels={
            "rank"           : "Phoneme Rank (1 = most common)",
            "frequency_pct"  : "Frequency (%)",
            "segment_class"  : "Class",
            "language_count" : "Languages",
        },
        opacity=0.7,
        size_max=8,
    )
    fig_scatter.update_layout(
        plot_bgcolor="white",
        yaxis=dict(gridcolor="#eeeeee"),
        xaxis=dict(gridcolor="#eeeeee"),
        legend_title="Class",
        height=350,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Full Frequency Table")
    st.dataframe(
        freq_filtered[[
            "rank", "Phoneme", "segment_class",
            "language_count", "frequency_pct"
        ]].rename(columns={
            "rank"           : "Rank",
            "segment_class"  : "Class",
            "language_count" : "Languages",
            "frequency_pct"  : "Frequency %",
        }),
        use_container_width=True,
        hide_index=True,
        height=350,
    )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — REGIONAL ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Regional Analysis")

    # ── Row 1: summary metrics per selected region ─────────────────────────
    region_show = region_inv[
        region_inv["macroarea"].isin(selected_regions)
    ].sort_values("avg_inventory", ascending=False)

    cols = st.columns(len(region_show))
    for col, (_, row) in zip(cols, region_show.iterrows()):
        col.metric(
            label = row["macroarea"],
            value = f"{row['avg_inventory']:.1f} phonemes",
            delta = f"{int(row['n_languages'])} languages",
        )

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Avg Inventory Size by Region")
        fig_reg = px.bar(
            region_show,
            x="macroarea",
            y="avg_inventory",
            color="macroarea",
            color_discrete_map=REGION_COLORS,
            error_y="std_inventory",
            text="avg_inventory",
            hover_data={
                "n_languages"    : True,
                "avg_consonants" : True,
                "avg_vowels"     : True,
                "pct_with_tones" : True,
            },
            labels={
                "avg_inventory" : "Avg Inventory",
                "macroarea"     : "Region",
            },
        )
        fig_reg.update_traces(
            texttemplate="%{text:.1f}",
            textposition="outside"
        )
        fig_reg.update_layout(
            plot_bgcolor="white",
            yaxis=dict(gridcolor="#eeeeee", title="Avg Phoneme Inventory"),
            showlegend=False,
            height=380,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_reg, use_container_width=True)

    with col2:
        st.markdown("#### Phoneme Class % by Region (Stacked)")
        pivot = region_class[
            region_class["macroarea"].isin(selected_regions)
        ].pivot(index="macroarea", columns="SegmentClass", values="pct").fillna(0)

        fig_stack = go.Figure()
        for cls, color in CLASS_COLORS.items():
            if cls in pivot.columns:
                fig_stack.add_trace(go.Bar(
                    name=cls.capitalize(),
                    x=pivot.index.tolist(),
                    y=pivot[cls].values,
                    marker_color=color,
                    text=[f"{v:.1f}%" for v in pivot[cls].values],
                    textposition="inside",
                    insidetextanchor="middle",
                ))
        fig_stack.update_layout(
            barmode="stack",
            plot_bgcolor="white",
            yaxis=dict(gridcolor="#eeeeee", title="% of phoneme tokens"),
            xaxis=dict(title="Region"),
            legend_title="Class",
            height=380,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_stack, use_container_width=True)

    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### Tone & Click Concentration by Region")
        fig_tc = go.Figure()
        fig_tc.add_trace(go.Bar(
            name="% Tonal languages",
            x=region_show["macroarea"],
            y=region_show["pct_with_tones"],
            marker_color="#1A9641",
            text=[f"{v:.0f}%" for v in region_show["pct_with_tones"]],
            textposition="outside",
        ))
        fig_tc.add_trace(go.Bar(
            name="% with Click phonemes",
            x=region_show["macroarea"],
            y=region_show["pct_with_clicks"],
            marker_color="#8B0000",
            text=[f"{v:.1f}%" for v in region_show["pct_with_clicks"]],
            textposition="outside",
        ))
        fig_tc.update_layout(
            barmode="group",
            plot_bgcolor="white",
            yaxis=dict(gridcolor="#eeeeee", title="% of languages"),
            xaxis=dict(title="Region"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=360,
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig_tc, use_container_width=True)

    with col4:
        st.markdown("#### Inventory Size Distribution by Region")
        lang_reg = lang_filtered.copy()
        order = (lang_reg.groupby("macroarea")["total_phonemes"]
                 .median().sort_values(ascending=False).index.tolist())

        fig_box = px.box(
            lang_reg,
            x="macroarea",
            y="total_phonemes",
            color="macroarea",
            color_discrete_map=REGION_COLORS,
            category_orders={"macroarea": order},
            points="outliers",
            hover_data=["LanguageName", "family_name"],
            labels={
                "macroarea"      : "Region",
                "total_phonemes" : "Inventory Size",
            },
        )
        fig_box.update_layout(
            plot_bgcolor="white",
            yaxis=dict(gridcolor="#eeeeee", title="Total Phonemes"),
            showlegend=False,
            height=360,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Regional Statistics Table")
    st.dataframe(
        region_show.rename(columns={
            "macroarea"          : "Region",
            "n_languages"        : "Languages",
            "avg_inventory"      : "Avg Inventory",
            "median_inventory"   : "Median Inventory",
            "avg_consonants"     : "Avg Consonants",
            "avg_vowels"         : "Avg Vowels",
            "avg_tones"          : "Avg Tones",
            "pct_with_clicks"    : "% with Clicks",
            "pct_with_tones"     : "% Tonal",
        }).drop(columns=["std_inventory", "avg_consonant_ratio",
                         "avg_vowel_ratio"], errors="ignore"),
        use_container_width=True,
        hide_index=True,
    )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — LANGUAGE FAMILIES
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Language Family Analysis")
    st.caption(
        f"Showing families with ≥ {min_lang_count} languages "
        f"| {len(family_filtered)} families"
    )

    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown("#### Top 20 Families by Average Inventory Size")
        top20_fam = (family_filtered
                     .nlargest(20, "avg_inventory")
                     .sort_values("avg_inventory", ascending=True))

        fig_fam = px.bar(
            top20_fam,
            x="avg_inventory",
            y="family_name",
            orientation="h",
            color="avg_inventory",
            color_continuous_scale="Purples",
            text="avg_inventory",
            hover_data={
                "n_languages"   : True,
                "avg_consonants": True,
                "avg_vowels"    : True,
                "pct_with_tones": True,
            },
            labels={
                "avg_inventory" : "Avg Inventory Size",
                "family_name"   : "Family",
                "n_languages"   : "Languages",
            },
        )
        fig_fam.update_traces(
            texttemplate="%{text:.1f}",
            textposition="outside"
        )
        fig_fam.update_layout(
            plot_bgcolor="white",
            xaxis=dict(gridcolor="#eeeeee", title="Avg Inventory Size",
                       range=[0, top20_fam["avg_inventory"].max() * 1.15]),
            yaxis=dict(title=""),
            coloraxis_showscale=False,
            height=540,
            margin=dict(l=10, r=60, t=10, b=10),
        )
        st.plotly_chart(fig_fam, use_container_width=True)

    with col_b:
        st.markdown("#### Bottom 10 Families")
        bot10_fam = (family_filtered
                     .nsmallest(10, "avg_inventory")
                     .sort_values("avg_inventory", ascending=False))

        fig_bot = px.bar(
            bot10_fam,
            x="avg_inventory",
            y="family_name",
            orientation="h",
            color="avg_inventory",
            color_continuous_scale="Oranges",
            text="avg_inventory",
            labels={
                "avg_inventory" : "Avg Inventory Size",
                "family_name"   : "Family",
            },
        )
        fig_bot.update_traces(
            texttemplate="%{text:.1f}",
            textposition="outside"
        )
        fig_bot.update_layout(
            plot_bgcolor="white",
            xaxis=dict(gridcolor="#eeeeee", title="Avg Inventory Size",
                       range=[0, bot10_fam["avg_inventory"].max() * 1.2]),
            yaxis=dict(title=""),
            coloraxis_showscale=False,
            height=340,
            margin=dict(l=10, r=60, t=10, b=10),
        )
        st.plotly_chart(fig_bot, use_container_width=True)

        st.markdown("#### Largest Single Inventories")
        top5_lang = (lang_filtered
                     .nlargest(5, "total_phonemes")
                     [["LanguageName", "family_name",
                        "macroarea", "total_phonemes"]]
                     .rename(columns={
                         "LanguageName"   : "Language",
                         "family_name"    : "Family",
                         "macroarea"      : "Region",
                         "total_phonemes" : "Phonemes",
                     }))
        st.dataframe(top5_lang, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Family Phoneme Profile Heatmap")
    st.caption(
        "Each row = a language family. "
        "Columns show avg consonants, avg vowels, and % tonal. "
        "Colour scale: blue = low, red = high."
    )

    top25_fam = family_filtered.nlargest(25, "avg_inventory")
    heatmap_data = top25_fam.set_index("family_name")[[
        "avg_consonants", "avg_vowels", "pct_with_tones"
    ]].rename(columns={
        "avg_consonants"  : "Avg Consonants",
        "avg_vowels"      : "Avg Vowels",
        "pct_with_tones"  : "% Tonal",
    })

    fig_heat = go.Figure(data=go.Heatmap(
        z     = heatmap_data.values,
        x     = heatmap_data.columns.tolist(),
        y     = heatmap_data.index.tolist(),
        colorscale   = "RdBu",
        reversescale = True,
        text         = np.round(heatmap_data.values, 1),
        texttemplate = "%{text}",
        textfont     = {"size": 10},
        hoverongaps  = False,
    ))
    fig_heat.update_layout(
        xaxis_title = "Feature",
        yaxis_title = "Language Family",
        height      = 620,
        margin      = dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Full Family Table")
    st.dataframe(
        family_filtered.sort_values("avg_inventory", ascending=False)
        .rename(columns={
            "family_name"      : "Family",
            "n_languages"      : "Languages",
            "avg_inventory"    : "Avg Inventory",
            "median_inventory" : "Median Inventory",
            "avg_consonants"   : "Avg Consonants",
            "avg_vowels"       : "Avg Vowels",
            "pct_with_clicks"  : "% with Clicks",
            "pct_with_tones"   : "% Tonal",
        }),
        use_container_width=True,
        hide_index=True,
        height=350,
    )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — WORLD MAPS
# ═════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### World Maps")

    map_df = lang_filtered[
        lang_filtered["latitude"].notna() &
        lang_filtered["longitude"].notna()
    ].copy()

    # ── Map selector ─────────────────────────────────────────────────────────
    map_choice = st.radio(
        "Select Map",
        options=[
            "🌡️  Inventory Size (Plotly)",
            "🔴  Tonal Languages (Folium)",
            "🟤  Click Languages (Folium)",
        ],
        horizontal=True,
    )

    st.markdown("---")

    # ── Map 1: Plotly scatter geo — inventory size ────────────────────────
    if map_choice == "🌡️  Inventory Size (Plotly)":
        st.markdown(
            "#### Phoneme Inventory Size per Language  \n"
            "Each dot = one language, sized and coloured by total phoneme count."
        )
        fig_map = px.scatter_geo(
            map_df,
            lat="latitude",
            lon="longitude",
            color="total_phonemes",
            hover_name="LanguageName",
            hover_data={
                "macroarea"      : True,
                "family_name"    : True,
                "total_phonemes" : True,
                "n_consonants"   : True,
                "n_vowels"       : True,
                "latitude"       : False,
                "longitude"      : False,
            },
            color_continuous_scale="Viridis",
            size="total_phonemes",
            size_max=16,
            projection="natural earth",
            labels={
                "total_phonemes" : "Inventory Size",
                "macroarea"      : "Region",
                "family_name"    : "Family",
            },
        )
        fig_map.update_layout(
            coloraxis_colorbar=dict(title="Inventory"),
            geo=dict(
                showland=True, landcolor="#f5f5f0",
                showocean=True, oceancolor="#cce5f0",
                showcoastlines=True, coastlinecolor="#aaa",
                showframe=False,
            ),
            height=560,
            margin=dict(l=0, r=0, t=0, b=0),
        )
        st.plotly_chart(fig_map, use_container_width=True)

        # Side stats
        st.markdown("**Largest Inventories in Current Filter**")
        st.dataframe(
            map_df.nlargest(8, "total_phonemes")[[
                "LanguageName", "macroarea", "family_name", "total_phonemes"
            ]].rename(columns={
                "LanguageName"   : "Language",
                "macroarea"      : "Region",
                "family_name"    : "Family",
                "total_phonemes" : "Phonemes",
            }),
            hide_index=True, use_container_width=True
        )

    # ── Map 2: Folium — tonal languages ──────────────────────────────────
    elif map_choice == "🔴  Tonal Languages (Folium)":
        st.markdown(
            "#### Tonal Language Distribution  \n"
            "🔴 Red = Tonal language &nbsp;&nbsp; 🔵 Blue = Non-tonal  \n"
            "Africa (64.6%) and North America (15%) dominate."
        )
        tone_map = folium.Map(
            location=[15, 30], zoom_start=2,
            tiles="CartoDB positron"
        )
        for _, row in map_df.iterrows():
            color = "#D7191C" if row["has_tone_segment"] else "#2C7BB6"
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=4,
                color=color,
                fill=True, fill_color=color, fill_opacity=0.7,
                popup=folium.Popup(
                    f"<b>{row['LanguageName']}</b><br>"
                    f"Region: {row['macroarea']}<br>"
                    f"Family: {row['family_name']}<br>"
                    f"Tonal: {'Yes' if row['has_tone_segment'] else 'No'}<br>"
                    f"Inventory: {int(row['total_phonemes'])}",
                    max_width=200
                ),
                tooltip=row["LanguageName"],
            ).add_to(tone_map)

        legend_html = """
        <div style="position:fixed; bottom:30px; left:30px; z-index:9999;
             background:white; padding:10px 15px; border-radius:8px;
             border:1px solid #ccc; font-size:13px;">
          <b>Tone Languages</b><br>
          <span style="color:#D7191C;">&#9679;</span> Tonal<br>
          <span style="color:#2C7BB6;">&#9679;</span> Non-tonal
        </div>"""
        tone_map.get_root().html.add_child(folium.Element(legend_html))
        st_folium(tone_map, width=None, height=520, returned_objects=[])

        # Quick stats
        total_sel  = len(map_df)
        tonal_sel  = map_df["has_tone_segment"].sum()
        st.markdown(
            f"**In current filter:** {tonal_sel:,} tonal "
            f"/ {total_sel:,} total ({tonal_sel/total_sel*100:.1f}%)"
        )

    # ── Map 3: Folium — click languages ───────────────────────────────────
    elif map_choice == "🟤  Click Languages (Folium)":
        st.markdown(
            "#### Click Language Distribution  \n"
            "🔴 Red = Language with click phonemes &nbsp;&nbsp; "
            "⚪ Grey = No clicks  \n"
            "Clicks are almost entirely restricted to southern/eastern Africa."
        )
        click_map = folium.Map(
            location=[0, 25], zoom_start=3,
            tiles="CartoDB positron"
        )
        non_click = map_df[map_df["has_click"] == False]
        click_langs = map_df[map_df["has_click"] == True]

        for _, row in non_click.iterrows():
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=2, color="#cccccc",
                fill=True, fill_color="#cccccc", fill_opacity=0.4,
                tooltip=row["LanguageName"],
            ).add_to(click_map)

        for _, row in click_langs.iterrows():
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=9, color="#8B0000",
                fill=True, fill_color="#D7191C", fill_opacity=0.9,
                popup=folium.Popup(
                    f"<b>{row['LanguageName']}</b><br>"
                    f"Family: {row['family_name']}<br>"
                    f"Inventory: {int(row['total_phonemes'])} phonemes",
                    max_width=200
                ),
                tooltip=f"[CLICK] {row['LanguageName']}",
            ).add_to(click_map)

        legend_html = """
        <div style="position:fixed; bottom:30px; left:30px; z-index:9999;
             background:white; padding:10px 15px; border-radius:8px;
             border:1px solid #ccc; font-size:13px;">
          <b>Click Languages</b><br>
          <span style="color:#D7191C; font-size:16px;">&#9679;</span>
            Has clicks<br>
          <span style="color:#ccc;">&#9679;</span> No clicks
        </div>"""
        click_map.get_root().html.add_child(folium.Element(legend_html))
        st_folium(click_map, width=None, height=520, returned_objects=[])

        st.markdown(
            f"**Languages with click phonemes:** {len(click_langs):,} "
            f"(all in Africa)"
        )
        if len(click_langs) > 0:
            st.dataframe(
                click_langs[["LanguageName", "family_name",
                              "total_phonemes"]]
                .sort_values("total_phonemes", ascending=False)
                .rename(columns={
                    "LanguageName"   : "Language",
                    "family_name"    : "Family",
                    "total_phonemes" : "Inventory",
                }),
                hide_index=True, use_container_width=True
            )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 — CLUSTERING
# ═════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("### Clustering Languages by Phoneme Profile")
    st.caption(
        "K-Means and Hierarchical clustering on a 2,103 x 103 binary "
        "phoneme presence matrix. Do languages naturally group by phoneme "
        "inventory -- and do those groups match geography?"
    )

    OPTIMAL_K = int(sil_df.loc[sil_df["silhouette_score"].idxmax(), "K"])

    # Metric cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Languages clustered", f"{len(lang_cl):,}")
    c2.metric("Features (phonemes)", "103")
    c3.metric("Optimal K", str(OPTIMAL_K))
    c4.metric("Best silhouette score",
              f"{sil_df['silhouette_score'].max():.4f}")

    st.markdown("---")

    # Row 1: Silhouette + Cluster sizes
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Silhouette Score vs K")
        st.caption(
            "Silhouette score measures how naturally separated the clusters "
            "are. Higher = better. The peak determines optimal K."
        )
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Scatter(
            x=sil_df["K"],
            y=sil_df["silhouette_score"],
            mode="lines+markers+text",
            text=[f"{v:.3f}" for v in sil_df["silhouette_score"]],
            textposition="top center",
            line=dict(color="#2C7BB6", width=2.5),
            marker=dict(size=10, color="#2C7BB6",
                        line=dict(color="white", width=2)),
        ))
        fig_sil.add_vline(
            x=OPTIMAL_K, line_dash="dash", line_color="#D7191C",
            annotation_text=f"Optimal K={OPTIMAL_K}",
            annotation_font_color="#D7191C",
        )
        fig_sil.update_layout(
            plot_bgcolor="white",
            xaxis=dict(title="K", gridcolor="#eeeeee", dtick=1),
            yaxis=dict(title="Silhouette Score", gridcolor="#eeeeee"),
            height=340,
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_sil, use_container_width=True)

    with col2:
        st.markdown("#### Cluster Sizes")
        st.caption("Number of languages in each cluster, coloured by dominant region.")
        cluster_sizes = (
            lang_cl.groupby("kmeans_cluster")
            .agg(n_languages=("Glottocode", "count"),
                 dominant_region=("macroarea",
                                  lambda x: x.value_counts().index[0]))
            .reset_index()
        )
        cluster_sizes["Cluster"] = (
            "C" + cluster_sizes["kmeans_cluster"].astype(str)
            + " (" + cluster_sizes["dominant_region"].str[:3] + ")"
        )
        fig_sz = px.bar(
            cluster_sizes,
            x="Cluster",
            y="n_languages",
            color="dominant_region",
            color_discrete_map=REGION_COLORS,
            text="n_languages",
            labels={"n_languages": "Languages",
                    "dominant_region": "Dominant Region"},
        )
        fig_sz.update_traces(textposition="outside")
        fig_sz.update_layout(
            plot_bgcolor="white",
            yaxis=dict(gridcolor="#eeeeee"),
            showlegend=True,
            legend_title="Dominant Region",
            height=340,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_sz, use_container_width=True)

    st.markdown("---")

    # Row 2: World map
    st.markdown("#### World Map -- Languages Coloured by Cluster")
    st.caption(
        "Each dot = one language. Colour = K-Means cluster assignment. "
        "Geographically coherent clusters confirm that phoneme inventories "
        "encode geographic information."
    )

    map_df7 = lang_cl[
        lang_cl["latitude"].notna() &
        lang_cl["longitude"].notna() &
        lang_cl["macroarea"].isin(selected_regions)
    ].copy()
    map_df7["Cluster"] = "Cluster " + map_df7["kmeans_cluster"].astype(str)

    fig_map7 = px.scatter_geo(
        map_df7,
        lat="latitude",
        lon="longitude",
        color="Cluster",
        hover_name="LanguageName",
        hover_data={
            "macroarea"      : True,
            "family_name"    : True,
            "total_phonemes" : True,
            "latitude"       : False,
            "longitude"      : False,
        },
        color_discrete_sequence=px.colors.qualitative.Bold,
        projection="natural earth",
        labels={
            "macroarea"      : "Region",
            "family_name"    : "Family",
            "total_phonemes" : "Inventory",
        },
    )
    fig_map7.update_traces(marker=dict(size=5, opacity=0.8))
    fig_map7.update_layout(
        geo=dict(
            showland=True, landcolor="#f5f5f0",
            showocean=True, oceancolor="#cce5f0",
            showcoastlines=True, coastlinecolor="#aaaaaa",
            showframe=False,
        ),
        height=520,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(title="Cluster"),
    )
    st.plotly_chart(fig_map7, use_container_width=True)

    st.markdown("---")

    # Row 3: Cluster purity
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### Geographic Composition per Cluster")
        ct_data = (
            lang_cl.groupby(["kmeans_cluster", "macroarea"])
            .size()
            .reset_index(name="count")
        )
        ct_total = ct_data.groupby("kmeans_cluster")["count"].transform("sum")
        ct_data["pct"] = (ct_data["count"] / ct_total * 100).round(1)
        ct_data["Cluster"] = "C" + ct_data["kmeans_cluster"].astype(str)
        cluster_order = ["C" + str(i) for i in sorted(ct_data["kmeans_cluster"].unique())]

        fig_comp = px.bar(
            ct_data,
            x="Cluster",
            category_orders={"Cluster": cluster_order},
            y="pct",
            color="macroarea",
            color_discrete_map=REGION_COLORS,
            text="pct",
            labels={"pct": "% of cluster", "macroarea": "Region"},
        )
        fig_comp.update_traces(
            texttemplate="%{text:.0f}%",
            textposition="inside",
        )
        fig_comp.update_layout(
            barmode="stack",
            plot_bgcolor="white",
            yaxis=dict(gridcolor="#eeeeee", title="% of cluster"),
            legend_title="Region",
            height=380,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    with col4:
        st.markdown("#### Key Findings")
        st.info(
            "Cluster 3 = 100% Australia. Australian languages are "
            "phonologically so distinctive (high consonant ratio, no tones, "
            "no clicks) that K-Means isolates them perfectly without being "
            "told their location.\n\n"
            "Cluster 8 = 99.3% Africa. Driven by tonal Niger-Congo and "
            "Nilo-Saharan languages. Tone segments drive this separation.\n\n"
            "Cluster 1 = 92.6% Eurasia. Complex consonant systems "
            "(fricatives, affricates, voiced stops) typical of European "
            "and Central Asian languages."
        )

    # Purity table moved outside columns to avoid React error #185
    st.markdown("#### Cluster Purity Summary")
    purity_rows = []
    for cid in sorted(lang_cl["kmeans_cluster"].unique()):
        sub = lang_cl[lang_cl["kmeans_cluster"] == cid]
        dom = sub["macroarea"].value_counts()
        purity_rows.append({
            "Cluster"         : f"Cluster {cid}",
            "Languages"       : len(sub),
            "Dominant Region" : dom.index[0],
            "Purity %"        : round(dom.iloc[0] / len(sub) * 100, 1),
            "2nd Region"      : dom.index[1] if len(dom) > 1 else "-",
        })
    purity_df = pd.DataFrame(purity_rows)
    st.dataframe(purity_df, hide_index=True,
                 use_container_width=True, height=320)

    st.markdown("---")

    # Row 4: Full table
    st.markdown("#### Full Cluster Assignment Table")
    cluster_filter = st.multiselect(
        "Filter by cluster",
        options=sorted(lang_cl["kmeans_cluster"].unique().tolist()),
        default=sorted(lang_cl["kmeans_cluster"].unique().tolist()),
        format_func=lambda x: f"Cluster {x}",
    )
    display_cl = lang_cl[
        lang_cl["kmeans_cluster"].isin(cluster_filter)
    ][[
        "LanguageName", "macroarea", "family_name",
        "total_phonemes", "kmeans_cluster", "hier_cluster"
    ]].rename(columns={
        "LanguageName"   : "Language",
        "macroarea"      : "Region",
        "family_name"    : "Family",
        "total_phonemes" : "Inventory Size",
        "kmeans_cluster" : "K-Means Cluster",
        "hier_cluster"   : "Hier. Cluster",
    }).sort_values("K-Means Cluster")

    st.dataframe(display_cl, hide_index=True,
                 use_container_width=True, height=380)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#aaa; font-size:12px;'>"
    "PHOIBLE v2.0 · Glottolog metadata · "
    "IIT Guwahati — Data Science & AI · "
    "Built with Streamlit + Plotly + Folium"
    "</p>",
    unsafe_allow_html=True
)
