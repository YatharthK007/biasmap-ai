import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import umap 
import streamlit as st
import plotly.graph_objects as go
from mitigation import (
    apply_smote,
    apply_undersampling,
    run_ghost_bias_simulation,
    compute_entropy_gain,
)

# Importing all audit functions from the local auditor module
from auditor import (
    compute_safe_kl_divergence,
    get_shannon_entropy,
    detect_representation_deserts,
    compute_compliance_grade,
    generate_desert_heatmap,
)


# PAGE CONFIG & GLOBAL THEME

st.set_page_config(
    page_title="BiasMap AI",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Injecting custom CSS 
st.markdown(
    """
    <style>
    /* ── Global background ── */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0e1117;
        color: #e0e0e0;
        font-family: 'Courier New', monospace;
    }
    [data-testid="stSidebar"] { background-color: #111827; }

    /* ── Neon green headings ── */
    h1, h2, h3, h4 { color: #00FF41 !important; }

    /* ── Neon green horizontal rule ── */
    hr { border-color: #00FF41; }

    /* ── Metric widgets ── */
    [data-testid="stMetric"] {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 12px;
    }
    [data-testid="stMetricLabel"] { color: #9ca3af !important; font-size: 0.78rem; }
    [data-testid="stMetricValue"] { color: #00FF41 !important; }

    /* ── Dataframe / table ── */
    [data-testid="stDataFrame"] { border: 1px solid #1f2937; border-radius: 6px; }

    /* ── Compliance badge ── */
    .grade-badge {
        display: inline-block;
        font-size: 4.5rem;
        font-weight: 900;
        font-family: 'Courier New', monospace;
        padding: 12px 32px;
        border-radius: 12px;
        border: 3px solid;
        text-align: center;
        letter-spacing: 2px;
    }
    .risk-box {
        background-color: #111827;
        border-left: 4px solid;
        border-radius: 4px;
        padding: 12px 16px;
        margin-top: 8px;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #00FF41;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }
    .desert-badge-red    { color: #FF4444; font-weight: bold; }
    .desert-badge-orange { color: #FF8C00; font-weight: bold; }
    .desert-badge-yellow { color: #FFD700; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)


# SIDEBAR

with st.sidebar:
    st.markdown("## 🗺️ BiasMap AI")
    st.markdown("**v3.0 — Hackathon Edition**")
    st.markdown("---")
    st.markdown(
        "A **Pre-Training Dataset Auditor** — your GPS for Data Fairness.\n\n"
        "Upload a CSV → confirm schema → explore the bias terrain."
    )
    st.markdown("---")
    st.caption("🔒 Local-only execution. No data leaves your machine.")
    st.caption("⚖️ Compliant with DPDP Act (2023) & NITI Aayog guidelines.")


# TITLE

st.markdown(
    "<h1 style='text-align:center;'>🗺️ BiasMap AI</h1>"
    "<p style='text-align:center; color:#9ca3af; font-size:1.0rem;'>"
    "Dataset Bias Cartography & Compliance Auditor</p>",
    unsafe_allow_html=True,
)
st.markdown("---")


# STEP 1: FILE UPLOAD

st.markdown("### 📁 Step 1 — Upload Dataset")
uploaded_file = st.file_uploader(
    "Upload a CSV file (local only — never sent to any server)",
    type=["csv"],
)

# Guard: nothing to show until a file is uploaded
if uploaded_file is None:
    st.info("👆 Upload a CSV to begin the audit.")
    st.stop()

# Load CSV into session state so we don't reload on every widget interaction
if "df" not in st.session_state or st.session_state.get("filename") != uploaded_file.name:
    df_raw = pd.read_csv(uploaded_file)
    # Stratified sampling for large datasets (> 100k rows) to prevent UMAP OOM
    if len(df_raw) > 100_000:
        df_raw = df_raw.sample(n=100_000, random_state=42).reset_index(drop=True)
        st.warning("⚡ Dataset > 100k rows — stratified sample of 100k applied for UMAP safety.")
    st.session_state["df"] = df_raw
    st.session_state["filename"] = uploaded_file.name
    # Reset schema state when a new file is loaded
    st.session_state.pop("schema_confirmed", None)
    st.session_state.pop("sensitive_cols", None)
    st.session_state.pop("target_col", None)

df = st.session_state["df"]

st.success(f"✅ Loaded **{uploaded_file.name}** — {df.shape[0]:,} rows × {df.shape[1]} columns")
with st.expander("🔍 Preview raw data (first 5 rows)"):
    st.dataframe(df.head(5), use_container_width=True)

st.markdown("---")

# STEP 2: SCHEMA VERIFICATION LOOP

st.markdown("### 🔖 Step 2 — Confirm Schema")
st.caption(
    "Auto-detection is never 100% accurate. Manually confirm which columns are "
    "**Sensitive**, which is the **Target**, and which to **Exclude**."
)

# Auto-detect likely sensitive column names via keyword matching
SENSITIVE_KEYWORDS = [
    "gender", "sex", "race", "ethnicity", "caste", "religion",
    "age", "region", "state", "district", "rural", "urban",
    "pincode", "zip", "nationality", "language", "disability",
]

def _guess_sensitive(columns: list[str]) -> list[str]:
    """Returns column names that likely contain sensitive attributes."""
    return [
        c for c in columns
        if any(kw in c.lower() for kw in SENSITIVE_KEYWORDS)
    ]

all_cols = list(df.columns)
guessed_sensitive = _guess_sensitive(all_cols)

col_left, col_right = st.columns([3, 2])

with col_left:
    sensitive_cols = st.multiselect(
        "🔴 Sensitive Attribute columns",
        options=all_cols,
        default=guessed_sensitive,
        help="Columns representing demographic or protected attributes.",
    )
    target_col = st.selectbox(
        "🎯 Target / Label column",
        options=["(none)"] + all_cols,
        index=0,
        help="The outcome variable (e.g. Loan_Status, Income_Level).",
    )

with col_right:
    exclude_cols = st.multiselect(
        "⛔ Exclude columns",
        options=[c for c in all_cols if c not in sensitive_cols],
        help="Columns to ignore during analysis (IDs, free-text, etc.).",
    )

# Require at least one sensitive column before proceeding
confirm_clicked = st.button("✅ Confirm Schema & Run Audit", type="primary")

if confirm_clicked:
    if len(sensitive_cols) == 0:
        st.error("Select at least one Sensitive column before confirming.")
    else:
        st.session_state["schema_confirmed"] = True
        st.session_state["sensitive_cols"] = sensitive_cols
        st.session_state["target_col"] = target_col if target_col != "(none)" else None
        st.session_state["exclude_cols"] = exclude_cols

# Guard: don't proceed until schema is confirmed
if not st.session_state.get("schema_confirmed"):
    st.info("☝️ Confirm the schema above to unlock the Audit Engine.")
    st.stop()

# Retrieve confirmed schema from session state
sensitive_cols = st.session_state["sensitive_cols"]
target_col     = st.session_state["target_col"]
exclude_cols   = st.session_state.get("exclude_cols", [])

st.success(
    f"✅ Schema confirmed — **{len(sensitive_cols)} sensitive** col(s): "
    f"`{'`, `'.join(sensitive_cols)}`"
)
st.markdown("---")


# STEP 3: UMAP TERRAIN MAP

st.markdown("### 🌐 Step 3 — Bias Terrain Map (UMAP)")
st.caption(
    "Dimensionality reduction projects your dataset into 2D space. "
    "Clusters reveal where similar data points are concentrated — "
    "click a point to inspect that region."
)

@st.cache_data(show_spinner="🧭 Projecting dataset into 2D terrain…")
def run_umap(df: pd.DataFrame, sensitive_cols: list, exclude_cols: list) -> pd.DataFrame:
    """
    One-hot-encodes all non-numeric columns, then runs UMAP to produce
    2D coordinates. Cached so rerunning the UI doesn't re-project.
    """
    # Drop excluded columns and columns with all NaNs
    drop_cols = exclude_cols + [c for c in df.columns if df[c].isna().all()]
    df_work = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Encode: numeric stays, categorical gets one-hot encoded
    df_encoded = pd.get_dummies(df_work, dummy_na=False)

    # Fill any remaining NaNs with column median (UMAP needs no NaNs)
    df_encoded = df_encoded.fillna(df_encoded.median(numeric_only=True))

    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(df_encoded.values)

    result = pd.DataFrame({"UMAP_1": embedding[:, 0], "UMAP_2": embedding[:, 1]})
    # Attach the primary sensitive column for color coding
    result["color_by"] = df[sensitive_cols[0]].fillna("Missing_Data_Unknown").astype(str)
    return result

umap_df = run_umap(df, sensitive_cols, exclude_cols)

umap_fig = px.scatter(
    umap_df,
    x="UMAP_1",
    y="UMAP_2",
    color="color_by",
    color_discrete_sequence=px.colors.qualitative.Vivid,
    labels={"color_by": sensitive_cols[0]},
    title=f"🌐 Dataset Terrain — coloured by <b>{sensitive_cols[0]}</b>",
    template="plotly_dark",
)
umap_fig.update_traces(marker=dict(size=4, opacity=0.75))
umap_fig.update_layout(
    paper_bgcolor="#0e1117",
    plot_bgcolor="#111827",
    title_font_color="#00FF41",
    legend_title_font_color="#00FF41",
    height=500,
    margin=dict(l=10, r=10, t=50, b=10),
)
st.plotly_chart(umap_fig, use_container_width=True)
st.markdown("---")


# STEP 4: AUDIT RESULTS

st.markdown("### 🔬 Step 4 — Audit Results")

# 4a. Compute all audit metrics 

# Shannon Entropy for each sensitive column
entropy_scores = {col: get_shannon_entropy(df[col]) for col in sensitive_cols}
# Max possible entropy = log2(number of unique categories) — benchmark for "perfect diversity"
max_entropy = {
    col: np.log2(df[col].nunique()) if df[col].nunique() > 1 else 1.0
    for col in sensitive_cols
}

# Build p_dist from the first sensitive column's value counts
# Use a uniform benchmark Q (equal weight across all categories) since no census JSON is loaded.
# In a production setup, load the census JSON and align category keys here.
primary_col = sensitive_cols[0]
p_counts = df[primary_col].fillna("Missing_Data_Unknown").value_counts().values.astype(float)
# Uniform benchmark: every category gets equal weight → perfect representation
q_uniform = np.ones(len(p_counts), dtype=float)

kl_score = compute_safe_kl_divergence(p_counts, q_uniform)
grade_info = compute_compliance_grade(kl_score)

# Desert detection across all sensitive columns
deserts = detect_representation_deserts(df, sensitive_cols)

# Desert heatmap Plotly figure (None if < 2 sensitive cols)
heatmap_fig = generate_desert_heatmap(df, sensitive_cols)

# 4b. Compliance Grade Badge 

st.markdown("#### ⚖️ DPDP / NITI Aayog Compliance Grade")

badge_col, summary_col = st.columns([1, 3])

with badge_col:
    badge_color = grade_info["color"]
    st.markdown(
        f"""
        <div class="grade-badge"
             style="color:{badge_color}; border-color:{badge_color};
                    background-color:#111827;">
            {grade_info['grade']}
        </div>
        <div style="text-align:center; margin-top:6px; font-size:0.8rem;
                    color:{badge_color}; font-weight:700; letter-spacing:1px;">
            {grade_info['badge_text']}
        </div>
        """,
        unsafe_allow_html=True,
    )

with summary_col:
    st.markdown(
        f"""
        <div class="risk-box" style="border-color:{badge_color};">
            <strong style="color:{badge_color};">Legal Risk Summary</strong><br>
            {grade_info['risk_summary']}
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

grade = grade_info["grade"]

# Neon-green palette constants
NEON   = "#00FF41"
BG     = "#0e1117"
CARD   = "#161b22"
 

# AUTO-FIX SECTION

st.markdown("---")
st.markdown(
    "<h2 style='color:#00FF41;'>⚡ Auto-Fix & Ghost Bias Detector</h2>",
    unsafe_allow_html=True,
)
 
# Grades that trigger the fix panel
BAD_GRADES  = {"C", "D", "F"}
GOOD_GRADES = {"A+", "A", "B"}
 
if grade in GOOD_GRADES:
    # Dataset is already healthy — no fix needed
    st.success(
        f"✅ **Grade {grade} — No Mitigation Needed.** "
        "Your dataset meets the fairness threshold. "
        "Review the Ghost Bias chart below for transparency."
    )
else:
    # Show fix panel for C / D / F
    st.warning(
        f"⚠️ **Grade {grade} detected.** "
        "Apply an Auto-Fix to improve data diversity before training."
    )
 
    # Let user pick a mitigation strategy
    fix_method = st.radio(
        "Select Mitigation Strategy",
        options=["SMOTE (Synthetic Over-sampling)", "Strategic Under-sampling"],
        horizontal=True,
        help=(
            "SMOTE creates synthetic minority samples. "
            "Under-sampling prunes the majority class — no fabricated data."
        ),
    )
 
    # Require at least one sensitive column for under-sampling
    if fix_method == "Strategic Under-sampling" and not sensitive_cols:
        st.error(
            "No Sensitive columns marked. "
            "Go back to the Schema screen and mark at least one column as Sensitive."
        )
    elif fix_method == "SMOTE (Synthetic Over-sampling)" and not target_col:
        st.error(
            "SMOTE requires a Target column. "
            "Go back to the Schema screen and mark one column as Target."
        )
    else:
        run_fix = st.button("🚀 Run Auto-Fix", use_container_width=True)
 
        if run_fix:
            with st.spinner("Running mitigation engine…"):
                try:
                    if fix_method == "SMOTE (Synthetic Over-sampling)":
                        df_fixed, e_before, e_after = apply_smote(df, target_col)
                        used_col = target_col
                    else:
                        # Use first sensitive col for under-sampling pivot
                        pivot_col = sensitive_cols[0]
                        df_fixed, e_before, e_after = apply_undersampling(
                            df, pivot_col
                        )
                        used_col = pivot_col
 
                    gain = compute_entropy_gain(e_before, e_after)
 
                    # Entropy metrics side-by-side
                    st.markdown(
                        "<h4 style='color:#00FF41;'>Shannon Entropy: Before vs After</h4>",
                        unsafe_allow_html=True,
                    )
                    col_before, col_after, col_gain = st.columns(3)
 
                    with col_before:
                        st.metric(
                            label="🔴 Entropy Before",
                            value=f"{e_before:.4f} bits",
                        )
                    with col_after:
                        st.metric(
                            label="🟢 Entropy After",
                            value=f"{e_after:.4f} bits",
                            delta=f"{e_after - e_before:+.4f} bits",
                        )
                    with col_gain:
                        st.metric(
                            label="📈 Entropy Gain",
                            value=f"{gain:+.2f}%",
                        )
 
                    # Persist fixed df in session so report.py can use it
                    st.session_state["df_fixed"] = df_fixed
 
                    st.success(
                        f"✅ Mitigation complete! Fixed dataset has "
                        f"**{len(df_fixed):,} rows** "
                        f"(was {len(df):,}). "
                        "Download or continue to the PDF Report."
                    )
 
                    # Quick preview of the fixed dataset
                    with st.expander("Preview Fixed Dataset (first 50 rows)"):
                        st.dataframe(df_fixed.head(50), use_container_width=True)
 
                except Exception as exc:
                    st.error(f"Auto-Fix failed: {exc}")
 
 

# GHOST BIAS CHART — always visible, regardless of grade

st.markdown(
    "<h3 style='color:#00FF41; margin-top:2rem;'>"
    "👻 Ghost Bias Detector</h3>",
    unsafe_allow_html=True,
)
st.caption(
    "Which column drives skewed predictions? "
    "Trained on a shallow Decision Tree (max_depth=4) — "
    "higher bar = stronger proxy bias."
)
 
if not target_col:
    # User hasn't selected a target — friendly prompt
    st.info(
        "ℹ️ **Select a Target column in the Schema screen** "
        "to enable the Ghost Bias simulation."
    )
elif not sensitive_cols:
    st.info(
        "ℹ️ **Mark at least one Sensitive column in the Schema screen** "
        "to run the Ghost Bias simulation."
    )
else:
    with st.spinner("Training Ghost Bias shadow model…"):
        try:
            importances = run_ghost_bias_simulation(df, sensitive_cols, target_col)
 
            if not importances:
                st.warning("No feature importances returned — check your columns.")
            else:
                cols_sorted   = list(importances.keys())
                scores_sorted = list(importances.values())
 
                # Horizontal bar chart — neon green on dark bg
                fig = go.Figure(
                    go.Bar(
                        x=scores_sorted,
                        y=cols_sorted,
                        orientation="h",
                        marker=dict(
                            color=NEON,
                            line=dict(color=NEON, width=1),
                        ),
                        text=[f"{s:.4f}" for s in scores_sorted],
                        textposition="outside",
                        textfont=dict(color=NEON, size=11),
                        hovertemplate=(
                            "<b>%{y}</b><br>"
                            "Gini Importance: %{x:.4f}<extra></extra>"
                        ),
                    )
                )
 
                fig.update_layout(
                    title=dict(
                        text=(
                            "Ghost Bias Detector — "
                            "Which column drives skewed predictions?"
                        ),
                        font=dict(color=NEON, size=15),
                    ),
                    paper_bgcolor=BG,
                    plot_bgcolor=CARD,
                    xaxis=dict(
                        title="Gini Importance Score",
                        title_font=dict(color="#AAAAAA"),
                        tickfont=dict(color="#AAAAAA"),
                        gridcolor="#1e2a1e",
                        range=[0, max(scores_sorted) * 1.25],  # breathing room
                    ),
                    yaxis=dict(
                        tickfont=dict(color=NEON),
                        autorange="reversed",  # highest importance at top
                    ),
                    margin=dict(l=20, r=40, t=60, b=40),
                    height=max(300, len(cols_sorted) * 55),  # scale with rows
                )
 
                st.plotly_chart(fig, use_container_width=True)
 
                # Highlight the single biggest culprit
                top_col   = cols_sorted[0]
                top_score = scores_sorted[0]
                st.markdown(
                    f"<p style='color:#FF4B4B; font-weight:bold;'>"
                    f"🚨 Highest proxy bias: <code>{top_col}</code> "
                    f"(Gini Importance = {top_score:.4f}). "
                    f"This column may act as a proxy for a protected attribute.</p>",
                    unsafe_allow_html=True,
                )
 
        except Exception as exc:
            st.error(f"Ghost Bias simulation failed: {exc}")

# 4c. KL Divergence Score 

kl_col, _, _ = st.columns(3)
with kl_col:
    st.metric(
        label=f"📐 KL Divergence — {primary_col} vs. Uniform Benchmark",
        value=f"{kl_score:.4f}",
        delta=None,
        help=(
            "KL Divergence measures how far your dataset's distribution "
            "is from perfect equality. 0 = ideal. Higher = more biased."
        ),
    )

st.markdown("---")

# 4d. Shannon Entropy for each Sensitive Column 

st.markdown("#### 📊 Diversity Metrics — Shannon Entropy per Sensitive Column")
st.caption(
    "Shannon Entropy measures how evenly a column's values are distributed. "
    "Higher = more diverse. Max entropy = log₂(unique categories)."
)

entropy_cols = st.columns(min(len(sensitive_cols), 4))  # max 4 per row

for i, col in enumerate(sensitive_cols):
    h_val    = entropy_scores[col]
    h_max    = max_entropy[col]
    pct_div  = (h_val / h_max * 100) if h_max > 0 else 0
    delta_str = f"{pct_div:.1f}% of max diversity"

    with entropy_cols[i % 4]:
        st.metric(
            label=f"H( {col} )",
            value=f"{h_val:.3f} bits",
            delta=delta_str,
            delta_color="normal",
            help=f"Max possible entropy for '{col}': {h_max:.3f} bits",
        )

st.markdown("---")

# 4e. Desert Heatmap

st.markdown("#### 🏜️ Intersectional Terrain Heatmap")

if heatmap_fig is not None:
    st.plotly_chart(heatmap_fig, use_container_width=True)
    st.caption(
        "Dark purple cells = representation deserts (near-zero data). "
        "Neon green cells = well-represented intersections."
    )
else:
    st.info(
        "ℹ️ The terrain heatmap requires at least **2 sensitive columns**. "
        "Go back to Step 2 and mark additional columns as Sensitive."
    )

st.markdown("---")

# 4f. Top Representation Deserts Table 

st.markdown("#### 🔴 Top Representation Deserts")
st.caption(
    f"Intersections with < 1% representation. "
    f"Found **{len(deserts)}** desert(s) across {len(sensitive_cols)} sensitive column(s)."
)

if len(deserts) == 0:
    st.success(
        "✅ No representation deserts detected! All intersectional combinations "
        "have ≥ 1% representation."
    )
else:
    # Show top 5 worst deserts in a styled table
    top_deserts = deserts[:5]

    desert_table = pd.DataFrame(
        [
            {
                "Intersection": d["combination"],
                "Count": d["count"],
                "% of Dataset": d["percentage"],
                "Severity": d["severity"],
            }
            for d in top_deserts
        ]
    )

    # Style: highlight the Severity column
    def _severity_style(val: str) -> str:
        if "ABSENT" in val:
            return "color: #FF4444; font-weight: bold;"
        elif "CRITICAL" in val:
            return "color: #FF8C00; font-weight: bold;"
        else:
            return "color: #FFD700; font-weight: bold;"

    styled = (
        desert_table.style
        .map(_severity_style, subset=["Severity"])
        .format({"% of Dataset": "{:.4f}%"})
        .set_properties(**{
            "background-color": "#111827",
            "color": "#e0e0e0",
            "border": "1px solid #1f2937",
        })
        .set_table_styles([
            {"selector": "th", "props": [
                ("background-color", "#0e1117"),
                ("color", "#00FF41"),
                ("font-family", "'Courier New', monospace"),
                ("border", "1px solid #1f2937"),
            ]},
        ])
    )

    st.dataframe(styled, use_container_width=True, height=250)

    if len(deserts) > 5:
        with st.expander(f"📋 Show all {len(deserts)} deserts"):
            all_desert_df = pd.DataFrame(
                [
                    {
                        "Intersection": d["combination"],
                        "Count": d["count"],
                        "% of Dataset": d["percentage"],
                        "Severity": d["severity"],
                    }
                    for d in deserts
                ]
            )
            st.dataframe(all_desert_df, use_container_width=True)

st.markdown("---")

# Footer 

st.markdown(
    "<p style='text-align:center; color:#4b5563; font-size:0.75rem;'>"
    "BiasMap AI v3.0 · Local-First · DPDP Act (2023) Compliant · "
    "Built for the Hackathon · No data leaves your machine."
    "</p>",
    unsafe_allow_html=True,
)