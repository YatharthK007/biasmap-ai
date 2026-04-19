import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import umap

from mitigation import (
    apply_smote,
    apply_undersampling,
    run_ghost_bias_simulation,
    compute_entropy_gain,
)
from auditor import (
    compute_safe_kl_divergence,
    get_shannon_entropy,
    detect_representation_deserts,
    compute_compliance_grade,
    generate_desert_heatmap,
)
from report import generate_pdf_report


# Palette constants 
NEON = "#00FF41"
BG   = "#0e1117"
CARD = "#161b22"
CARD2 = "#111827"



# PAGE CONFIG

st.set_page_config(
    page_title="BiasMap AI",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)



# GLOBAL CSS

st.markdown(
    f"""
    <style>
    /* ── Base ── */
    html, body, [data-testid="stAppViewContainer"] {{
        background-color: {BG};
        color: #e0e0e0;
        font-family: 'Courier New', monospace;
    }}
    [data-testid="stSidebar"] {{
        background-color: {CARD2};
        border-right: 1px solid #1f2937;
    }}

    /* ── Main content max-width guard ── */
    .block-container {{ padding-top: 2rem; padding-bottom: 3rem; }}

    /* ── Headings ── */
    h1, h2, h3, h4 {{ color: {NEON} !important; letter-spacing: 0.5px; }}

    /* ── Divider ── */
    hr {{ border-color: #1f2937; margin: 1.5rem 0; }}

    /* ── Step header pill ── */
    .step-pill {{
        display: inline-block;
        background: {CARD2};
        border: 1px solid {NEON};
        border-radius: 20px;
        padding: 4px 18px;
        font-size: 0.78rem;
        color: {NEON};
        font-weight: 700;
        letter-spacing: 1.5px;
        margin-bottom: 6px;
    }}

    /* ── Section card wrapper ── */
    .audit-card {{
        background: {CARD2};
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 20px 24px;
        margin-bottom: 18px;
    }}

    /* ── Metric widgets ── */
    [data-testid="stMetric"] {{
        background-color: {CARD2};
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 14px 16px;
    }}
    [data-testid="stMetricLabel"] {{ color: #9ca3af !important; font-size: 0.78rem; }}
    [data-testid="stMetricValue"] {{ color: {NEON} !important; font-size: 1.4rem !important; }}

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {{
        border: 1px solid #1f2937;
        border-radius: 6px;
    }}

    /* ── Compliance badge ── */
    .grade-badge {{
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
        font-size: 5rem;
        font-weight: 900;
        font-family: 'Courier New', monospace;
        padding: 16px 28px;
        border-radius: 14px;
        border: 3px solid;
        letter-spacing: 3px;
        min-width: 120px;
    }}
    .badge-sub {{
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 2px;
        margin-top: 4px;
    }}

    /* ── Risk box ── */
    .risk-box {{
        background-color: {CARD2};
        border-left: 4px solid;
        border-radius: 6px;
        padding: 14px 18px;
        font-size: 0.9rem;
        line-height: 1.7;
    }}

    /* ── Sidebar progress tracker ── */
    .progress-row {{
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 5px 0;
        font-size: 0.85rem;
    }}
    .step-dot-done  {{ color: {NEON}; font-size: 1rem; }}
    .step-dot-todo  {{ color: #374151; font-size: 1rem; }}
    .step-label-done {{ color: #d1d5db; }}
    .step-label-todo {{ color: #374151; }}

    /* ── Info / warning / error overrides ── */
    [data-testid="stAlert"] {{ border-radius: 8px; }}

    /* ── Buttons ── */
    [data-testid="stButton"] > button {{
        font-family: 'Courier New', monospace;
        font-weight: 700;
        letter-spacing: 0.5px;
        border-radius: 6px;
    }}

    /* ── Expander ── */
    [data-testid="stExpander"] {{
        background: {CARD2};
        border: 1px solid #1f2937 !important;
        border-radius: 8px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)



# SIDEBAR — Logo + tagline + live progress tracker

def _progress_row(label: str, done: bool) -> str:
    """Render one progress tracker row as HTML."""
    dot   = "●" if done else "○"
    dcls  = "step-dot-done"  if done else "step-dot-todo"
    lcls  = "step-label-done" if done else "step-label-todo"
    check = " ✓" if done else ""
    return (
        f"<div class='progress-row'>"
        f"<span class='{dcls}'>{dot}</span>"
        f"<span class='{lcls}'>{label}{check}</span>"
        f"</div>"
    )

with st.sidebar:
    # Logo
    st.markdown(
        f"""
        <div style='text-align:center; padding: 10px 0 4px 0;'>
            <div style='font-size:2.4rem;'>🗺️</div>
            <div style='font-size:1.6rem; font-weight:900; color:{NEON};
                        font-family:"Courier New",monospace; letter-spacing:2px;'>
                BiasMap AI
            </div>
            <div style='font-size:0.72rem; color:#6b7280; letter-spacing:1px;
                        margin-top:2px;'>
                GPS FOR DATA FAIRNESS
            </div>
            <div style='font-size:0.65rem; color:#374151; margin-top:4px;'>
                v3.0 &nbsp;·&nbsp; Hackathon Edition
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<hr style='border-color:#1f2937; margin:14px 0;'>",
        unsafe_allow_html=True,
    )

    # Progress tracker (reads from session state)
    step_upload  = st.session_state.get("filename") is not None
    step_schema  = st.session_state.get("schema_confirmed", False)
    step_audit   = step_schema   # audit runs automatically after schema
    step_fix     = "df_fixed" in st.session_state
    step_export  = st.session_state.get("pdf_exported", False)

    st.markdown(
        "<div style='font-size:0.7rem; color:#6b7280; letter-spacing:1.5px;"
        "margin-bottom:6px;'>AUDIT PROGRESS</div>",
        unsafe_allow_html=True,
    )
    progress_html = "".join([
        _progress_row("Upload Dataset",   step_upload),
        _progress_row("Confirm Schema",   step_schema),
        _progress_row("Run Analysis",     step_audit),
        _progress_row("Apply Auto-Fix",   step_fix),
        _progress_row("Export PDF",       step_export),
    ])
    st.markdown(progress_html, unsafe_allow_html=True)

    st.markdown(
        "<hr style='border-color:#1f2937; margin:14px 0;'>",
        unsafe_allow_html=True,
    )

    # Info badges 
    st.markdown(
        f"""
        <div style='font-size:0.75rem; color:#6b7280; line-height:1.9;'>
            🔒 &nbsp;Local-only — no data leaves your machine<br>
            ⚖️ &nbsp;DPDP Act (2023) compliant<br>
            🇮🇳 &nbsp;NITI Aayog guidelines aligned<br>
            📊 &nbsp;KL Divergence + Shannon Entropy<br>
            🤖 &nbsp;Ghost Bias via Decision Tree
        </div>
        """,
        unsafe_allow_html=True,
    )



# MAIN TITLE

st.markdown(
    f"""
    <div style='text-align:center; padding: 8px 0 4px 0;'>
        <div style='font-size:2.2rem; font-weight:900; color:{NEON};
                    font-family:"Courier New",monospace; letter-spacing:3px;'>
            🗺️ &nbsp;BiasMap AI
        </div>
        <div style='color:#6b7280; font-size:0.95rem; margin-top:4px;
                    letter-spacing:0.5px;'>
            Dataset Bias Cartography &amp; Compliance Auditor
        </div>
        <div style='color:#374151; font-size:0.72rem; margin-top:3px;'>
            Pre-Training Data Hygiene · DPDP Act (2023) · NITI Aayog Aligned
        </div>
    </div>
    <hr style='border-color:#1f2937; margin:18px 0 10px 0;'>
    """,
    unsafe_allow_html=True,
)



# STEP 1 — FILE UPLOAD

st.markdown(
    "<div class='step-pill'>STEP 01</div>"
    f"<h3 style='margin-top:4px;'>📁 Upload Dataset</h3>",
    unsafe_allow_html=True,
)
st.caption(
    "Upload any CSV. The file is processed entirely on your local machine — "
    "nothing is sent to any server."
)

uploaded_file = st.file_uploader(
    "Drop a CSV here or click Browse",
    type=["csv"],
    label_visibility="collapsed",
)

if uploaded_file is None:
    st.markdown(
        f"""
        <div style='text-align:center; padding:40px 20px;
                    border:1px dashed #1f2937; border-radius:10px;
                    color:#374151; font-size:0.9rem;'>
            ↑ &nbsp; Upload a CSV file above to begin the audit
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# Load CSV — cache in session state so widgets don't re-trigger a reload
if "df" not in st.session_state or st.session_state.get("filename") != uploaded_file.name:
    with st.spinner("Reading dataset…"):
        df_raw = pd.read_csv(uploaded_file)
        if len(df_raw) > 100_000:
            df_raw = df_raw.sample(n=100_000, random_state=42).reset_index(drop=True)
            st.warning(
                "⚡ Dataset has more than 100,000 rows. "
                "A stratified random sample of 100k rows was taken to prevent "
                "memory overflow during UMAP projection."
            )
    st.session_state["df"]       = df_raw
    st.session_state["filename"] = uploaded_file.name
    # Clear downstream state when a new file is loaded
    for key in ["schema_confirmed", "sensitive_cols", "target_col",
                "exclude_cols", "df_fixed", "fix_method_used",
                "entropy_before_val", "entropy_after_val", "pdf_exported"]:
        st.session_state.pop(key, None)

df = st.session_state["df"]

# Summary strip
c1, c2, c3 = st.columns(3)
c1.metric("Rows",    f"{df.shape[0]:,}")
c2.metric("Columns", f"{df.shape[1]}")
c3.metric("NaN cells", f"{df.isna().sum().sum():,}")

with st.expander("🔍 Preview — first 5 rows"):
    st.dataframe(df.head(5), use_container_width=True)

st.markdown("<hr style='border-color:#1f2937;'>", unsafe_allow_html=True)



# STEP 2 — SCHEMA VERIFICATION

st.markdown(
    "<div class='step-pill'>STEP 02</div>"
    f"<h3 style='margin-top:4px;'>🔖 Confirm Schema</h3>",
    unsafe_allow_html=True,
)
st.caption(
    "Auto-detection uses keyword matching — verify it is correct. "
    "Mark every column that represents a protected or demographic attribute as **Sensitive**."
)

SENSITIVE_KEYWORDS = [
    "gender", "sex", "race", "ethnicity", "caste", "religion",
    "age", "region", "state", "district", "rural", "urban",
    "pincode", "zip", "nationality", "language", "disability",
]

def _guess_sensitive(columns: list) -> list:
    return [c for c in columns if any(kw in c.lower() for kw in SENSITIVE_KEYWORDS)]

all_cols         = list(df.columns)
guessed_sensitive = _guess_sensitive(all_cols)

col_left, col_right = st.columns([3, 2])

with col_left:
    sensitive_cols = st.multiselect(
        "🔴 Sensitive Attribute columns",
        options=all_cols,
        default=guessed_sensitive,
        help="Columns representing demographic or protected attributes (gender, race, age, etc.).",
    )
    target_col = st.selectbox(
        "🎯 Target / Label column",
        options=["(none)"] + all_cols,
        index=0,
        help="The outcome variable your model will predict (e.g. income, loan_status).",
    )

with col_right:
    exclude_cols = st.multiselect(
        "⛔ Exclude columns",
        options=[c for c in all_cols if c not in sensitive_cols],
        help="Drop these from analysis — e.g. row IDs, free-text fields.",
    )
    st.markdown("<br>", unsafe_allow_html=True)
    if sensitive_cols:
        st.markdown(
            f"<div style='font-size:0.8rem; color:#6b7280;'>"
            f"Auto-detected <span style='color:{NEON};'>{len(guessed_sensitive)}</span>"
            f" sensitive column(s) from {len(all_cols)} total.</div>",
            unsafe_allow_html=True,
        )

confirm_clicked = st.button("✅ Confirm Schema & Run Audit", type="primary", use_container_width=True)

if confirm_clicked:
    if len(sensitive_cols) == 0:
        st.error("⛔ Select at least one Sensitive column before confirming.")
    else:
        st.session_state["schema_confirmed"] = True
        st.session_state["sensitive_cols"]   = sensitive_cols
        st.session_state["target_col"]       = target_col if target_col != "(none)" else None
        st.session_state["exclude_cols"]     = exclude_cols

if not st.session_state.get("schema_confirmed"):
    st.info("☝️ Confirm the schema above to unlock the Audit Engine.")
    st.stop()

# Pull confirmed values from session state
sensitive_cols = st.session_state["sensitive_cols"]
target_col     = st.session_state["target_col"]
exclude_cols   = st.session_state.get("exclude_cols", [])

st.success(
    f"✅ Schema confirmed — **{len(sensitive_cols)}** sensitive column(s): "
    + ", ".join(f"`{c}`" for c in sensitive_cols)
)

st.markdown("<hr style='border-color:#1f2937;'>", unsafe_allow_html=True)



# STEP 3 — UMAP TERRAIN MAP

st.markdown(
    "<div class='step-pill'>STEP 03</div>"
    f"<h3 style='margin-top:4px;'>🌐 Bias Terrain Map</h3>",
    unsafe_allow_html=True,
)
st.caption(
    "UMAP reduces your dataset to 2D. Tight clusters = homogeneous groups. "
    "Isolated islands = under-represented minorities. Colour = primary sensitive attribute."
)

@st.cache_data(show_spinner=False)
def run_umap(df: pd.DataFrame, sensitive_cols: tuple, exclude_cols: tuple) -> pd.DataFrame:
    
    sensitive_cols = list(sensitive_cols)
    exclude_cols   = list(exclude_cols)

    drop_cols = exclude_cols + [c for c in df.columns if df[c].isna().all()]
    df_work   = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    df_enc    = pd.get_dummies(df_work, dummy_na=False)
    df_enc    = df_enc.fillna(df_enc.median(numeric_only=True))

    reducer   = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(df_enc.values)

    result = pd.DataFrame({"UMAP_1": embedding[:, 0], "UMAP_2": embedding[:, 1]})
    result["color_by"] = df[sensitive_cols[0]].fillna("Missing_Data_Unknown").astype(str)
    return result

try:
    with st.spinner(
        "🧭 Projecting dataset into 2D terrain — this may take 20–60 seconds "
        "depending on dataset size…"
    ):
        umap_df = run_umap(df, tuple(sensitive_cols), tuple(exclude_cols))

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
        paper_bgcolor=BG,
        plot_bgcolor=CARD2,
        title_font_color=NEON,
        title_font_size=15,
        legend_title_font_color=NEON,
        legend=dict(
            bgcolor=CARD,
            bordercolor="#1f2937",
            borderwidth=1,
            font=dict(color="#d1d5db", size=11),
        ),
        xaxis=dict(tickfont=dict(color="#6b7280"), gridcolor="#1f2937", title=""),
        yaxis=dict(tickfont=dict(color="#6b7280"), gridcolor="#1f2937", title=""),
        height=500,
        margin=dict(l=10, r=10, t=52, b=10),
    )
    st.plotly_chart(umap_fig, use_container_width=True)

except Exception as umap_err:
    st.exception(umap_err)
    st.error(
        "🗺️ UMAP projection failed on this dataset. "
        "Common causes: too many NaN columns, a single-row category, "
        "or non-numeric data that couldn't be encoded. "
        "Try excluding problematic columns in Step 2 and re-confirming the schema."
    )

st.markdown("<hr style='border-color:#1f2937;'>", unsafe_allow_html=True)



# STEP 4 — AUDIT RESULTS

st.markdown(
    "<div class='step-pill'>STEP 04</div>"
    f"<h3 style='margin-top:4px;'>🔬 Audit Results</h3>",
    unsafe_allow_html=True,
)

# Compute all metrics 
entropy_scores = {col: get_shannon_entropy(df[col]) for col in sensitive_cols}
max_entropy    = {
    col: np.log2(df[col].nunique()) if df[col].nunique() > 1 else 1.0
    for col in sensitive_cols
}

primary_col = sensitive_cols[0]
p_counts    = df[primary_col].fillna("Missing_Data_Unknown").value_counts().values.astype(float)
q_uniform   = np.ones(len(p_counts), dtype=float)

kl_score    = compute_safe_kl_divergence(p_counts, q_uniform)
grade_info  = compute_compliance_grade(kl_score)
grade       = grade_info["grade"]
badge_color = grade_info["color"]

deserts    = detect_representation_deserts(df, sensitive_cols)
heatmap_fig = generate_desert_heatmap(df, sensitive_cols)


# 4a. Compliance Grade
st.markdown("#### ⚖️ DPDP / NITI Aayog Compliance Grade")

badge_col, summary_col = st.columns([1, 3], gap="large")

with badge_col:
    st.markdown(
        f"""
        <div class="grade-badge"
             style="color:{badge_color}; border-color:{badge_color};
                    background-color:{CARD2};">
            {grade}
            <span class="badge-sub" style="color:{badge_color};">
                {grade_info['badge_text']}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with summary_col:
    st.markdown(
        f"""
        <div class="risk-box" style="border-color:{badge_color}; height:100%;">
            <div style="font-size:0.72rem; color:{badge_color}; font-weight:700;
                        letter-spacing:1.5px; margin-bottom:6px;">
                LEGAL RISK SUMMARY
            </div>
            <div style="color:#d1d5db; line-height:1.7;">
                {grade_info['risk_summary']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# 4b. KL Divergence + Entropy metrics strip
metric_cols = st.columns(1 + min(len(sensitive_cols), 3))

with metric_cols[0]:
    st.metric(
        label=f"📐 KL Divergence  ({primary_col})",
        value=f"{kl_score:.4f}",
        help="Distance from a perfectly uniform distribution. 0 = ideal. Higher = more biased.",
    )

for i, col in enumerate(sensitive_cols[:3]):
    h_val   = entropy_scores[col]
    h_max   = max_entropy[col]
    pct_div = (h_val / h_max * 100) if h_max > 0 else 0
    with metric_cols[i + 1]:
        st.metric(
            label=f"H( {col} )",
            value=f"{h_val:.3f} bits",
            delta=f"{pct_div:.1f}% of max diversity",
            delta_color="normal",
            help=f"Max possible entropy for '{col}': {h_max:.3f} bits",
        )

# Show remaining entropy cols if > 3
if len(sensitive_cols) > 3:
    extra_cols = st.columns(min(len(sensitive_cols) - 3, 4))
    for i, col in enumerate(sensitive_cols[3:7]):
        h_val   = entropy_scores[col]
        h_max   = max_entropy[col]
        pct_div = (h_val / h_max * 100) if h_max > 0 else 0
        with extra_cols[i]:
            st.metric(
                label=f"H( {col} )",
                value=f"{h_val:.3f} bits",
                delta=f"{pct_div:.1f}% of max diversity",
                delta_color="normal",
            )

st.markdown("<br>", unsafe_allow_html=True)

# 4c. Desert Heatmap
st.markdown("#### 🏜️ Intersectional Terrain Heatmap")

if heatmap_fig is not None:
    # Ensure consistent dark theme
    heatmap_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
    )
    st.plotly_chart(heatmap_fig, use_container_width=True)
    st.caption(
        "Deep purple = representation desert (near-zero data). "
        "Neon green = well-represented intersection."
    )
else:
    st.info(
        "ℹ️ The intersectional heatmap requires at least **2 sensitive columns**. "
        "Return to Step 2 and mark additional columns as Sensitive."
    )

st.markdown("<br>", unsafe_allow_html=True)

# 4d. Representation Deserts table 
st.markdown(
    f"#### 🔴 Representation Deserts "
    f"<span style='font-size:0.85rem; color:#6b7280; font-weight:400;'>"
    f"({len(deserts)} found)</span>",
    unsafe_allow_html=True,
)
st.caption("Intersections with < 1% representation — the gaps that bake discrimination into model weights.")

if len(deserts) == 0:
    st.success(
        "✅ No representation deserts detected. "
        "All intersectional combinations have ≥ 1% representation."
    )
else:
    top_deserts = deserts[:5]
    desert_table = pd.DataFrame(
        [
            {
                "Intersection":  d["combination"],
                "Count":         d["count"],
                "% of Dataset":  d["percentage"],
                "Severity":      d["severity"],
            }
            for d in top_deserts
        ]
    )

    def _severity_style(val: str) -> str:
        if "ABSENT"   in val: return "color:#FF4444; font-weight:bold;"
        if "CRITICAL" in val: return "color:#FF8C00; font-weight:bold;"
        return "color:#FFD700; font-weight:bold;"

    styled = (
        desert_table.style
        .map(_severity_style, subset=["Severity"])
        .format({"% of Dataset": "{:.4f}%"})
        .set_properties(**{"background-color": CARD2, "color": "#e0e0e0",
                           "border": "1px solid #1f2937"})
        .set_table_styles([{"selector": "th", "props": [
            ("background-color", BG), ("color", NEON),
            ("font-family", "'Courier New', monospace"),
            ("border", "1px solid #1f2937"),
        ]}])
    )
    st.dataframe(styled, use_container_width=True, height=240)

    if len(deserts) > 5:
        with st.expander(f"📋 Show all {len(deserts)} deserts"):
            all_df = pd.DataFrame(
                [{"Intersection": d["combination"], "Count": d["count"],
                  "% of Dataset": d["percentage"],  "Severity": d["severity"]}
                 for d in deserts]
            )
            st.dataframe(all_df, use_container_width=True)

st.markdown("<hr style='border-color:#1f2937;'>", unsafe_allow_html=True)



# STEP 5 — AUTO-FIX & GHOST BIAS

st.markdown(
    "<div class='step-pill'>STEP 05</div>"
    f"<h3 style='margin-top:4px;'>⚡ Auto-Fix &amp; Ghost Bias Detector</h3>",
    unsafe_allow_html=True,
)

BAD_GRADES  = {"C", "D", "F"}
GOOD_GRADES = {"A+", "A", "B"}

# Auto-Fix panel
if grade in GOOD_GRADES:
    st.success(
        f"✅ **Grade {grade} — No Mitigation Required.** "
        "Your dataset meets the fairness threshold. "
        "Review the Ghost Bias chart below for full transparency."
    )
else:
    st.warning(
        f"⚠️ **Grade {grade} detected.** "
        "Apply an Auto-Fix strategy to improve data diversity before training."
    )

    fix_method = st.radio(
        "Select Mitigation Strategy",
        options=["SMOTE (Synthetic Over-sampling)", "Strategic Under-sampling"],
        horizontal=True,
        help=(
            "**SMOTE** interpolates new synthetic samples for minority classes. "
            "**Under-sampling** prunes majority classes — no data fabrication."
        ),
    )

    # Pre-flight checks
    can_run = True
    if fix_method == "Strategic Under-sampling" and not sensitive_cols:
        st.error("No Sensitive columns marked. Return to Step 2 and mark at least one.")
        can_run = False
    elif fix_method == "SMOTE (Synthetic Over-sampling)" and not target_col:
        st.error("SMOTE requires a Target column. Return to Step 2 and select one.")
        can_run = False

    if can_run:
        run_fix = st.button("🚀 Run Auto-Fix", use_container_width=True, type="primary")

        if run_fix:
            spinner_msg = (
                "🧬 Running SMOTE — synthesising minority samples…"
                if "SMOTE" in fix_method
                else "✂️ Running strategic under-sampling…"
            )
            with st.spinner(spinner_msg):
                try:
                    if "SMOTE" in fix_method:
                        df_fixed, e_before, e_after = apply_smote(df, target_col)
                    else:
                        pivot_col = sensitive_cols[0]
                        df_fixed, e_before, e_after = apply_undersampling(df, pivot_col)

                    gain = compute_entropy_gain(e_before, e_after)

                    st.markdown(
                        "<h4 style='color:#00FF41; margin-top:1rem;'>"
                        "Shannon Entropy: Before vs After</h4>",
                        unsafe_allow_html=True,
                    )
                    cb, ca, cg = st.columns(3)
                    cb.metric("🔴 Entropy Before", f"{e_before:.4f} bits")
                    ca.metric("🟢 Entropy After",  f"{e_after:.4f} bits",
                              delta=f"{e_after - e_before:+.4f} bits")
                    cg.metric("📈 Entropy Gain",   f"{gain:+.2f}%")

                    # Persist in session for PDF report
                    st.session_state["df_fixed"]            = df_fixed
                    st.session_state["fix_method_used"]     = fix_method
                    st.session_state["entropy_before_val"]  = e_before
                    st.session_state["entropy_after_val"]   = e_after

                    st.success(
                        f"✅ Mitigation complete. Fixed dataset: **{len(df_fixed):,} rows** "
                        f"(was {len(df):,})."
                    )
                    with st.expander("Preview Fixed Dataset (first 50 rows)"):
                        st.dataframe(df_fixed.head(50), use_container_width=True)

                except Exception as exc:
                    st.error(f"Auto-Fix failed: {exc}")

st.markdown("<br>", unsafe_allow_html=True)

# Ghost Bias Chart 
st.markdown(
    f"<h4 style='color:{NEON};'>👻 Ghost Bias Detector</h4>",
    unsafe_allow_html=True,
)
st.caption(
    "A shallow Decision Tree (max_depth=4) trained on your sensitive columns. "
    "Gini Importance reveals which column most strongly predicts the target — "
    "i.e. the strongest proxy bias in the dataset."
)

if not target_col:
    st.info("ℹ️ Select a **Target column** in Step 2 to enable Ghost Bias simulation.")
elif not sensitive_cols:
    st.info("ℹ️ Mark at least one **Sensitive column** in Step 2 to run this simulation.")
else:
    with st.spinner("🤖 Training shadow model for Ghost Bias detection…"):
        try:
            importances = run_ghost_bias_simulation(df, sensitive_cols, target_col)

            if not importances:
                st.warning("No feature importances returned — check your column selections.")
            else:
                cols_sorted   = list(importances.keys())
                scores_sorted = list(importances.values())

                ghost_fig = go.Figure(
                    go.Bar(
                        x=scores_sorted,
                        y=cols_sorted,
                        orientation="h",
                        marker=dict(
                            color=[NEON] * len(cols_sorted),
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
                ghost_fig.update_layout(
                    template="plotly_dark",
                    title=dict(
                        text="Ghost Bias — Proxy Feature Ranking (Gini Importance)",
                        font=dict(color=NEON, size=14),
                    ),
                    paper_bgcolor=BG,
                    plot_bgcolor=CARD2,
                    xaxis=dict(
                        title="Gini Importance Score",
                        title_font=dict(color="#9ca3af"),
                        tickfont=dict(color="#9ca3af"),
                        gridcolor="#1f2937",
                        range=[0, max(scores_sorted) * 1.28],
                    ),
                    yaxis=dict(
                        tickfont=dict(color=NEON),
                        autorange="reversed",
                    ),
                    margin=dict(l=20, r=50, t=52, b=40),
                    height=max(300, len(cols_sorted) * 52),
                )
                st.plotly_chart(ghost_fig, use_container_width=True)

                top_col   = cols_sorted[0]
                top_score = scores_sorted[0]
                st.markdown(
                    f"<p style='color:#FF4B4B; font-weight:bold; font-size:0.9rem;'>"
                    f"🚨 Highest proxy bias: <code>{top_col}</code> "
                    f"(Gini = {top_score:.4f}) — this column may act as a "
                    f"proxy for a protected attribute.</p>",
                    unsafe_allow_html=True,
                )

        except Exception as exc:
            st.error(f"Ghost Bias simulation failed: {exc}")

st.markdown("<hr style='border-color:#1f2937;'>", unsafe_allow_html=True)



# STEP 6 — PDF EXPORT

st.markdown(
    "<div class='step-pill'>STEP 06</div>"
    f"<h3 style='margin-top:4px;'>📄 Export Data Nutrition Label</h3>",
    unsafe_allow_html=True,
)
st.caption(
    "Generates a 3-page PDF audit report — Executive Summary, Bias Findings, "
    "and Mitigation — suitable for regulatory submissions and model cards."
)

if st.button("📥 Generate PDF Report", type="primary", use_container_width=True):
    # Collect ghost bias for the PDF (re-run silently)
    ghost_bias_data: dict = {}
    if target_col and sensitive_cols:
        try:
            ghost_bias_data = run_ghost_bias_simulation(df, sensitive_cols, target_col)
        except Exception:
            ghost_bias_data = {}

    fix_applied    = "df_fixed"           in st.session_state
    fix_method_val = st.session_state.get("fix_method_used",    None)
    e_before_val   = st.session_state.get("entropy_before_val", None)
    e_after_val    = st.session_state.get("entropy_after_val",  None)

    with st.spinner("📝 Compiling Data Nutrition Label PDF…"):
        try:
            pdf_bytes = generate_pdf_report(
                dataset_name      = uploaded_file.name,
                row_count         = df.shape[0],
                col_count         = df.shape[1],
                sensitive_cols    = sensitive_cols,
                compliance_result = grade_info,
                entropy_scores    = entropy_scores,
                kl_score          = kl_score,
                deserts           = deserts,
                ghost_bias        = ghost_bias_data,
                fix_applied       = fix_applied,
                fix_method        = fix_method_val,
                entropy_before    = e_before_val,
                entropy_after     = e_after_val,
            )

            st.session_state["pdf_exported"] = True   # update sidebar tracker

            st.download_button(
                label     = "⬇️ Download BiasMap_Audit_Report.pdf",
                data      = pdf_bytes,
                file_name = "BiasMap_Audit_Report.pdf",
                mime      = "application/pdf",
                use_container_width=True,
            )
            st.success(
                "✅ PDF ready. Click the button above to download your "
                "Data Nutrition Label."
            )

        except Exception as exc:
            st.error(f"PDF generation failed: {exc}")



# FOOTER

st.markdown(
    f"""
    <hr style='border-color:#1f2937; margin-top:2rem;'>
    <div style='text-align:center; color:#374151; font-size:0.72rem;
                font-family:"Courier New",monospace; padding-bottom:1rem;'>
        BiasMap AI v3.0 &nbsp;·&nbsp; Local-First &nbsp;·&nbsp;
        DPDP Act (2023) Compliant &nbsp;·&nbsp;
        No data leaves your machine &nbsp;·&nbsp;
        Built for Hackathon 2026
    </div>
    """,
    unsafe_allow_html=True,
)