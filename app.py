# BiasMap AI | Upload + Schema Verification Page

import re
import json
import streamlit as st
import pandas as pd

# 1. PAGE CONFIG 
st.set_page_config(
    page_title="BiasMap AI",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 2. GLOBAL CSS {Obsidian Auditor Theme (Injects dark background + neon green accent across all widgets)}
st.markdown(
    """
    <style>
        /* ---- Base background ---- */
        html, body, [data-testid="stAppViewContainer"],
        [data-testid="stHeader"], [data-testid="stToolbar"] {
            background-color: #0e1117 !important;
            color: #e0e0e0 !important;
        }

        /* ---- Sidebar (collapsed by default, style anyway) ---- */
        [data-testid="stSidebar"] { background-color: #161b22 !important; }

        /* ---- Metric cards / info boxes ---- */
        [data-testid="stMetric"] {
            background-color: #161b22;
            border: 1px solid #00FF41;
            border-radius: 8px;
            padding: 12px;
        }

        /* ---- Dataframe header ---- */
        thead tr th {
            background-color: #161b22 !important;
            color: #00FF41 !important;
        }

        /* ---- Radio buttons label color ---- */
        .stRadio label { color: #c0c0c0 !important; }

        /* ---- File uploader border ---- */
        [data-testid="stFileUploader"] {
            border: 1px dashed #00FF41;
            border-radius: 8px;
            padding: 8px;
        }

        /* ---- Confirm button — override Streamlit primary ---- */
        div[data-testid="stFormSubmitButton"] > button,
        .confirm-btn > button {
            background-color: #00FF41 !important;
            color: #0e1117 !important;
            font-weight: 700 !important;
            border: none !important;
            border-radius: 6px !important;
            padding: 0.5rem 2rem !important;
            font-size: 1rem !important;
        }

        /* ---- Section divider ---- */
        hr { border-color: #00FF41; opacity: 0.3; }

        /* ---- Neon heading helper ---- */
        .neon { color: #00FF41; }
    </style>
    """,
    unsafe_allow_html=True,
)

# 3. CONSTANTS (regex keywords that flag sensitive columns)
SENSITIVE_KEYWORDS = [
    r"gender", r"sex", r"race", r"age", r"religion",
    r"caste", r"pincode", r"zip", r"region", r"state",
    r"language", r"nationality", r"income",
]

# Pre compile into one pattern
SENSITIVE_PATTERN = re.compile(
    "|".join(SENSITIVE_KEYWORDS), flags=re.IGNORECASE
)


# 4. HELPER (detect sensitive columns via regex)
def detect_sensitive_columns(columns: list[str]) -> dict[str, str]:
    """
    Returns a dict  {col_name: suggested_role}
    suggested_role is 'Sensitive' if regex matches, else 'Exclude'.
    """
    suggestions = {}
    for col in columns:
        if SENSITIVE_PATTERN.search(col):
            suggestions[col] = "Sensitive"
        else:
            suggestions[col] = "Exclude"
    return suggestions


# 5. HELPER (load uploaded file into a DataFrame)
def load_file(uploaded_file) -> pd.DataFrame | None:
    """Supports CSV and JSON; returns None on parse failure."""
    try:
        if uploaded_file.name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".json"):
            return pd.read_json(uploaded_file)
        else:
            st.error("❌ Unsupported file type. Please upload a CSV or JSON file.")
            return None
    except Exception as exc:
        st.error(f"❌ Failed to parse file: {exc}")
        return None


# 6. SESSION STATE (initialise keys once)
# Guards against re-init on every Streamlit rerun
for key, default in {
    "df": None,               # Raw DataFrame
    "schema": {},             # {col: role} — user confirmed roles
    "schema_confirmed": False # Gate flag for analysis section
}.items():
    if key not in st.session_state:
        st.session_state[key] = default



#  PAGE LAYOUT

# Header
st.markdown(
    "<h1 style='color:#00FF41; letter-spacing:2px;'>🗺️ BiasMap AI</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#888; margin-top:-12px;'>"
    "Dataset Bias Cartography & DPDP Compliance Auditor &nbsp;|&nbsp; "
    "<span style='color:#00FF41;'>Local-Only · Zero Cloud · Privacy-First</span>"
    "</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# 7. FILE UPLOADER
st.markdown("### 📂 Upload Your Dataset")
st.caption("All processing happens locally on your machine. No data is transmitted anywhere.")

uploaded_file = st.file_uploader(
    label="Drop a CSV or JSON file here",
    type=["csv", "json"],
    help="Max recommended size: ~200 MB for smooth performance.",
)

# 8. ON FILE UPLOAD (preview + auto-detect)
if uploaded_file is not None:

    # Loading only if it's a new file (avoid re-parsing on every widget interaction)
    if st.session_state.df is None or uploaded_file.name not in st.session_state.get("loaded_filename", ""):
        st.session_state.df = load_file(uploaded_file)
        st.session_state.loaded_filename = uploaded_file.name
        st.session_state.schema_confirmed = False  # Reset gate on new upload

    df = st.session_state.df

    if df is not None:

        st.markdown("---")
        # 8a. File stats row
        st.markdown("### 📊 File Preview")
        col_rows, col_cols, col_name = st.columns(3)

        col_rows.metric("Total Rows", f"{len(df):,}")
        col_cols.metric("Total Columns", f"{len(df.columns):,}")
        col_name.metric("File", uploaded_file.name)

        # Show first 5 rows in a styled table
        st.dataframe(
            df.head(5),
            use_container_width=True,
            hide_index=False,
        )

        st.markdown("---")

        # 9. SCHEMA VERIFICATION UI
        st.markdown("### 🔍 Schema Verification — Confirm Column Roles")
        st.info(
            "BiasMap AI has **auto-detected** sensitive columns via keyword matching. "
            "Review each column and assign the correct role before proceeding. "
            "This step prevents *Garbage In → Garbage Out* errors.",
            icon="ℹ️",
        )

        # Run auto-detection once per file
        auto_suggestions = detect_sensitive_columns(df.columns.tolist())

        # Role options for radio buttons
        ROLE_OPTIONS = ["Sensitive", "Target", "Exclude"]

        # Build the schema table header
        header_cols = st.columns([3, 2, 2, 5])
        header_cols[0].markdown("<span class='neon'>**Column Name**</span>", unsafe_allow_html=True)
        header_cols[1].markdown("<span class='neon'>**Dtype**</span>", unsafe_allow_html=True)
        header_cols[2].markdown("<span class='neon'>**Auto-Tag**</span>", unsafe_allow_html=True)
        header_cols[3].markdown("<span class='neon'>**Your Role Selection**</span>", unsafe_allow_html=True)

        st.markdown("<hr style='margin:4px 0 10px 0;'>", unsafe_allow_html=True)

        # Temporary dict to collect radio selections before confirmation
        pending_schema: dict[str, str] = {}

        for col in df.columns:
            auto_role = auto_suggestions[col]
            dtype_str = str(df[col].dtype)

            # Colour-code the auto-tag badge
            badge_color = "#00FF41" if auto_role == "Sensitive" else "#555"
            badge_html = (
                f"<span style='background:{badge_color}; color:#0e1117; "
                f"padding:2px 8px; border-radius:4px; font-size:0.75rem; font-weight:700;'>"
                f"{auto_role}</span>"
            )

            # Wrap each row in a container to prevent Streamlit column collapse
            with st.container():
                row_cols = st.columns([3, 2, 2, 5])
                row_cols[0].markdown(f"`{col}`")
                row_cols[1].markdown(
                    f"<span style='color:#888;'>{dtype_str}</span>",
                    unsafe_allow_html=True,
                )
                row_cols[2].markdown(badge_html, unsafe_allow_html=True)

                # Sanitize col name to make a safe Streamlit widget key
                safe_key = re.sub(r"[^a-zA-Z0-9_]", "_", col)

                selected_role = row_cols[3].radio(
                    label=f"Role for {col}",
                    options=ROLE_OPTIONS,
                    index=ROLE_OPTIONS.index(auto_role),
                    horizontal=True,
                    label_visibility="collapsed",
                    key=f"radio_{safe_key}",   # sanitized key — fixes the grouping bug
                )
            pending_schema[col] = selected_role

        st.markdown("---")

        # 10. CONFIRM BUTTON
        st.markdown("<div class='confirm-btn'>", unsafe_allow_html=True)
        confirm_clicked = st.button(
            "✅  Confirm Schema & Proceed",
            type="primary",          # triggers the CSS override above
            use_container_width=False,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if confirm_clicked:
            # Validate: at least one Sensitive and one Target column selected
            roles = list(pending_schema.values())
            if "Sensitive" not in roles:
                st.error("❌ Please mark at least **one column** as **Sensitive** before proceeding.")
            elif "Target" not in roles:
                st.warning("⚠️ No **Target** column selected. Ghost Bias analysis will be skipped.")
                # Still allow confirmation — Target is optional for entropy-only audit
                st.session_state.schema = pending_schema
                st.session_state.schema_confirmed = True
            else:
                # Save confirmed schema to session state
                st.session_state.schema = pending_schema
                st.session_state.schema_confirmed = True

        # 11. POST CONFIRMATION STATE
        if st.session_state.schema_confirmed:

            # Count roles for summary
            confirmed = st.session_state.schema
            sensitive_cols = [c for c, r in confirmed.items() if r == "Sensitive"]
            target_cols    = [c for c, r in confirmed.items() if r == "Target"]
            excluded_cols  = [c for c, r in confirmed.items() if r == "Exclude"]

            st.success(
                f"✅ Schema confirmed! "
                f"**{len(sensitive_cols)}** Sensitive · "
                f"**{len(target_cols)}** Target · "
                f"**{len(excluded_cols)}** Excluded columns locked in.",
                icon="🔒",
            )

            # Show a compact role summary
            with st.expander("📋 View Confirmed Schema", expanded=False):
                summary_df = pd.DataFrame(
                    [(col, role) for col, role in confirmed.items()],
                    columns=["Column", "Role"],
                )
                # Colour-map the Role column
                def highlight_role(val):
                    colors = {
                        "Sensitive": "color: #00FF41; font-weight:bold",
                        "Target":    "color: #FFD700; font-weight:bold",
                        "Exclude":   "color: #555555",
                    }
                    return colors.get(val, "")

                st.dataframe(
                    summary_df.style.map(highlight_role, subset=["Role"]),
                    use_container_width=True,
                    hide_index=True,
                )

            st.markdown("---")
            
            # STEP 2: CARTOGRAPHY ENGINE
            # Runs after schema is confirmed. Encodes sensitive cols → UMAP → Plotly map.

            import umap                                     # dimensionality reduction
            from sklearn.preprocessing import LabelEncoder  # encode categorical → numeric
            import plotly.express as px                     # interactive scatter plot

            st.markdown("## 🗺️ Terrain Map — Bias Cartography Engine")
            st.markdown("---")

            # 2a. Identify sensitive columns from confirmed schema
            sensitive_cols = [
                col for col, role in st.session_state.schema.items()
                if role == "Sensitive"
            ]

            if not sensitive_cols:
                st.error("❌ No Sensitive columns found in schema. Go back and mark at least one.")
            else:
                # 2b. Stratified sampling gate
                # PRD requirement: datasets > 100k rows must be sampled to prevent
                # UMAP memory overflow
                SAMPLE_LIMIT = 100_000
                df_full = st.session_state.df   # original full dataframe

                if len(df_full) > SAMPLE_LIMIT:
                    # Stratify by the first sensitive column so minority groups
                    # are proportionally preserved in the sample
                    strat_col = sensitive_cols[0]

                    # Fill NaNs in strat column so groupby doesn't drop rows
                    df_strat = df_full.copy()
                    df_strat[strat_col] = df_strat[strat_col].fillna("Data_Gap_Unknown")

                    # Compute per-stratum sample sizes (proportional)
                    group_fractions = df_strat[strat_col].value_counts(normalize=True)
                    sampled_parts = []
                    for group, fraction in group_fractions.items():
                        group_df = df_strat[df_strat[strat_col] == group]
                        n = max(1, int(fraction * SAMPLE_LIMIT))   # at least 1 row per group
                        sampled_parts.append(group_df.sample(n=min(n, len(group_df)), random_state=42))

                    df_work = pd.concat(sampled_parts).reset_index(drop=True)
                    scan_mode = f"⚡ Stratified Sample — {len(df_work):,} rows (stratified on `{strat_col}`)"
                    badge_color = "#FFD700"   # gold badge for sampled mode

                else:
                    df_work = df_full.copy()
                    scan_mode = f"✅ Full Scan — {len(df_work):,} rows"
                    badge_color = "#00FF41"   # green badge for full scan

                # Display scan mode badge
                st.markdown(
                    f"<div style='display:inline-block; background:{badge_color}; "
                    f"color:#0e1117; font-weight:700; padding:6px 16px; "
                    f"border-radius:6px; font-size:0.9rem; margin-bottom:12px;'>"
                    f"{scan_mode}</div>",
                    unsafe_allow_html=True,
                )

                # 2c. Build UMAP input matrix 
                # UMAP needs numeric data — LabelEncode every sensitive column.
                # NaNs are filled with the PRD-mandated 'Data_Gap_Unknown' sentinel.
                st.markdown("#### ⚙️ Encoding & Projecting…")
                encode_progress = st.progress(0, text="Encoding sensitive columns…")

                umap_input = pd.DataFrame()   # will hold encoded columns
                le = LabelEncoder()

                for i, col in enumerate(sensitive_cols):
                    filled = df_work[col].fillna("Data_Gap_Unknown").astype(str)
                    umap_input[col] = le.fit_transform(filled)
                    encode_progress.progress(
                        int((i + 1) / len(sensitive_cols) * 40),
                        text=f"Encoded: {col}",
                    )

                # 2d. Run UMAP
                # n_components=2 → 2D scatter; random_state for reproducibility.
                # n_neighbors default (15) works well; min_dist=0.1 keeps clusters tight.
                encode_progress.progress(45, text="Running UMAP projection (this may take ~30s for large files)…")

                reducer = umap.UMAP(
                    n_components=2,
                    random_state=42,
                    n_neighbors=15,
                    min_dist=0.1,
                    metric="euclidean",
                )
                embedding = reducer.fit_transform(umap_input.values)  # returns (N, 2) array

                encode_progress.progress(90, text="Attaching coordinates…")

                # Attach UMAP coordinates back to working dataframe
                df_work = df_work.copy()
                df_work["umap_x"] = embedding[:, 0]
                df_work["umap_y"] = embedding[:, 1]

                encode_progress.progress(100, text="Done ✅")
                encode_progress.empty()   # remove progress bar once complete

                # 2e. Build hover data dict
                # Plotly hover_data: show all sensitive column values per point
                hover_data = {col: True for col in sensitive_cols}
                hover_data["umap_x"] = False   # hide raw coordinates from hover
                hover_data["umap_y"] = False

                # 2f. Plotly scatter — BiasMap Terrain
                # Color by first sensitive column; dark Obsidian Auditor theme
                color_col = sensitive_cols[0]

                fig = px.scatter(
                    df_work,
                    x="umap_x",
                    y="umap_y",
                    color=df_work[color_col].fillna("Data_Gap_Unknown").astype(str),
                    hover_data=hover_data,
                    title="BiasMap AI — Dataset Terrain Map",
                    labels={"color": color_col, "umap_x": "UMAP Dimension 1", "umap_y": "UMAP Dimension 2"},
                    color_discrete_sequence=px.colors.qualitative.Vivid,
                )

                # Apply Obsidian Auditor dark theme styling
                fig.update_layout(
                    plot_bgcolor="#0e1117",
                    paper_bgcolor="#0e1117",
                    font_color="#e0e0e0",
                    title_font=dict(color="#00FF41", size=18, family="monospace"),
                    legend=dict(
                        bgcolor="#161b22",
                        bordercolor="#00FF41",
                        borderwidth=1,
                        font=dict(color="#e0e0e0"),
                    ),
                    xaxis=dict(
                        gridcolor="#1f2937",
                        zerolinecolor="#1f2937",
                        title_font=dict(color="#888"),
                        tickfont=dict(color="#888"),
                    ),
                    yaxis=dict(
                        gridcolor="#1f2937",
                        zerolinecolor="#1f2937",
                        title_font=dict(color="#888"),
                        tickfont=dict(color="#888"),
                    ),
                    margin=dict(l=20, r=20, t=50, b=20),
                )

                fig.update_traces(
                    marker=dict(size=4, opacity=0.75),
                )

                st.plotly_chart(fig, use_container_width=True)

                # 2g. Show sampled dataframe below the map
                st.markdown("#### 📋 Dataset Used for Projection")
                st.caption(
                    f"Showing the {len(df_work):,} rows passed into UMAP. "
                    "Sensitive column values are as-loaded (NaNs shown as 'Data_Gap_Unknown' internally)."
                )
                st.dataframe(
                    df_work.drop(columns=["umap_x", "umap_y"]).head(500),
                    use_container_width=True,
                    hide_index=False,
                )
                st.caption("👆 Displaying first 500 rows of the working dataset for performance.")

                # Store working dataframe in session state for Step 3
                st.session_state.df_work = df_work
                st.session_state.sensitive_cols = sensitive_cols

# 13. EMPTY STATE (no file yet)
else:
    st.markdown(
        "<div style='text-align:center; padding: 60px 0; color:#444;'>"
        "<h3>⬆️ Upload a dataset above to begin your bias audit.</h3>"
        "<p>Supported formats: <code>.csv</code> &nbsp;|&nbsp; <code>.json</code></p>"
        "</div>",
        unsafe_allow_html=True,
    )