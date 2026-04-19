import numpy as np
import pandas as pd
from scipy.stats import entropy
import plotly.graph_objects as go
import plotly.express as px
from itertools import combinations



# 1. KL DIVERGENCE (with Laplace Smoothing)

def compute_safe_kl_divergence(p_dist: np.ndarray, q_dist: np.ndarray, alpha: float = 1e-6) -> float:
    """
    Computes KL Divergence between dataset distribution (p) and benchmark (q).
    Laplace Smoothing prevents log(0) / division-by-zero when a category
    exists in the benchmark but is entirely absent in the dataset.

    Args:
        p_dist: 1-D array of raw counts/probabilities for the dataset.
        q_dist: 1-D array of raw counts/probabilities for the benchmark.
        alpha:  Smoothing constant (default 1e-6 per PRD spec).

    Returns:
        Scalar float — the KL divergence score (lower = closer to benchmark).
    """
    p_arr = np.array(p_dist, dtype=float)
    q_arr = np.array(q_dist, dtype=float)

    # Apply Laplace smoothing so no probability is exactly 0
    p_smooth = (p_arr + alpha) / (np.sum(p_arr) + alpha * len(p_arr))
    q_smooth = (q_arr + alpha) / (np.sum(q_arr) + alpha * len(q_arr))

    # KL(P || Q) = Σ P(x) * log(P(x) / Q(x))
    return float(np.sum(p_smooth * np.log(p_smooth / q_smooth)))



# 2. SHANNON ENTROPY

def get_shannon_entropy(series: pd.Series) -> float:
    """
    Computes Shannon Entropy H(X) = -Σ P(x) * log2(P(x)) for a column.
    NaN values are treated as a distinct category 'Missing_Data_Unknown'
    so that missing data is surfaced as bias, not silently dropped.

    Args:
        series: A Pandas Series representing one sensitive attribute column.

    Returns:
        Scalar float — entropy in bits (max = log2(n_unique_categories)).
    """
    # Replace NaNs with a sentinel string — treated as its own bias category
    clean_series = series.fillna("Missing_Data_Unknown")

    # Compute normalized value counts (probability distribution)
    counts = clean_series.value_counts(normalize=True)

    # scipy.stats.entropy with base=2 gives Shannon entropy in bits
    return float(entropy(counts, base=2))



# 3. REPRESENTATION DESERT DETECTION

def detect_representation_deserts(df: pd.DataFrame, sensitive_cols: list[str]) -> list[dict]:
    """
    Performs multi-dimensional binning across ALL sensitive columns to find
    intersectional representation deserts — combinations that appear in
    fewer than 1% of the total dataset (the "Rural Woman" problem).

    Strategy:
      - For pairs of sensitive columns, build a cross-tab.
      - Each cell's percentage is checked against the 1% threshold.
      - Returns a sorted list of dicts (worst deserts first).

    Args:
        df:             The uploaded DataFrame.
        sensitive_cols: List of column names the user marked as Sensitive.

    Returns:
        List of dicts, each with keys:
          'combination', 'count', 'percentage', 'severity'
    """
    if len(sensitive_cols) < 2:
        # Can't compute intersections with fewer than 2 sensitive columns
        return []

    total_rows = len(df)
    deserts = []

    # Fill NaNs so they show up as a visible category in groupby
    df_clean = df[sensitive_cols].fillna("Missing_Data_Unknown")

    # Check all pairwise combinations (2-col intersections)
    for col_a, col_b in combinations(sensitive_cols, 2):
        # Build a cross-tab of counts for this pair
        crosstab = pd.crosstab(df_clean[col_a], df_clean[col_b])

        # Iterate every cell in the cross-tab
        for val_a in crosstab.index:
            for val_b in crosstab.columns:
                count = int(crosstab.loc[val_a, val_b])
                pct = (count / total_rows) * 100

                # Flag as desert if below 1% threshold
                if pct < 1.0:
                    # Severity label based on how bad the underrepresentation is
                    if count == 0:
                        severity = "🔴 ABSENT"
                    elif pct < 0.1:
                        severity = "🟠 CRITICAL"
                    else:
                        severity = "🟡 WARNING"

                    deserts.append({
                        "combination": f"{col_a}={val_a}  ×  {col_b}={val_b}",
                        "col_a": col_a,
                        "val_a": str(val_a),
                        "col_b": col_b,
                        "val_b": str(val_b),
                        "count": count,
                        "percentage": round(pct, 4),
                        "severity": severity,
                    })

    # Sort by percentage ascending — worst deserts shown first
    deserts.sort(key=lambda x: x["percentage"])
    return deserts



# 4. COMPLIANCE GRADE

def compute_compliance_grade(kl_score: float) -> dict:
    """
    Maps a KL Divergence score to an Alpha-Grade and DPDP Legal Risk Summary.

    Thresholds (from PRD):
      < 0.1  → A+  (Excellent)
      < 0.3  → A   (Good)
      < 0.6  → B   (Moderate)
      < 1.0  → C   (Elevated Risk)
      < 1.5  → D   (High Risk)
      >= 1.5 → F   (Non-Compliant)

    Args:
        kl_score: The computed KL Divergence value.

    Returns:
        Dict with keys: grade, risk_summary, color, badge_text
    """
    if kl_score < 0.1:
        return {
            "grade": "A+",
            "risk_summary": (
                "Dataset distribution closely mirrors the benchmark. "
                "Low DPDP / NITI Aayog legal risk. Suitable for production training."
            ),
            "color": "#00FF41",       # Neon green — all clear
            "badge_text": "COMPLIANT",
        }
    elif kl_score < 0.3:
        return {
            "grade": "A",
            "risk_summary": (
                "Minor distributional drift detected. Generally compliant with "
                "DPDP 'Fair Processing' norms. Minor augmentation recommended."
            ),
            "color": "#39FF14",       # Slightly deeper green
            "badge_text": "COMPLIANT",
        }
    elif kl_score < 0.6:
        return {
            "grade": "B",
            "risk_summary": (
                "Moderate bias detected. Some demographic groups are under-represented. "
                "Run the Auto-Fix engine before training. Disclose limitations per DPDP."
            ),
            "color": "#FFD700",       # Yellow — caution
            "badge_text": "REVIEW REQUIRED",
        }
    elif kl_score < 1.0:
        return {
            "grade": "C",
            "risk_summary": (
                "Elevated bias risk. Significant representation gaps found. "
                "High likelihood of DPDP 'Fair Processing' violation. Do NOT train without remediation."
            ),
            "color": "#FF8C00",       # Orange — elevated risk
            "badge_text": "HIGH RISK",
        }
    elif kl_score < 1.5:
        return {
            "grade": "D",
            "risk_summary": (
                "Severe distributional skew. Dataset likely to produce discriminatory model outputs. "
                "Probable DPDP non-compliance. Mandatory remediation required."
            ),
            "color": "#FF4500",       # Red-orange — danger
            "badge_text": "NON-COMPLIANT",
        }
    else:
        return {
            "grade": "F",
            "risk_summary": (
                "Critical failure. Dataset is fundamentally unrepresentative. "
                "Training on this data is a direct DPDP / NITI Aayog violation. "
                "Halt all training immediately and remediate the source data."
            ),
            "color": "#FF0000",       # Red — full stop
            "badge_text": "CRITICAL VIOLATION",
        }



# 5. DESERT HEATMAP (Plotly)

def generate_desert_heatmap(df: pd.DataFrame, sensitive_cols: list[str]) -> go.Figure | None:
    """
    Generates an interactive Plotly heatmap showing co-occurrence counts
    for the first two sensitive columns. Cells with low counts (< 1% of
    total) are visually distinct — they are the "representation deserts."

    If fewer than 2 sensitive columns exist, returns None.

    Args:
        df:             The uploaded DataFrame.
        sensitive_cols: List of column names marked Sensitive by the user.

    Returns:
        A Plotly Figure object, or None if insufficient columns.
    """
    if len(sensitive_cols) < 2:
        return None

    # Use the first two sensitive columns for the primary heatmap
    col_a, col_b = sensitive_cols[0], sensitive_cols[1]

    # Fill NaNs so they appear as their own category in the heatmap
    df_clean = df[[col_a, col_b]].fillna("Missing_Data_Unknown")

    # Build cross-tab of raw counts
    crosstab = pd.crosstab(df_clean[col_a], df_clean[col_b])

    total = df_clean.shape[0]

    # Compute percentage matrix for the hover tooltip
    pct_matrix = (crosstab / total * 100).round(3)

    # Annotate cells: show both count and % for readability
    annotation_text = crosstab.astype(str) + "<br>(" + pct_matrix.astype(str) + "%)"

    fig = go.Figure(
        data=go.Heatmap(
            z=crosstab.values,
            x=list(crosstab.columns.astype(str)),
            y=list(crosstab.index.astype(str)),
            text=annotation_text.values,
            texttemplate="%{text}",
            textfont={"size": 11, "color": "white"},
            hovertemplate=(
                f"<b>{col_a}</b>: %{{y}}<br>"
                f"<b>{col_b}</b>: %{{x}}<br>"
                "<b>Count</b>: %{z}<extra></extra>"
            ),
            # Colorscale: dark purple for deserts → neon green for dense clusters
            colorscale=[
                [0.0,  "#1a0033"],   # Near-zero = desert (deep purple)
                [0.15, "#6600cc"],   # Very sparse
                [0.4,  "#0066ff"],   # Moderate
                [0.7,  "#00cc88"],   # Good representation
                [1.0,  "#00FF41"],   # Fully represented (neon green)
            ],
            showscale=True,
            colorbar=dict(
                title="Count",
                title_font_color="#00FF41",
                tickfont=dict(color="#aaaaaa"),
                bgcolor="#0e1117",
                bordercolor="#333333",
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text=f"🗺️ Intersectional Terrain Map: <b>{col_a}</b> × <b>{col_b}</b>",
            font=dict(color="#00FF41", size=16),
            x=0.01,
        ),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#cccccc"),
        xaxis=dict(
            title=col_b,
            title_font_color="#00FF41",
            tickfont=dict(color="#aaaaaa"),
            gridcolor="#1f2937",
        ),
        yaxis=dict(
            title=col_a,
            title_font_color="#00FF41",
            tickfont=dict(color="#aaaaaa"),
            gridcolor="#1f2937",
        ),
        margin=dict(l=10, r=10, t=50, b=10),
        height=420,
    )

    return fig