import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE



# INTERNAL HELPERS

def _shannon_entropy(series: pd.Series) -> float:
    clean = series.fillna("Missing_Data_Unknown")
    probs = clean.value_counts(normalize=True).values
    return float(scipy_entropy(probs, base=2))


def _safe_encode_df(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:

    df_out = df.copy()
    encoders: dict[str, LabelEncoder] = {}

    for col in df_out.columns:
        if pd.api.types.is_numeric_dtype(df_out[col]):
            # Safe: median only called on numeric columns
            df_out[col] = df_out[col].fillna(df_out[col].median())
        else:
            # String / category / object / mixed → label encode
            le = LabelEncoder()
            df_out[col] = le.fit_transform(
                df_out[col].astype(str).fillna("Missing_Data_Unknown")
            )
            encoders[col] = le

    return df_out, encoders



# 1. SMOTE — Synthetic Minority Over-sampling

def apply_smote(
    df: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, float, float]:

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    entropy_before = _shannon_entropy(df[target_col])

    # Drop rows where target is NaN (SMOTE can't handle missing labels)
    df_clean = df.dropna(subset=[target_col]).copy()

    # Robustly encode all columns using the shared helper
    df_encoded, encoders = _safe_encode_df(df_clean)

    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]

    # Determine minimum class size; need at least k_neighbors + 1 samples
    min_class_count = int(y.value_counts().min())
    k_neighbors = min(5, min_class_count - 1)  # safe fallback for tiny classes

    if k_neighbors < 1:
        # Can't run SMOTE with only 1 sample in a class — return original
        return df_clean, entropy_before, entropy_before

    smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    # Reconstruct a DataFrame with original column order
    df_res = pd.DataFrame(X_res, columns=X.columns)
    df_res[target_col] = y_res

    # Decode label-encoded columns back to original strings for readability
    for col, le in encoders.items():
        if col in df_res.columns:
            # Clip to valid range before inverse_transform to avoid OOB errors
            max_label = len(le.classes_) - 1
            df_res[col] = le.inverse_transform(
                df_res[col].astype(int).clip(0, max_label)
            )

    entropy_after = _shannon_entropy(df_res[target_col])

    return df_res, entropy_before, entropy_after



# 2. Strategic Under-sampling

def apply_undersampling(
    df: pd.DataFrame,
    sensitive_col: str,
) -> tuple[pd.DataFrame, float, float]:

    if sensitive_col not in df.columns:
        raise ValueError(f"Sensitive column '{sensitive_col}' not found in dataframe.")

    entropy_before = _shannon_entropy(df[sensitive_col])

    # Fill NaNs so they appear as a distinct category in the groupby
    df_work = df.copy()
    df_work[sensitive_col] = df_work[sensitive_col].fillna("Missing_Data_Unknown")

    # Minority class count becomes the target count for ALL classes
    min_count = int(df_work[sensitive_col].value_counts().min())

    # Sample exactly min_count rows from every group (stratified pruning)
    resampled = (
        df_work.groupby(sensitive_col, group_keys=False)
        .apply(lambda g: g.sample(n=min_count, random_state=42))
        .reset_index(drop=True)
    )

    entropy_after = _shannon_entropy(resampled[sensitive_col])

    return resampled, entropy_before, entropy_after



# 3. Ghost Bias Simulation (Explainability Layer)

def run_ghost_bias_simulation(
    df: pd.DataFrame,
    sensitive_cols: list[str],
    target_col: str,
) -> dict[str, float]:

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    # Keep only sensitive cols + target; drop rows missing the target label
    cols_to_use = [c for c in sensitive_cols if c in df.columns]
    if not cols_to_use:
        raise ValueError("None of the specified sensitive columns exist in the dataframe.")

    df_work = df[cols_to_use + [target_col]].dropna(subset=[target_col]).copy()

    # Robustly encode using the shared helper (fixes the median crash)
    df_encoded, _ = _safe_encode_df(df_work)

    X = df_encoded[cols_to_use]
    y = df_encoded[target_col]

    # Shallow tree — interpretable proxy model, not production-grade
    clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    clf.fit(X, y)

    # Build importance dict and sort descending (highest proxy bias first)
    importances = dict(zip(cols_to_use, clf.feature_importances_))
    sorted_importances = dict(
        sorted(importances.items(), key=lambda x: x[1], reverse=True)
    )

    return sorted_importances



# 4. Entropy Gain Calculator

def compute_entropy_gain(entropy_before: float, entropy_after: float) -> float:

    if entropy_before == 0:
        return 0.0
    gain = ((entropy_after - entropy_before) / entropy_before) * 100
    return round(gain, 2)