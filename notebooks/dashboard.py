import json
import os

import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path


# ===============================================================
# CORRECT PATH CONFIGURATION (dashboard.py is in /notebooks)
# ===============================================================

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ===============================================================
# DATA LOADING FUNCTIONS
# ===============================================================

@st.cache_data
def load_datasets():
    df_clean = None
    df_anova = None

    path_clean = DATA_DIR / "Cleaned_Features_for_ML.csv"
    path_anova = DATA_DIR / "Cleaned_Features_for_ML_20ANOVA.csv"

    if path_clean.exists():
        df_clean = pd.read_csv(path_clean)
    if path_anova.exists():
        df_anova = pd.read_csv(path_anova)

    return df_clean, df_anova


@st.cache_data
def load_model_results():
    json_path = DATA_DIR / "model_results.json"
    if not json_path.exists():
        return None

    with open(json_path, "r") as f:
        return json.load(f)


# ⭐ NEW — FEATURE RANKINGS LOADER
@st.cache_data
def load_feature_rankings():
    """
    Loads:
      - top20_features.csv
      - feature_ranking_full.csv
    from /data folder
    """
    top20_df = None
    full_df = None

    top20_path = DATA_DIR / "top20_features.csv"
    full_path = DATA_DIR / "feature_ranking_full.csv"

    if top20_path.exists():
        top20_df = pd.read_csv(top20_path)

    if full_path.exists():
        full_df = pd.read_csv(full_path)

    return top20_df, full_df


# ===============================================================
# CONVERT RESULTS DICTIONARY -> TABLE
# ===============================================================

def build_results_table(results):
    rows = []

    for model_name, res in results.items():

        cm = np.array(res.get("confusion_matrix"))
        report = res.get("classification_report", {})

        roc_auc = res.get("roc_auc", None)
        f2 = res.get("f2_score", None)
        time_sec = res.get("computation_time_sec", None)
        acc = report.get("accuracy", None)

        # Positive class key
        pos_key = "1" if "1" in report else None
        if not pos_key:
            numeric_keys = [k for k in report.keys() if k.isdigit()]
            if numeric_keys:
                pos_key = numeric_keys[-1]

        precision_1 = recall_1 = f1_1 = None
        if pos_key and pos_key in report:
            precision_1 = report[pos_key].get("precision")
            recall_1 = report[pos_key].get("recall")
            f1_1 = report[pos_key].get("f1-score")

        # Confusion matrix
        tn = fp = fn = tp = None
        if isinstance(cm, np.ndarray) and cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()

        rows.append({
            "Model": model_name,
            "Accuracy": acc,
            "Precision (class 1)": precision_1,
            "Recall (class 1)": recall_1,
            "F1-score (class 1)": f1_1,
            "F2-score": f2,
            "ROC-AUC": roc_auc,
            "Computation Time (sec)": time_sec,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp
        })

    return pd.DataFrame(rows)


# ===============================================================
# PAGE: DATA OVERVIEW
# ===============================================================

def render_overview_page(df_clean, df_anova):

    st.title("📊 Dataset Overview")

    dataset_choice = st.radio(
        "Select dataset to explore:",
        ["Cleaned Dataset", "ANOVA Reduced Dataset"]
    )

    df = df_clean if dataset_choice == "Cleaned Dataset" else df_anova

    if df is None:
        st.error("Selected dataset not found in /data folder.")
        return

    st.write(f"**Dataset shape:** {df.shape[0]:,} rows × {df.shape[1]:,} columns")

    cols = st.multiselect(
        "Select columns to display:",
        df.columns.tolist(),
        default=df.columns[:10].tolist()
    )

    if len(cols) == 0:
        st.info("Please select at least one column.")
        return

    st.subheader("🔍 Data Preview")
    st.dataframe(df[cols].head())

    if st.checkbox("Show summary statistics"):
        st.write(df[cols].describe(include="all"))

    if st.checkbox("Show correlation matrix (numeric columns only)"):

        numeric_cols = df[cols].select_dtypes(include=[np.number])

        if numeric_cols.shape[1] < 2:
            st.warning("Select at least two numeric columns.")
        else:
            st.subheader("📈 Correlation Matrix")

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(numeric_cols.corr(), cmap="Blues", annot=False, ax=ax)
            st.pyplot(fig)


# ===============================================================
# PAGE: MODEL RESULTS
# ===============================================================

def render_model_results_page(results):

    st.title("🏆 Model Performance — Ranked by AUC")

    if results is None:
        st.warning("model_results.json not found in /data folder.")
        return

    df_results = build_results_table(results)
    df_sorted = df_results.sort_values(by="ROC-AUC", ascending=False)

    styled = df_sorted.style.background_gradient(
        cmap="Blues",
        subset=[
            "Accuracy", "Precision (class 1)", "Recall (class 1)",
            "F1-score (class 1)", "F2-score", "ROC-AUC",
            "Computation Time (sec)"
        ]
    )

    st.dataframe(styled)


# ===============================================================
# ⭐ NEW PAGE: FEATURE SELECTION (ANOVA)
# ===============================================================

def render_feature_selection_page():
    st.title("📊 Feature Selection (ANOVA F-score)")

    top20_df, full_df = load_feature_rankings()

    if top20_df is None and full_df is None:
        st.warning("No feature ranking files found in /data.")
        st.info("Expected files: top20_features.csv, feature_ranking_full.csv")
        return

    # --- Display Top-20 ---
    if top20_df is not None:
        st.subheader("🏅 Top-20 Selected Features")
        st.dataframe(top20_df)

    st.markdown("---")

    # --- Display Full Ranking ---
    if full_df is not None:

        st.subheader("📋 Full ANOVA Feature Ranking")
        st.dataframe(
            full_df.sort_values("ANOVA_F_score", ascending=False)
        )

        # Bar chart for top-20
        st.subheader("📈 Top-20 Features — F-score Bar Chart")

        chart_df = full_df.sort_values("ANOVA_F_score", ascending=False).head(20)
        st.bar_chart(chart_df.set_index("Feature")["ANOVA_F_score"])


# ===============================================================
# MAIN APP
# ===============================================================

def main():
    st.set_page_config(page_title="ML Dashboard", layout="wide")

    df_clean, df_anova = load_datasets()
    results = load_model_results()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to:",
        ["Overview", "Model Results", "Feature Selection (ANOVA)"]   # ⭐ NEW
    )

    if page == "Overview":
        render_overview_page(df_clean, df_anova)

    elif page == "Model Results":
        render_model_results_page(results)

    else:  # ⭐ NEW PAGE
        render_feature_selection_page()


if __name__ == "__main__":
    main()
