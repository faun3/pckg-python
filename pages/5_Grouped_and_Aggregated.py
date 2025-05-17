import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from data_utils import (
    load_data,
    impute_missing_values,
    handle_extreme_values,
    encode_category,
    scale_numeric,
)

st.header("Grouped and Aggregated Data Analysis")

df = load_data()

if df.empty:
    st.warning("No data loaded.")
    st.stop()

# --- Step 1: Handle Missing Values ---
st.subheader("Step 1: Handle Missing Values")
impute_method = st.selectbox(
    "Choose imputation method",
    [
        "Mean (numeric)",
        "KNN (numeric)",
        "Forward Fill",
        "Backward Fill",
        "Drop Rows",
        "None",
    ],
)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

df_imputed = impute_missing_values(df, impute_method, numeric_cols)

# --- Step 2: Handle Extreme Values ---
st.subheader("Step 2: Handle Extreme Values")
extreme_method = st.selectbox(
    "Choose method to handle extreme values",
    ["None", "IQR", "Z-score", "Quantiles 1-99"],
)

df_extreme = handle_extreme_values(df_imputed, extreme_method, numeric_cols)

st.write(f"Shape after missing/extreme value handling: {df_extreme.shape}")
st.dataframe(df_extreme.head())

# --- Step 3: Data Encoding ---
st.subheader("Step 3: Data Encoding (Categorical Columns Only)")
if "category" not in df_extreme.columns:
    st.info("'category' column not available for encoding.")
    encoded_df = df_extreme.copy()
else:
    encoding_method = st.selectbox(
        "Choose encoding method for 'category' column",
        ["Label Encoding", "One-Hot Encoding"],
    )
    encoded_df = encode_category(df_extreme, encoding_method)
    if encoding_method == "Label Encoding":
        st.write("Applying label encoding to 'category' column...")
        st.dataframe(encoded_df[["category", "category_label"]].head())
    elif encoding_method == "One-Hot Encoding":
        st.write("Applying one-hot encoding to 'category' column...")
        st.dataframe(encoded_df.head())

# --- Step 4: Scaling ---
st.subheader("Step 4: Scaling Numeric Columns")
scaler_method = st.selectbox(
    "Choose scaling method",
    ["None", "StandardScaler", "MinMaxScaler", "RobustScaler"],
)

scaled_df = scale_numeric(encoded_df, scaler_method, numeric_cols)

if not categorical_cols or not numeric_cols:
    st.warning("Not enough categorical or numeric columns for group-by analysis.")
    st.stop()

# --- User Selections for GroupBy/Aggregation ---
st.subheader("Group-By and Aggregation Settings")
group_col = st.selectbox(
    "Select a categorical column to group by", ["category", "rating"]
)

# Add checkbox for coarse rating grouping if grouping by 'rating'
use_coarse_rating = False
if group_col == "rating":
    use_coarse_rating = st.checkbox(
        "Use coarse rating groups (e.g., [3, 4), [4, 5), ...)",
        value=True,
        help="Group ratings into intervals instead of using exact values.",
    )
    if use_coarse_rating:
        # Define bins (adjust as needed for your data's rating range)
        min_rating = np.floor(scaled_df["rating"].astype(float).min())
        max_rating = np.ceil(scaled_df["rating"].astype(float).max())
        bins = np.arange(min_rating, max_rating + 1, 1)
        labels = [f"[{int(bins[i])}, {int(bins[i+1])})" for i in range(len(bins) - 1)]
        scaled_df["rating_group"] = pd.cut(
            scaled_df["rating"],
            bins=bins,
            labels=labels,
            right=False,
            include_lowest=True,
        )
        group_col_actual = "rating_group"
    else:
        group_col_actual = "rating"
else:
    group_col_actual = group_col

agg_cols = st.multiselect(
    "Select numeric columns to aggregate", numeric_cols, default=numeric_cols[:1]
)
agg_funcs = st.multiselect(
    "Select aggregation functions",
    ["mean", "sum", "min", "max", "count", "median", "std"],
    default=["mean", "count"],
)

if not agg_cols or not agg_funcs:
    st.info("Please select at least one numeric column and one aggregation function.")
    st.stop()

# --- Grouped Data ---
st.subheader(f"Grouped Data: {group_col}")
grouped = scaled_df.groupby(group_col_actual)[agg_cols].agg(agg_funcs)
# Flatten MultiIndex columns if needed
grouped.columns = [f"{col}_{func}" for col, func in grouped.columns]
grouped = grouped.reset_index()
st.dataframe(grouped)

# --- Visualizations ---
st.header("Visualizations of Grouped Data")

# Bar chart for group means (if mean is selected)
if "mean" in agg_funcs:
    st.subheader("Bar Chart: Group Means")
    for col in agg_cols:
        if f"{col}_mean" in grouped.columns:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x=group_col_actual, y=f"{col}_mean", data=grouped, ax=ax)
            ax.set_title(f"Mean of {col} by {group_col}")
            ax.set_ylabel(f"Mean {col}")
            ax.set_xlabel(group_col)
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)

# Boxplot for group distributions
st.subheader("Boxplot: Distribution by Group")
for col in agg_cols:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(x=group_col_actual, y=col, data=scaled_df, ax=ax)
    ax.set_title(f"Distribution of {col} by {group_col}")
    ax.set_ylabel(col)
    ax.set_xlabel(group_col)
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

# Heatmap for group-numeric aggregation (if more than one numeric col and group is not too large)
if len(agg_cols) > 1 and grouped.shape[0] <= 30:
    st.subheader("Heatmap: Aggregated Values by Group")
    # Use mean if available, else first agg func
    func = "mean" if "mean" in agg_funcs else agg_funcs[0]
    heatmap_data = grouped.set_index(group_col_actual)[
        [f"{col}_{func}" for col in agg_cols if f"{col}_{func}" in grouped.columns]
    ]
    fig, ax = plt.subplots(figsize=(2 + len(agg_cols), 1 + 0.3 * len(heatmap_data)))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
    ax.set_title(f"{func.title()} of Numeric Columns by {group_col}")
    st.pyplot(fig)

st.info(
    "Tip: All group-by and aggregation is performed on data that has been cleaned, encoded, and scaled as per your selections above!"
)
