import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


def load_data():
    try:
        df = pd.read_csv("amazon.csv")
        df["discounted_price"] = (
            df["discounted_price"]
            .str.replace("₹", "", regex=False)
            .str.replace(",", "", regex=False)
            .astype(float)
        )
        df["actual_price"] = (
            df["actual_price"].str.replace("₹", "").str.replace(",", "").astype(float)
        )
        df["discount_percentage"] = (
            df["discount_percentage"].str.replace("%", "").astype(float)
        )
        df["rating_count"] = df["rating_count"].str.replace(",", "").astype(float)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


df = load_data()

if df.empty:
    st.warning("No data loaded.")
    st.stop()

numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

# --- Visualize Extreme Values for a Selected Column ---
st.header("Visualize Extreme Values for a Selected Column")
if numeric_cols:
    selected_col = st.selectbox("Select a numeric column to visualize", numeric_cols)
    st.write(f"Visualizing extreme values for: {selected_col}")
    col_data = df[selected_col].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # Boxplot
    sns.boxplot(x=col_data, ax=axes[0], color="skyblue")
    axes[0].set_title(f"Boxplot of {selected_col}")
    # Histogram
    sns.histplot(col_data, bins=30, kde=True, ax=axes[1], color="salmon")
    axes[1].set_title(f"Histogram of {selected_col}")
    st.pyplot(fig)
else:
    st.info("No numeric columns available for visualization.")

# --- Visualize missing values heatmap ---
st.subheader("Missing Values Heatmap")
if df.isnull().sum().sum() > 0:
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis", ax=ax)
    ax.set_title("Missing Values Heatmap (yellow = missing)")
    st.pyplot(fig)
else:
    st.write("No missing values detected.")

st.header("Handling Missing Values")
missing_values = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100
missing_data = pd.concat([missing_values, missing_percent], axis=1)
missing_data.columns = ["Missing Values", "Percent (%)"]
st.write("Missing values analysis:")
st.dataframe(missing_data[missing_data["Missing Values"] > 0])

st.write(f"Numeric columns: {numeric_cols}")
st.write(f"Categorical columns: {categorical_cols}")

st.subheader("Imputation Methods")
impute_method = st.selectbox(
    "Choose imputation method",
    [
        "Mean (numeric)",
        "KNN (numeric)",
        "Forward Fill",
        "Backward Fill",
        "Drop Rows",
    ],
)

df_imputed = df.copy()
if impute_method == "Mean (numeric)":
    numeric_imputer = SimpleImputer(strategy="mean")
    if numeric_cols:
        df_imputed[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
    st.write("After mean imputation for numeric columns:")
    st.dataframe(df_imputed[numeric_cols].isnull().sum())
elif impute_method == "KNN (numeric)":
    for col in numeric_cols:
        if np.isinf(df_imputed[col]).any():
            df_imputed[col] = df_imputed[col].replace([np.inf, -np.inf], np.nan)
    knn_imputer = KNNImputer(n_neighbors=5)
    if numeric_cols:
        df_imputed[numeric_cols] = knn_imputer.fit_transform(df_imputed[numeric_cols])
    st.write("After KNN imputation for numeric columns:")
    st.dataframe(df_imputed[numeric_cols].isnull().sum())
elif impute_method == "Forward Fill":
    df_imputed = df_imputed.ffill()
    st.write("After forward fill:")
    st.write(f"Total missing values: {df_imputed.isnull().sum().sum()}")
elif impute_method == "Backward Fill":
    df_imputed = df_imputed.bfill()
    st.write("After backward fill:")
    st.write(f"Total missing values: {df_imputed.isnull().sum().sum()}")
elif impute_method == "Drop Rows":
    df_imputed = df_imputed.dropna()
    st.write(f"Original shape: {df.shape}")
    st.write(f"Shape after dropping rows with missing values: {df_imputed.shape}")

st.subheader("Preview after imputation")
st.dataframe(df_imputed.head())

# --- Handling Extreme Values ---
st.header("Handling Extreme Values")
st.write("Extreme value detection and handling is performed on the imputed data.")
method = st.selectbox(
    "Choose method to handle extreme values",
    ["None", "IQR", "Z-score", "Quantiles 1-99"],
)

df_extreme = df_imputed.copy()
if method == "IQR":
    for col in numeric_cols:
        Q1 = df_extreme[col].quantile(0.25)
        Q3 = df_extreme[col].quantile(0.75)
        df_extreme = df_extreme[(df_extreme[col] >= Q1) & (df_extreme[col] <= Q3)]
elif method == "Z-score":
    for col in numeric_cols:
        df_extreme = df_extreme[np.abs(stats.zscore(df_extreme[col], nan_policy='omit')) <= 3]
elif method == "Quantiles 1-99":
    for col in numeric_cols:
        lower = df_extreme[col].quantile(0.01)
        upper = df_extreme[col].quantile(0.99)
        df_extreme = df_extreme[(df_extreme[col] >= lower) & (df_extreme[col] <= upper)]
# 'None' means do nothing

st.write(f"Shape after handling extreme values: {df_extreme.shape}")
st.subheader("Preview after handling extreme values")
st.dataframe(df_extreme.head())
