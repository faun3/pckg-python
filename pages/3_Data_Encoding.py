import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

st.header("Data Encoding Methods")


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

# --- Step 1: Handle Missing Values ---
st.subheader("Step 1: Handle Missing Values")

# Dropdown for numeric columns
impute_method_numeric = st.selectbox(
    "How were missing values in numeric columns handled?",
    ["Mean", "KNN", "Forward Fill", "Backward Fill", "Drop Rows", "None"],
)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

df_imputed = df.copy()

# Handle numeric columns
if impute_method_numeric == "Mean":
    numeric_imputer = SimpleImputer(strategy="mean")
    if numeric_cols:
        df_imputed[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
elif impute_method_numeric == "KNN":
    for col in numeric_cols:
        if np.isinf(df_imputed[col]).any():
            df_imputed[col] = df_imputed[col].replace([np.inf, -np.inf], np.nan)
    knn_imputer = KNNImputer(n_neighbors=5)
    if numeric_cols:
        df_imputed[numeric_cols] = knn_imputer.fit_transform(df_imputed[numeric_cols])
elif impute_method_numeric == "Forward Fill":
    df_imputed[numeric_cols] = df_imputed[numeric_cols].ffill()
elif impute_method_numeric == "Backward Fill":
    df_imputed[numeric_cols] = df_imputed[numeric_cols].bfill()
elif impute_method_numeric == "Drop Rows":
    df_imputed = df_imputed.dropna(subset=numeric_cols)
# 'None' means do nothing for numeric columns

# --- Step 2: Choose how extreme values were handled ---
st.subheader("Step 2: Handle Extreme Values")
extreme_method = st.selectbox(
    "How were extreme values handled?",
    ["None", "IQR (Interquartile Range)", "Z-score", "Quantiles 1-99"],
)

df_extreme = df_imputed.copy()
if extreme_method == "IQR":
    for col in numeric_cols:
        Q1 = df_extreme[col].quantile(0.25)
        Q3 = df_extreme[col].quantile(0.75)
        df_extreme = df_extreme[(df_extreme[col] >= Q1) & (df_extreme[col] <= Q3)]
elif extreme_method == "Z-score":
    from scipy import stats
    for col in numeric_cols:
        df_extreme = df_extreme[np.abs(stats.zscore(df_extreme[col], nan_policy='omit')) <= 3]
elif extreme_method == "Quantiles 1-99":
    for col in numeric_cols:
        lower = df_extreme[col].quantile(0.01)
        upper = df_extreme[col].quantile(0.99)
        df_extreme = df_extreme[(df_extreme[col] >= lower) & (df_extreme[col] <= upper)]
# 'None' means do nothing

st.write(f"Shape of data after extreme value handling: {df_extreme.shape}")

st.write("Preview of data after missing and extreme value handling:")
st.dataframe(df_extreme.head())

# --- Step 3: Data Encoding ---
st.subheader("Step 3: Data Encoding (Categorical Columns Only)")
if "category" not in df_extreme.columns:
    st.info("'category' column not available for encoding.")
else:
    encoding_method = st.selectbox(
        "Choose encoding method for 'category' column",
        ["Label Encoding", "One-Hot Encoding"],
    )

    encoded_df = df_extreme.copy()

    if encoding_method == "Label Encoding":
        st.write("Applying label encoding to 'category' column...")
        le = LabelEncoder()
        if encoded_df["category"].isnull().sum() > 0:
            encoded_df["category"] = encoded_df["category"].fillna("missing")
        encoded_df["category_label"] = le.fit_transform(
            encoded_df["category"].astype(str)
        )
        st.dataframe(encoded_df[["category", "category_label"]].head())
        # Plot distribution of label-encoded column
        fig, ax = plt.subplots()
        sns.histplot(encoded_df["category_label"], bins=20, ax=ax)
        ax.set_title("Distribution of category_label")
        st.pyplot(fig)

    elif encoding_method == "One-Hot Encoding":
        st.write("Applying one-hot encoding to 'category' column...")
        encoded_df = pd.get_dummies(encoded_df, columns=["category"], drop_first=True)
        st.dataframe(encoded_df.head())
        # Show heatmap of one-hot encoded columns
        onehot_cols = [c for c in encoded_df.columns if c.startswith("category_")]
        if onehot_cols:
            st.write("Correlation heatmap of one-hot encoded 'category' columns:")
            corr = encoded_df[onehot_cols].corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

    # --- Statistics ---
    st.subheader("Statistics of Encoded Data")
    st.write(encoded_df.describe(include="all"))
