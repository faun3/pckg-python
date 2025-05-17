import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from data_utils import load_data, impute_missing_values, handle_extreme_values, encode_category, scale_numeric

st.header("Data Scaling Methods")

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
    ["StandardScaler", "MinMaxScaler", "RobustScaler"],
)

scaled_df = scale_numeric(encoded_df, scaler_method, numeric_cols)
if numeric_cols:
    st.write(f"Scaled numeric columns using {scaler_method}.")
    st.dataframe(scaled_df[numeric_cols].head())

    # Visualize scaled data
    st.subheader("Visualizations of Scaled Numeric Columns")
    fig, axs = plt.subplots(len(numeric_cols), 2, figsize=(12, 4 * len(numeric_cols)))
    if len(numeric_cols) == 1:
        axs = np.array([axs])
    for i, col in enumerate(numeric_cols):
        sns.histplot(scaled_df[col], bins=30, kde=True, ax=axs[i, 0], color="skyblue")
        axs[i, 0].set_title(f"Histogram of {col} (scaled)")
        sns.boxplot(x=scaled_df[col], ax=axs[i, 1], color="salmon")
        axs[i, 1].set_title(f"Boxplot of {col} (scaled)")
    st.pyplot(fig)
else:
    st.info("No numeric columns available for scaling.") 