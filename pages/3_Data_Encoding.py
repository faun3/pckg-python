import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from data_utils import load_data, impute_missing_values, handle_extreme_values, encode_category

st.header("Data Encoding Methods")

df = load_data()

if df.empty:
    st.warning("No data loaded.")
    st.stop()

# --- Step 1: Handle Missing Values ---
st.subheader("Step 1: Handle Missing Values")
impute_method_numeric = st.selectbox(
    "How were missing values in numeric columns handled?",
    ["Mean (numeric)", "KNN (numeric)", "Forward Fill", "Backward Fill", "Drop Rows", "None"],
)

numeric_cols = df.select_dtypes(include=[float, int]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

df_imputed = impute_missing_values(df, impute_method_numeric, numeric_cols)

# --- Step 2: Handle Extreme Values ---
st.subheader("Step 2: Handle Extreme Values")
extreme_method = st.selectbox(
    "How were extreme values handled?",
    ["None", "IQR", "Z-score", "Quantiles 1-99"],
)

df_extreme = handle_extreme_values(df_imputed, extreme_method, numeric_cols)

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
    encoded_df = encode_category(df_extreme, encoding_method)
    if encoding_method == "Label Encoding":
        st.write("Applying label encoding to 'category' column...")
        st.dataframe(encoded_df[["category", "category_label"]].head())
        st.info(f"Number of categories: {encoded_df['category'].nunique()}")
        # Plot distribution of label-encoded column
        fig, ax = plt.subplots()
        sns.histplot(encoded_df["category_label"], bins=20, ax=ax)
        ax.set_title("Distribution of category_label")
        st.pyplot(fig)
    elif encoding_method == "One-Hot Encoding":
        st.write("Applying one-hot encoding to 'category' column...")
        st.dataframe(encoded_df.head())
        onehot_cols = [c for c in encoded_df.columns if c.startswith("category_")]
        st.info(f"Number of categories (one-hot columns): {len(onehot_cols)}")
        # Show heatmap of one-hot encoded columns
        if onehot_cols:
            st.write("Correlation heatmap of one-hot encoded 'category' columns:")
            corr = encoded_df[onehot_cols].corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
    st.subheader("Statistics of Encoded Data")
    st.write(encoded_df.describe(include="all"))
