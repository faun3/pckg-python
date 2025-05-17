import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from data_utils import load_data, impute_missing_values, handle_extreme_values

st.header("Visualize Extreme Values for a Selected Column")

df = load_data()

if df.empty:
    st.warning("No data loaded.")
    st.stop()

numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

# --- Visualize Extreme Values for a Selected Column ---
if numeric_cols:
    selected_col = st.selectbox("Select a numeric column to visualize", numeric_cols)
    st.write(f"Visualizing extreme values for: {selected_col}")
    col_data = df[selected_col].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(x=col_data, ax=axes[0], color="skyblue")
    axes[0].set_title(f"Boxplot of {selected_col}")
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
missing_data = missing_values.to_frame("Missing Values")
missing_data["Percent (%)"] = missing_percent
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

df_imputed = impute_missing_values(df, impute_method, numeric_cols)
if impute_method in ["Mean (numeric)", "KNN (numeric)"]:
    st.write(f"After {impute_method.lower()} for numeric columns:")
    st.dataframe(df_imputed[numeric_cols].isnull().sum())
elif impute_method in ["Forward Fill", "Backward Fill"]:
    st.write(f"After {impute_method.lower()}:")
    st.write(f"Total missing values: {df_imputed.isnull().sum().sum()}")
elif impute_method == "Drop Rows":
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

df_extreme = handle_extreme_values(df_imputed, method, numeric_cols)

st.write(f"Shape after handling extreme values: {df_extreme.shape}")
st.subheader("Preview after handling extreme values")
st.dataframe(df_extreme.head())
