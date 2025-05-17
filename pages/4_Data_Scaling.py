import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

st.header("Data Scaling Methods")

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

df_imputed = df.copy()
if impute_method == "Mean (numeric)":
    numeric_imputer = SimpleImputer(strategy="mean")
    if numeric_cols:
        df_imputed[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
elif impute_method == "KNN (numeric)":
    for col in numeric_cols:
        if np.isinf(df_imputed[col]).any():
            df_imputed[col] = df_imputed[col].replace([np.inf, -np.inf], np.nan)
    knn_imputer = KNNImputer(n_neighbors=5)
    if numeric_cols:
        df_imputed[numeric_cols] = knn_imputer.fit_transform(df_imputed[numeric_cols])
elif impute_method == "Forward Fill":
    df_imputed = df_imputed.ffill()
elif impute_method == "Backward Fill":
    df_imputed = df_imputed.bfill()
elif impute_method == "Drop Rows":
    df_imputed = df_imputed.dropna()
# 'None' means do nothing

# --- Step 2: Handle Extreme Values ---
st.subheader("Step 2: Handle Extreme Values")
extreme_method = st.selectbox(
    "Choose method to handle extreme values",
    ["None", "IQR", "Z-score", "Quantiles 1-99"],
)

df_extreme = df_imputed.copy()
if extreme_method == "IQR":
    for col in numeric_cols:
        Q1 = df_extreme[col].quantile(0.25)
        Q3 = df_extreme[col].quantile(0.75)
        df_extreme = df_extreme[(df_extreme[col] >= Q1) & (df_extreme[col] <= Q3)]
elif extreme_method == "Z-score":
    for col in numeric_cols:
        df_extreme = df_extreme[np.abs(stats.zscore(df_extreme[col], nan_policy='omit')) <= 3]
elif extreme_method == "Quantiles 1-99":
    for col in numeric_cols:
        lower = df_extreme[col].quantile(0.01)
        upper = df_extreme[col].quantile(0.99)
        df_extreme = df_extreme[(df_extreme[col] >= lower) & (df_extreme[col] <= upper)]
# 'None' means do nothing

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
    elif encoding_method == "One-Hot Encoding":
        st.write("Applying one-hot encoding to 'category' column...")
        encoded_df = pd.get_dummies(encoded_df, columns=["category"], drop_first=True)
        st.dataframe(encoded_df.head())

# --- Step 4: Scaling ---
st.subheader("Step 4: Scaling Numeric Columns")
scaler_method = st.selectbox(
    "Choose scaling method",
    ["StandardScaler", "MinMaxScaler", "RobustScaler"],
)

scaled_df = encoded_df.copy()
if numeric_cols:
    if scaler_method == "StandardScaler":
        scaler = StandardScaler()
    elif scaler_method == "MinMaxScaler":
        scaler = MinMaxScaler()
    elif scaler_method == "RobustScaler":
        scaler = RobustScaler()
    scaled_df[numeric_cols] = scaler.fit_transform(scaled_df[numeric_cols])
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