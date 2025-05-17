import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from data_utils import (
    load_data,
    impute_missing_values,
    handle_extreme_values,
    encode_category,
    scale_numeric,
)

st.header("Clustering Analysis (User-Selected Columns)")

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

# --- Step 2: Handle Extreme Values ---
st.subheader("Step 2: Handle Extreme Values")
extreme_method = st.selectbox(
    "Choose method to handle extreme values",
    ["None", "IQR", "Z-score", "Quantiles 1-99"],
)

df_imputed = impute_missing_values(df, impute_method, numeric_cols)
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

# --- Step 5: User selects columns for clustering ---
st.header("Clustering by User-Selected Columns")

available_columns = [
    "category",
    "rating",
    "discounted_price",
    "actual_price",
    "discount_percentage",
    "rating_count",
]

selected_cols = st.multiselect(
    "Select TWO columns to cluster by:",
    available_columns,
    default=[col for col in ["category", "rating"] if col in scaled_df.columns],
    max_selections=2,
)

if len(selected_cols) != 2:
    st.warning("Please select exactly two columns for clustering.")
    st.stop()

# Prepare features for clustering
features = []
for col in selected_cols:
    if col == "category":
        if "category_label" in scaled_df.columns:
            features.append("category_label")
        elif any(c.startswith("category_") for c in scaled_df.columns):
            features += [c for c in scaled_df.columns if c.startswith("category_")]
    else:
        if col in scaled_df.columns:
            features.append(col)

if not features or len(scaled_df) < 2:
    st.warning("Not enough data or features for clustering.")
    st.stop()

X = scaled_df[features].dropna()

n_clusters = st.slider("Select number of clusters", min_value=2, max_value=6, value=3)

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

scaled_df = scaled_df.loc[X.index].copy()
scaled_df["cluster"] = labels

st.subheader("Clustered Data Preview")
st.dataframe(scaled_df[[*features, "cluster"]].head())

# --- PCA for visualization ---
st.subheader("PCA Cluster Visualization (2D)")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

fig, ax = plt.subplots(figsize=(8, 5))
scatter = ax.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=scaled_df["cluster"],
    cmap="Set1",
    alpha=0.8,
    edgecolor=None,
    s=60
)
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.set_title(
    f"Clusters visualized in PCA space (selected: {selected_cols[0]}, {selected_cols[1]})"
)
legend1 = ax.legend(*scatter.legend_elements(), title="Cluster")
ax.add_artist(legend1)
st.pyplot(fig)
