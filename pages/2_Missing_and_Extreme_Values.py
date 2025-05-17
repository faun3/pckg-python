import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from scipy import stats

def load_data():
    try:
        df = pd.read_csv('amazon.csv')
        df['discounted_price'] = df['discounted_price'].str.replace('₹', '', regex=False).str.replace(',', '', regex=False).astype(float)
        df['actual_price'] = df['actual_price'].str.replace('₹', '').str.replace(',', '').astype(float)
        df['discount_percentage'] = df['discount_percentage'].str.replace('%', '').astype(float)
        df['rating_count'] = df['rating_count'].str.replace(',', '').astype(float)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.warning("No data loaded.")
    st.stop()

st.header("Handling Missing Values")
missing_values = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100
missing_data = pd.concat([missing_values, missing_percent], axis=1)
missing_data.columns = ['Missing Values', 'Percent (%)']
st.write("Missing values analysis:")
st.dataframe(missing_data[missing_data['Missing Values'] > 0])

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

st.write(f"Numeric columns: {numeric_cols}")
st.write(f"Categorical columns: {categorical_cols}")

st.subheader("Imputation Methods")
impute_method = st.selectbox("Choose imputation method", ["Mean (numeric)", "Most Frequent (categorical)", "KNN (numeric)", "Forward Fill", "Backward Fill", "Drop Rows"])

df_imputed = df.copy()
if impute_method == "Mean (numeric)":
    numeric_imputer = SimpleImputer(strategy='mean')
    if numeric_cols:
        df_imputed[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
    st.write("After mean imputation for numeric columns:")
    st.dataframe(df_imputed[numeric_cols].isnull().sum())
elif impute_method == "Most Frequent (categorical)":
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    if categorical_cols:
        df_imputed[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])
    st.write("After most frequent imputation for categorical columns:")
    st.dataframe(df_imputed[categorical_cols].isnull().sum())
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

st.header("Handling Extreme Values")
st.write("Extreme value detection and handling is performed on the imputed data.")
method = st.selectbox("Choose outlier detection method", ["Z-score", "IQR"])
if method == "Z-score":
    z_scores = {}
    for col in numeric_cols:
        z_scores[col] = np.abs(stats.zscore(df_imputed[col]))
    outlier_counts = {col: int((z_scores[col] > 3).sum()) for col in numeric_cols}
    st.write("Number of outliers (Z-score > 3):")
    st.write(outlier_counts)
elif method == "IQR":
    iqr_outliers = {}
    for col in numeric_cols:
        Q1 = df_imputed[col].quantile(0.25)
        Q3 = df_imputed[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((df_imputed[col] < lower_bound) | (df_imputed[col] > upper_bound)).sum()
        iqr_outliers[col] = int(outliers)
    st.write("Number of outliers (IQR method):")
    st.write(iqr_outliers)

st.subheader("Handle Extreme Values")
handle_method = st.selectbox("Choose method to handle outliers", ["None", "Capping (Winsorization)", "Remove Outliers"])
df_extreme = df_imputed.copy()
if handle_method == "Capping (Winsorization)":
    for col in numeric_cols:
        lower_limit = df_extreme[col].quantile(0.01)
        upper_limit = df_extreme[col].quantile(0.99)
        df_extreme[col] = df_extreme[col].clip(lower=lower_limit, upper=upper_limit)
    st.write("Statistics after capping:")
    st.dataframe(df_extreme[numeric_cols].describe())
elif handle_method == "Remove Outliers":
    for col in numeric_cols:
        df_extreme = df_extreme[np.abs(stats.zscore(df_extreme[col])) <= 3]
    st.write(f"Shape after removing extreme values (Z-score method): {df_extreme.shape}")
else:
    st.write("No outlier handling applied.")
st.subheader("Preview after handling extreme values")
st.dataframe(df_extreme.head()) 