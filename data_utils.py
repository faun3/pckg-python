import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from scipy import stats

def load_data():
    df = pd.read_csv("amazon.csv")
    conversion_rate = 85.55  # 1 USD = 85.55 INR
    df["discounted_price"] = (
        df["discounted_price"]
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(float)
    )
    df["discounted_price"] = df["discounted_price"] / conversion_rate

    df["actual_price"] = (
        df["actual_price"].str.replace("₹", "").str.replace(",", "").astype(float)
    )
    df["actual_price"] = df["actual_price"] / conversion_rate

    df["discount_percentage"] = (
        df["discount_percentage"].str.replace("%", "").astype(float)
    )
    
    df["rating_count"] = df["rating_count"].str.replace(",", "").astype(float)
    return df

def impute_missing_values(df, method, numeric_cols):
    df = df.copy()
    if method == "Mean (numeric)":
        imputer = SimpleImputer(strategy="mean")
        if numeric_cols:
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    elif method == "KNN (numeric)":
        for col in numeric_cols:
            if np.isinf(df[col]).any():
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        imputer = KNNImputer(n_neighbors=5)
        if numeric_cols:
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    elif method == "Forward Fill":
        df = df.ffill()
    elif method == "Backward Fill":
        df = df.bfill()
    elif method == "Drop Rows":
        df = df.dropna()
    # 'None' means do nothing
    return df

def handle_extreme_values(df, method, numeric_cols):
    df = df.copy()
    if method == "IQR":
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            df = df[(df[col] >= Q1) & (df[col] <= Q3)]
    elif method == "Z-score":
        for col in numeric_cols:
            df = df[np.abs(stats.zscore(df[col], nan_policy='omit')) <= 3]
    elif method == "Quantiles 1-99":
        for col in numeric_cols:
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            df = df[(df[col] >= lower) & (df[col] <= upper)]
    # 'None' means do nothing
    return df

def encode_category(df, method):
    df = df.copy()
    if "category" not in df.columns:
        return df
    if method == "Label Encoding":
        le = LabelEncoder()
        if df["category"].isnull().sum() > 0:
            df["category"] = df["category"].fillna("missing")
        df["category_label"] = le.fit_transform(df["category"].astype(str))
    elif method == "One-Hot Encoding":
        df = pd.get_dummies(df, columns=["category"], drop_first=True)
    return df

def scale_numeric(df, method, numeric_cols):
    df = df.copy()
    if not numeric_cols or method == "None":
        return df
    if method == "StandardScaler":
        scaler = StandardScaler()
    elif method == "MinMaxScaler":
        scaler = MinMaxScaler()
    elif method == "RobustScaler":
        scaler = RobustScaler()
    else:
        return df
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df 