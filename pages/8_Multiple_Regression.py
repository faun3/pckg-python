import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from data_utils import (
    load_data,
    impute_missing_values,
    handle_extreme_values,
    encode_category,
    scale_numeric,
)

st.header("Multiple Regression Analysis (User-Selected Columns)")

# --- Load Data ---
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

# --- Step 5: Select Target and Features ---
st.header("Multiple Regression Setup")
st.markdown("""
Select a numeric target variable and one or more features (numeric or encoded categorical) for multiple regression. Only the first 5 rows of the data are shown below for preview.
""")

# Only allow numeric columns as target
target_col = st.selectbox(
    "Select target variable (numeric):",
    numeric_cols,
)

# Features: all columns except target, and drop obviously non-feature columns
feature_candidates = [col for col in scaled_df.columns if col != target_col and col not in ["product_id", "product_name", "about_product", "user_id", "user_name", "review_id", "review_title", "review_content", "img_link", "product_link"]]

selected_features = st.multiselect(
    "Select features (independent variables):",
    feature_candidates,
    default=[col for col in feature_candidates if np.issubdtype(scaled_df[col].dtype, np.number)][:2],
)

if not selected_features:
    st.warning("Please select at least one feature for regression.")
    st.stop()

# Prepare data
X = scaled_df[selected_features].dropna()
y = scaled_df.loc[X.index, target_col]

if len(X) < 5:
    st.warning("Not enough data after preprocessing for regression.")
    st.stop()

# --- Fit Multiple Regression (OLS) ---
st.subheader("Multiple Regression Results (OLS)")
try:
    X_ = sm.add_constant(X)  # add intercept
    model = sm.OLS(y, X_)
    results = model.fit()
    st.write("**Regression Summary:**")
    st.text(results.summary())

    # Show coefficients
    st.write("**Model Coefficients:**")
    coef_df = pd.DataFrame({
        "Feature": ["Intercept"] + selected_features,
        "Coefficient": results.params.values,
        "P-value": results.pvalues.values,
        "Std Err": results.bse.values,
    })
    st.dataframe(coef_df)

    # Plot actual vs predicted
    st.write("**Actual vs. Predicted Plot:**")
    y_pred = results.predict(X_)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y, y_pred, alpha=0.5)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs. Predicted Values")
    st.pyplot(fig)

    # Residual plot
    st.write("**Residual Plot:**")
    residuals = y - y_pred
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.scatter(y_pred, residuals, alpha=0.5)
    ax2.axhline(0, color='red', linestyle='--')
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Residuals")
    ax2.set_title("Residuals vs. Predicted Values")
    st.pyplot(fig2)

except Exception as e:
    st.error(f"Error fitting multiple regression: {e}") 