import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from data_utils import (
    load_data,
    impute_missing_values,
    handle_extreme_values,
    encode_category,
    scale_numeric,
)
import pandas as pd

st.header("Logistic Regression Analysis (User-Selected Columns)")

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

# --- Step 5: Define 'Bad' Products ---
st.header("Are bad products the only ones that go on big sales?")
st.markdown("""
We define a 'bad' product as one with a low rating and/or a low number of ratings. 
You can adjust the thresholds below. We'll use logistic regression to see if high discounts are associated with 'bad' products.
""")

# Let user select thresholds
def_rating = 3.5 if 'rating' in scaled_df.columns else None
def_rating_count = int(scaled_df['rating_count'].median()) if 'rating_count' in scaled_df.columns else None

rating_threshold = st.number_input(
    "Rating threshold (products with rating BELOW this are considered 'bad'):",
    min_value=0.0, max_value=5.0, value=def_rating or 3.5, step=0.1,
    disabled='rating' not in scaled_df.columns
)
rating_count_threshold = st.number_input(
    "Rating count threshold (products with rating_count BELOW this are considered 'bad'):",
    min_value=0, value=def_rating_count or 0, step=1,
    disabled='rating_count' not in scaled_df.columns
)

# Create binary target
def is_bad(row):
    bad = False
    if 'rating' in row and not np.isnan(row['rating']):
        bad = bad or (row['rating'] < rating_threshold)
    if 'rating_count' in row and not np.isnan(row['rating_count']):
        bad = bad or (row['rating_count'] < rating_count_threshold)
    return int(bad)

scaled_df['bad_product'] = scaled_df.apply(is_bad, axis=1)
st.write(f"Number of 'bad' products: {scaled_df['bad_product'].sum()} / {len(scaled_df)}")
st.dataframe(scaled_df[['rating', 'rating_count', 'discount_percentage', 'bad_product']].head())

# --- Step 6: Select features (mainly discount_percentage) ---
main_features = ['discount_percentage']
other_features = [col for col in scaled_df.columns if col not in main_features + ['bad_product', 'category', 'category_label', 'rating', 'rating_count'] and np.issubdtype(scaled_df[col].dtype, np.number)]

selected_features = st.multiselect(
    "Select additional features (optional):",
    other_features,
    default=[],
)
features = main_features + selected_features

# Prepare data
X = scaled_df[features].dropna()
y = scaled_df.loc[X.index, 'bad_product']

if y.nunique() < 2:
    st.warning("Not enough variation in 'bad_product' to fit logistic regression.")
    st.stop()

# --- Fit Logistic Regression ---
st.subheader("Logistic Regression Results: Predicting 'Bad' Products from Discount Percentage")
try:
    model = LogisticRegression(max_iter=200, multi_class="auto", solver="lbfgs")
    model.fit(X, y)
    y_pred = model.predict(X)
    st.write("**Confusion Matrix:**")
    fig, ax = plt.subplots(figsize=(6, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix(y, y_pred), display_labels=["Good", "Bad"])
    disp.plot(ax=ax, cmap="Blues")
    st.pyplot(fig)

    # ROC Curve (only for binary classification)
    if y.nunique() == 2:
        from sklearn.metrics import roc_curve, auc
        y_score = model.predict_proba(X)[:, 1]
        fpr, tpr, thresholds = roc_curve(y, y_score)
        roc_auc = auc(fpr, tpr)
        st.subheader("ROC Curve (Binary Classification)")
        fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
        ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f}')
        ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('Receiver Operating Characteristic (ROC)')
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)

    # Precision, Recall, Accuracy
    from sklearn.metrics import precision_score, recall_score, accuracy_score
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    accuracy = accuracy_score(y, y_pred)
    st.write(f"**Precision:** {precision:.2f}")
    st.write(f"**Recall:** {recall:.2f}")
    st.write(f"**Accuracy:** {accuracy:.2f}")

    # Classification Report
    st.write("**Classification Report:**")
    st.text(classification_report(y, y_pred))

    # Show coefficients
    st.write("**Model Coefficients:**")
    coef_df = pd.DataFrame({
        "Feature": features,
        "Coefficient": model.coef_[0]
    })
    st.dataframe(coef_df)

    # Interpretation
    st.markdown("""
    **Interpretation:**
    - If the coefficient for `discount_percentage` is positive and significant, it suggests that higher discounts are associated with a higher likelihood of a product being 'bad'.
    - If the coefficient is negative, it suggests the opposite.
    - Consider the magnitude and sign of the coefficient for `discount_percentage` to answer the question: *Are bad products the only ones that go on big sales?*
    """)
except Exception as e:
    st.error(f"Error fitting logistic regression: {e}") 