import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

st.header("Basic Dataset Information")
st.write(f"**Dataset shape:** {df.shape}")
st.write("**First 5 rows:**")
st.dataframe(df.head())

with st.expander("Show data types"):
    st.write(df.dtypes)

with st.expander("Show missing values per column"):
    st.write(df.isnull().sum())

with st.expander("Show basic statistics"):
    st.write(df.describe())

st.header("Visualizations")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribution of Discounted Price")
    st.bar_chart(df['discounted_price'].dropna())

with col2:
    st.subheader("Distribution of Actual Price")
    st.bar_chart(df['actual_price'].dropna())

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
if categorical_cols:
    st.header("Categorical Columns Value Counts")
    cat_col = st.selectbox("Select a categorical column", categorical_cols)
    st.write(df[cat_col].value_counts())

# --- Additional Visualizations ---
st.header("More Visualizations")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Histograms for all numeric columns
st.subheader("Histograms of Numeric Columns")
fig, axs = plt.subplots(len(numeric_cols), 1, figsize=(6, 3*len(numeric_cols)))
if len(numeric_cols) == 1:
    axs = [axs]
for i, col in enumerate(numeric_cols):
    axs[i].hist(df[col].dropna(), bins=30, color='skyblue', edgecolor='black')
    axs[i].set_title(f'Histogram of {col}')
    axs[i].set_xlabel(col)
    axs[i].set_ylabel('Frequency')
st.pyplot(fig)

# Boxplots for all numeric columns
st.subheader("Boxplots of Numeric Columns")
fig2, axs2 = plt.subplots(1, len(numeric_cols), figsize=(4*len(numeric_cols), 4))
if len(numeric_cols) == 1:
    axs2 = [axs2]
for i, col in enumerate(numeric_cols):
    axs2[i].boxplot(df[col].dropna(), vert=True)
    axs2[i].set_title(f'Boxplot of {col}')
    axs2[i].set_ylabel(col)
st.pyplot(fig2)

# Correlation heatmap
if len(numeric_cols) > 1:
    st.subheader("Correlation Heatmap")
    corr = df[numeric_cols].corr()
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax3)
    st.pyplot(fig3)

# Pairplot (sampled for performance)
if len(numeric_cols) > 1 and len(df) > 10:
    st.subheader("Pairplot of Numeric Columns (sampled)")
    sample_df = df[numeric_cols].dropna().sample(min(200, len(df)), random_state=42)
    fig4 = sns.pairplot(sample_df)
    st.pyplot(fig4)