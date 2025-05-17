import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from scipy import stats
import re

# Set display options for better output
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', 20)


# Load data
def load_data():
    try:
        # Load data from CSV
        df = pd.read_csv('amazon.csv')

        # Clean price columns - remove currency symbols and convert to numeric
        df['discounted_price'] = df['discounted_price'].str.replace('₹', '', regex=False).str.replace(',', '', regex=False).astype(float)
        df['actual_price'] = df['actual_price'].str.replace('₹', '').str.replace(',', '').astype(float)
        df['discount_percentage'] = df['discount_percentage'].str.replace('%', '').astype(float)

        # Convert rating_count to numeric
        df['rating_count'] = df['rating_count'].str.replace(',', '').astype(float)

        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()


# Load the data
df = load_data()
print(df)

# Display basic information
print("=" * 80)
print("BASIC DATASET INFORMATION")
print("=" * 80)
print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nMissing values per column:")
print(df.isnull().sum())
print("\nBasic statistics:")
print(df.describe())

# ======================================================================
# PART 1: HANDLING MISSING VALUES
# ======================================================================
print("\n" + "=" * 80)
print("PART 1: HANDLING MISSING VALUES")
print("=" * 80)

# Check for missing values
missing_values = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100
missing_data = pd.concat([missing_values, missing_percent], axis=1)
missing_data.columns = ['Missing Values', 'Percent (%)']
print("\nMissing values analysis:")
print(missing_data[missing_data['Missing Values'] > 0])

# Separate numerical and categorical columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumeric columns: {numeric_cols}")
print(f"Categorical columns: {categorical_cols}")

# 1.1 Simple imputation for numeric columns
print("\n1.1 Simple imputation for numeric columns")
# Make a copy of the dataframe
df_imputed = df.copy()

# Apply mean imputation to numeric columns
numeric_imputer = SimpleImputer(strategy='mean')
if numeric_cols:
    df_imputed[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])

print("After mean imputation for numeric columns:")
print(df_imputed[numeric_cols].isnull().sum())

# 1.2 Simple imputation for categorical columns
print("\n1.2 Simple imputation for categorical columns")
# Apply most frequent imputation to categorical columns
categorical_imputer = SimpleImputer(strategy='most_frequent')
if categorical_cols:
    df_imputed[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])

print("After most frequent imputation for categorical columns:")
print(df_imputed[categorical_cols].isnull().sum())

# 1.3 KNN imputation for numeric columns
print("\n1.3 KNN imputation for numeric columns")
# Create a copy of the dataframe
df_knn_imputed = df.copy()

# Apply KNN imputation to numeric columns
if numeric_cols:
    # First ensure there are no infinite values
    for col in numeric_cols:
        if np.isinf(df_knn_imputed[col]).any():
            df_knn_imputed[col] = df_knn_imputed[col].replace([np.inf, -np.inf], np.nan)

    # Apply KNN imputation
    knn_imputer = KNNImputer(n_neighbors=5)
    df_knn_imputed[numeric_cols] = knn_imputer.fit_transform(df_knn_imputed[numeric_cols])

print("After KNN imputation for numeric columns:")
print(df_knn_imputed[numeric_cols].isnull().sum())

# 1.4 Forward fill and backward fill
print("\n1.4 Forward fill and backward fill")
# Create a copy of the dataframe
df_fill = df.copy()

# Apply forward fill
df_ffill = df_fill.ffill()
print("After forward fill:")
print(df_ffill.isnull().sum().sum())

# Apply backward fill
df_bfill = df_fill.bfill()
print("After backward fill:")
print(df_bfill.isnull().sum().sum())

# 1.5 Dropping rows with missing values
print("\n1.5 Dropping rows with missing values")
# Create a copy of the dataframe
df_dropped = df.copy()

# Drop rows with missing values
df_dropped = df_dropped.dropna()
print(f"Original shape: {df.shape}")
print(f"Shape after dropping rows with missing values: {df_dropped.shape}")

# ======================================================================
# PART 2: HANDLING EXTREME VALUES
# ======================================================================
print("\n" + "=" * 80)
print("PART 2: HANDLING EXTREME VALUES")
print("=" * 80)

# Let's use the imputed dataframe for handling extreme values
df_clean = df_imputed.copy()

# 2.1 Identify extreme values using Z-score
print("\n2.1 Identify extreme values using Z-score")
# Calculate Z-score for numeric columns
z_scores = {}
for col in numeric_cols:
    z_scores[col] = np.abs(stats.zscore(df_clean[col]))

# Print number of extreme values using Z-score > 3 as threshold
for col in numeric_cols:
    num_outliers = np.sum(z_scores[col] > 3)
    percent_outliers = (num_outliers / len(df_clean)) * 100
    print(f"{col}: {num_outliers} outliers ({percent_outliers:.2f}%)")

# 2.2 Identify extreme values using IQR method
print("\n2.2 Identify extreme values using IQR method")
# Calculate IQR for numeric columns
iqr_outliers = {}
for col in numeric_cols:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
    iqr_outliers[col] = outliers
    percent_outliers = (outliers / len(df_clean)) * 100
    print(f"{col}: {outliers} outliers ({percent_outliers:.2f}%)")

# 2.3 Handle extreme values by capping
print("\n2.3 Handle extreme values by capping (Winsorization)")
# Create a copy of the dataframe
df_capped = df_clean.copy()

# Cap extreme values using percentile method (Winsorization)
for col in numeric_cols:
    # Calculate the upper and lower limits
    lower_limit = df_capped[col].quantile(0.01)  # 1st percentile
    upper_limit = df_capped[col].quantile(0.99)  # 99th percentile

    # Cap the values
    df_capped[col] = df_capped[col].clip(lower=lower_limit, upper=upper_limit)

print("Statistics after capping:")
print(df_capped[numeric_cols].describe())

# 2.4 Handle extreme values by removing
print("\n2.4 Handle extreme values by removing")
# Create a copy of the dataframe
df_removed_outliers = df_clean.copy()

# Remove extreme values based on Z-score
for col in numeric_cols:
    df_removed_outliers = df_removed_outliers[np.abs(stats.zscore(df_removed_outliers[col])) <= 3]

print(f"Original shape: {df_clean.shape}")
print(f"Shape after removing extreme values (Z-score method): {df_removed_outliers.shape}")

# 2.5 Handle extreme values using log transformation
print("\n2.5 Handle extreme values using log transformation")
# Create a copy of the dataframe
df_log_transform = df_clean.copy()

# Apply log transformation to numeric columns with positive values
for col in numeric_cols:
    # Check if all values are positive
    if (df_log_transform[col] > 0).all():
        df_log_transform[col + '_log'] = np.log(df_log_transform[col])
    # If there are zeros, add a small value before log
    elif (df_log_transform[col] >= 0).all():
        df_log_transform[col + '_log'] = np.log(df_log_transform[col] + 1)
    # For columns with negative values, we'll skip log transformation
    else:
        print(f"Skipping log transformation for {col} as it contains negative values")

# Print statistics for log-transformed columns
log_cols = [col for col in df_log_transform.columns if col.endswith('_log')]
if log_cols:
    print("Statistics after log transformation:")
    print(df_log_transform[log_cols].describe())
else:
    print("No columns were log-transformed.")

# ======================================================================
# PART 3: DATA ENCODING METHODS
# ======================================================================
print("\n" + "=" * 80)
print("PART 3: DATA ENCODING METHODS")
print("=" * 80)

# Let's use the cleaned dataframe for encoding
df_encoded = df_capped.copy()

# 3.1 Label Encoding
print("\n3.1 Label Encoding")
# Apply label encoding to categorical columns
label_encoders = {}
df_label_encoded = df_encoded.copy()

for col in categorical_cols:
    le = LabelEncoder()
    # Check if the column has any missing values
    if df_label_encoded[col].isnull().sum() > 0:
        # Fill missing values with a placeholder
        df_label_encoded[col] = df_label_encoded[col].fillna('missing')

    # Apply label encoding
    df_label_encoded[col + '_label'] = le.fit_transform(df_label_encoded[col])
    label_encoders[col] = le

    # Print unique values and their encodings
    unique_values = df_label_encoded[col].unique()[:5]  # Show first 5 unique values
    for value in unique_values:
        if value == 'missing':
            continue
        encoded_value = le.transform([value])[0]
        print(f"{col}: '{value}' -> {encoded_value}")

# 3.2 One-Hot Encoding
print("\n3.2 One-Hot Encoding")
# Apply one-hot encoding to categorical columns
df_onehot = df_encoded.copy()

# Select a subset of categorical columns for demonstration
categorical_subset = categorical_cols[:3]  # Take first 3 categorical columns
print(f"Applying one-hot encoding to: {categorical_subset}")

# Apply one-hot encoding
df_onehot = pd.get_dummies(df_onehot, columns=categorical_subset, drop_first=True)

# Print shape before and after one-hot encoding
print(f"Shape before one-hot encoding: {df_encoded.shape}")
print(f"Shape after one-hot encoding: {df_onehot.shape}")
print(f"New columns created: {df_onehot.shape[1] - df_encoded.shape[1]}")

# Show the first few one-hot encoded columns
onehot_cols = [col for col in df_onehot.columns if col not in df_encoded.columns]
print("First few one-hot encoded columns:")
if onehot_cols:
    print(df_onehot[onehot_cols[:5]].head())
else:
    print("No one-hot encoded columns created.")

# 3.3 Binary Encoding
print("\n3.3 Binary Encoding (Custom implementation)")
# We'll implement a simple binary encoding for categorical columns
df_binary = df_encoded.copy()


def binary_encode(column):
    """Convert categorical values to binary representation"""
    unique_values = column.unique()
    encoding_map = {val: format(i, 'b').zfill(len(bin(len(unique_values))[2:]))
                    for i, val in enumerate(unique_values)}

    # Create binary columns
    result_df = pd.DataFrame()
    for i in range(len(bin(len(unique_values))[2:])):
        col_name = f"{column.name}_bin_{i}"
        result_df[col_name] = column.map(lambda x: int(encoding_map[x][i]))

    return result_df


# Apply binary encoding to first categorical column as an example
if categorical_cols:
    example_col = categorical_cols[0]
    print(f"Applying binary encoding to: {example_col}")

    # If column has missing values, fill them first
    if df_binary[example_col].isnull().sum() > 0:
        df_binary[example_col] = df_binary[example_col].fillna('missing')

    # Apply binary encoding
    binary_cols = binary_encode(df_binary[example_col])
    df_binary = pd.concat([df_binary, binary_cols], axis=1)

    # Print example of binary encoding
    unique_values = df_binary[example_col].unique()[:3]  # Show first 3 unique values
    for value in unique_values:
        if value == 'missing':
            continue
        print(f"{example_col}: '{value}' -> ", end="")
        for i in range(binary_cols.shape[1]):
            val = df_binary.loc[df_binary[example_col] == value, f"{example_col}_bin_{i}"].iloc[0]
            print(val, end="")
        print()

# 3.4 Ordinal Encoding (Custom implementation)
print("\n3.4 Ordinal Encoding (Custom implementation)")
# We'll implement a custom ordinal encoding for specific columns
df_ordinal = df_encoded.copy()

# Let's say we want to encode the 'rating' column as an ordinal feature
if 'rating' in df_ordinal.columns:
    print("Applying ordinal encoding to 'rating' column")

    # Define the mapping (higher rating = higher value)
    rating_map = {
        1.0: 1,
        1.5: 2,
        2.0: 3,
        2.5: 4,
        3.0: 5,
        3.5: 6,
        4.0: 7,
        4.5: 8,
        5.0: 9
    }

    # Apply the mapping
    df_ordinal['rating_ordinal'] = df_ordinal['rating'].map(rating_map)

    # Print the mapping
    for k, v in rating_map.items():
        print(f"Rating {k} -> {v}")

    # Print the result
    print("Original 'rating' and ordinal encoding:")
    print(df_ordinal[['rating', 'rating_ordinal']].head())

# 3.5 Feature Hashing
print("\n3.5 Feature Hashing (Custom implementation)")
# We'll implement a simple feature hashing for categorical columns
df_hash = df_encoded.copy()


def feature_hash(column, n_features=8):
    """Apply feature hashing to a categorical column"""
    result_df = pd.DataFrame()
    for i in range(n_features):
        col_name = f"{column.name}_hash_{i}"
        result_df[col_name] = column.map(lambda x: hash(str(x)) % 2)  # Simple binary hash
    return result_df


# Apply feature hashing to first categorical column as an example
if categorical_cols:
    example_col = categorical_cols[0]
    print(f"Applying feature hashing to: {example_col}")

    # If column has missing values, fill them first
    if df_hash[example_col].isnull().sum() > 0:
        df_hash[example_col] = df_hash[example_col].fillna('missing')

    # Apply feature hashing
    hash_cols = feature_hash(df_hash[example_col], n_features=4)
    df_hash = pd.concat([df_hash, hash_cols], axis=1)

    # Print example of feature hashing
    unique_values = df_hash[example_col].unique()[:3]  # Show first 3 unique values
    for value in unique_values:
        if value == 'missing':
            continue
        print(f"{example_col}: '{value}' -> ", end="")
        for i in range(hash_cols.shape[1]):
            val = df_hash.loc[df_hash[example_col] == value, f"{example_col}_hash_{i}"].iloc[0]
            print(val, end="")
        print()

# ======================================================================
# PART 4: DATA SCALING METHODS
# ======================================================================
print("\n" + "=" * 80)
print("PART 4: DATA SCALING METHODS")
print("=" * 80)

# Let's use the cleaned dataframe for scaling
df_scaling = df_capped.copy()

# Select numeric columns for scaling
numeric_cols_to_scale = [col for col in numeric_cols if col not in ['product_id', 'index']]
print(f"Columns to scale: {numeric_cols_to_scale}")

# Show original statistics
print("\nOriginal statistics:")
print(df_scaling[numeric_cols_to_scale].describe().T[['min', 'max', 'mean', 'std']])

# 4.1 Standard Scaling (Z-score normalization)
print("\n4.1 Standard Scaling (Z-score normalization)")
# Apply standard scaling to numeric columns
df_standard = df_scaling.copy()
standard_scaler = StandardScaler()
df_standard[numeric_cols_to_scale] = standard_scaler.fit_transform(df_standard[numeric_cols_to_scale])

# Print statistics after standard scaling
print("Statistics after standard scaling:")
print(df_standard[numeric_cols_to_scale].describe().T[['min', 'max', 'mean', 'std']])

# 4.2 Min-Max Scaling
print("\n4.2 Min-Max Scaling")
# Apply min-max scaling to numeric columns
df_minmax = df_scaling.copy()
minmax_scaler = MinMaxScaler()
df_minmax[numeric_cols_to_scale] = minmax_scaler.fit_transform(df_minmax[numeric_cols_to_scale])

# Print statistics after min-max scaling
print("Statistics after min-max scaling:")
print(df_minmax[numeric_cols_to_scale].describe().T[['min', 'max', 'mean', 'std']])

# 4.3 Robust Scaling (using quartiles)
print("\n4.3 Robust Scaling (using quartiles)")
# Apply robust scaling to numeric columns
df_robust = df_scaling.copy()
robust_scaler = RobustScaler()
df_robust[numeric_cols_to_scale] = robust_scaler.fit_transform(df_robust[numeric_cols_to_scale])

# Print statistics after robust scaling
print("Statistics after robust scaling:")
print(df_robust[numeric_cols_to_scale].describe().T[['min', 'max', 'mean', 'std']])

# 4.4 Log Scaling
print("\n4.4 Log Scaling")
# Apply log scaling to numeric columns
df_log_scale = df_scaling.copy()

# Apply log transformation (add 1 to avoid log(0))
for col in numeric_cols_to_scale:
    # Skip columns with negative values
    if (df_log_scale[col] < 0).any():
        print(f"Skipping log scaling for {col} as it contains negative values")
        continue

    # Apply log scaling
    df_log_scale[col] = np.log1p(df_log_scale[col])

# Print statistics after log scaling
print("Statistics after log scaling:")
print(df_log_scale[numeric_cols_to_scale].describe().T[['min', 'max', 'mean', 'std']])

# 4.5 Unit Vector Scaling (L2 normalization)
print("\n4.5 Unit Vector Scaling (L2 normalization)")
# Apply unit vector scaling to numeric columns
df_unit = df_scaling.copy()

# Apply L2 normalization
from sklearn.preprocessing import normalize

df_unit[numeric_cols_to_scale] = normalize(df_unit[numeric_cols_to_scale], norm='l2', axis=0)

# Print statistics after unit vector scaling
print("Statistics after unit vector scaling:")
print(df_unit[numeric_cols_to_scale].describe().T[['min', 'max', 'mean', 'std']])

# ======================================================================
# SUMMARY
# ======================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print("\nMethods implemented:")
print("1. Handling Missing Values:")
print("   - Simple imputation (mean, most frequent)")
print("   - KNN imputation")
print("   - Forward/backward fill")
print("   - Dropping rows with missing values")

print("\n2. Handling Extreme Values:")
print("   - Z-score method")
print("   - IQR method")
print("   - Winsorization (capping)")
print("   - Removing outliers")
print("   - Log transformation")

print("\n3. Data Encoding Methods:")
print("   - Label Encoding")
print("   - One-Hot Encoding")
print("   - Binary Encoding")
print("   - Ordinal Encoding")
print("   - Feature Hashing")

print("\n4. Data Scaling Methods:")
print("   - Standard Scaling (Z-score normalization)")
print("   - Min-Max Scaling")
print("   - Robust Scaling")
print("   - Log Scaling")
print("   - Unit Vector Scaling (L2 normalization)")

print("\nFinal prepared dataset has:")
print(f"- Original shape: {df.shape}")
print(f"- Shape after handling missing values: {df_imputed.shape}")
print(f"- Shape after handling extreme values: {df_capped.shape}")
print(f"- Shape after encoding (one-hot example): {df_onehot.shape}")