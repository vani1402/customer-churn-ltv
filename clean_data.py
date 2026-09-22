import pandas as pd
import os

# -----------------------------
# 1. Load raw dataset
# -----------------------------

input_file = "data-d1/raw-d1/Data-analytics-p1-dataset-raw.csv"

print("Loading:", input_file)

df = pd.read_csv(input_file)

print("\nOriginal dataset shape:", df.shape)

# -----------------------------
# 2. Remove Customer ID
# -----------------------------

if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])
    print("Removed customerID column.")

# -----------------------------
# 3. Convert TotalCharges
# -----------------------------

if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

# -----------------------------
# 4. Check missing values
# -----------------------------

print("\nMissing values before cleaning:")
print(df.isnull().sum())

# -----------------------------
# 5. Fill numerical missing values
# -----------------------------

numeric_columns = df.select_dtypes(include=["number"]).columns

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

# -----------------------------
# 6. Fill categorical missing values
# -----------------------------

categorical_columns = df.select_dtypes(include=["object"]).columns

for column in categorical_columns:
    if df[column].isnull().sum() > 0:
        df[column] = df[column].fillna(df[column].mode()[0])

# -----------------------------
# 7. Remove duplicates
# -----------------------------

duplicates = df.duplicated().sum()

print("\nDuplicate rows found:", duplicates)

df = df.drop_duplicates()

# -----------------------------
# 8. Clean text columns
# -----------------------------

for column in df.select_dtypes(include=["object"]).columns:
    df[column] = df[column].str.strip()

# -----------------------------
# 9. Create cleaned-data folder
# -----------------------------

output_folder = "data-d1/cleaned-d1"

os.makedirs(output_folder, exist_ok=True)

# -----------------------------
# 10. Save cleaned dataset
# -----------------------------

output_file = os.path.join(
    output_folder,
    "cleaned_dataset.csv"
)

df.to_csv(output_file, index=False)

# -----------------------------
# 11. Final validation
# -----------------------------

print("\nCleaning completed successfully!")

print("Final dataset shape:", df.shape)

print(
    "Remaining missing values:",
    df.isnull().sum().sum()
)

print("\nCleaned dataset saved to:")
print(output_file)