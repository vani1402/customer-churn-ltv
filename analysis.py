import pandas as pd

# Load the cleaned dataset
df = pd.read_excel("cleaned-d1/Data-analytics-p1-dataset-raw.xlsx")
# Display first 5 rows
print("First 5 rows:")
print(df.head())

# Display column names
print("\nColumns:")
print(df.columns.tolist())

# Display number of rows and columns
print("\nDataset Shape:")
print(df.shape)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())
print("\nChurn Distribution:")
print(df["Churn"].value_counts())

print("\nChurn Percentage:")
print(df["Churn"].value_counts(normalize=True) * 100)
# Churn by Contract
print("\nChurn by Contract:")
print(pd.crosstab(df["Contract"], df["Churn"], normalize="index") * 100)
# Create tenure groups
df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[0, 12, 24, 48, float("inf")],
    labels=["0-12 Months", "13-24 Months", "25-48 Months", "49+ Months"]
)

# Churn by Tenure Group
print("\nChurn by Tenure Group:")
print(pd.crosstab(df["TenureGroup"], df["Churn"], normalize="index") * 100)