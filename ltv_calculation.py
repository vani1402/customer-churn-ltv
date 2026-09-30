import pandas as pd
from pathlib import Path

# Get the project folder
project_folder = Path(__file__).parent

# Find the feature-engineered dataset anywhere inside the project
files = list(project_folder.rglob("feature_engineered_dataset.csv"))

if not files:
    print("ERROR: feature_engineered_dataset.csv was not found.")
    print("Please check that the feature-engineered dataset exists in the project folder.")
    exit()

# Use the first matching file
dataset_path = files[0]

print("Dataset found at:")
print(dataset_path)

# Load the dataset
df = pd.read_csv(dataset_path)

print("\nDataset shape:", df.shape)

# --------------------------------------------------
# LTV CALCULATION
# --------------------------------------------------

# LTV = Monthly Charges × Tenure
df["LTV"] = df["MonthlyCharges"] * df["tenure"]

# --------------------------------------------------
# LTV CATEGORY
# --------------------------------------------------

def classify_ltv(ltv):
    if ltv < 1000:
        return "Low"
    elif ltv < 3000:
        return "Medium"
    else:
        return "High"


df["LTV_Category"] = df["LTV"].apply(classify_ltv)

# --------------------------------------------------
# LTV SUMMARY
# --------------------------------------------------

print("\nLTV Summary:")
print(df["LTV"].describe())

# --------------------------------------------------
# LTV CATEGORY DISTRIBUTION
# --------------------------------------------------

print("\nLTV Category Distribution:")
print(df["LTV_Category"].value_counts())

# --------------------------------------------------
# HIGH-VALUE CUSTOMERS WHO CHURNED
# --------------------------------------------------

high_value_churn = df[
    (df["LTV_Category"] == "High") &
    (df["Churn"] == "Yes")
]

print("\nHigh-Value Customers Who Churned:")
print("Number of customers:", len(high_value_churn))

# --------------------------------------------------
# CHURN BY LTV CATEGORY
# --------------------------------------------------

print("\nChurn by LTV Category:")

ltv_churn = pd.crosstab(
    df["LTV_Category"],
    df["Churn"],
    normalize="index"
) * 100

print(ltv_churn.round(2))

# --------------------------------------------------
# SAVE OUTPUT
# --------------------------------------------------

output_path = project_folder / "customer_ltv_dataset.csv"

df.to_csv(output_path, index=False)

print("\nLTV calculation completed successfully!")
print("Output saved to:")
print(output_path)

