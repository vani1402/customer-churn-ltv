import pandas as pd

# ==========================================
# STAGE 7 - CUSTOMER CHURN + LTV ANALYSIS
# ==========================================

# 1. Load cleaned customer data
df = pd.read_csv(
    "data-d1/cleaned-d1/feature_engineered_dataset.csv"
)

print("Dataset shape:", df.shape)

# 2. Overall churn analysis
print("\nChurn Distribution:")
print(df["Churn"].value_counts())

print("\nChurn Percentage:")
print(
    df["Churn"].value_counts(normalize=True) * 100
)

# 3. Churn by Contract
print("\nChurn by Contract:")
print(
    pd.crosstab(
        df["Contract"],
        df["Churn"],
        normalize="index"
    ) * 100
)

# 4. Churn by Tenure Group
print("\nChurn by Tenure Group:")
print(
    pd.crosstab(
        df["TenureGroup"],
        df["Churn"],
        normalize="index"
    ) * 100
)

# 5. Churn by Monthly Charge Group
print("\nChurn by Monthly Charge Group:")
print(
    pd.crosstab(
        df["MonthlyChargeGroup"],
        df["Churn"],
        normalize="index"
    ) * 100
)

# 6. Churn by Service Count
print("\nChurn by Service Count:")
print(
    pd.crosstab(
        df["ServiceCount"],
        df["Churn"],
        normalize="index"
    ) * 100
)
# ==========================================
# LTV ANALYSIS
# ==========================================

# Estimated Lifetime Value
# Simple project formula:
# Monthly Charges × Tenure

df["EstimatedLTV"] = df["MonthlyCharges"] * df["tenure"]

print("\nEstimated LTV Summary:")
print(df["EstimatedLTV"].describe())

print("\nAverage Estimated LTV by Churn:")
print(
    df.groupby("Churn")["EstimatedLTV"].mean()
)

print("\nAverage Estimated LTV by Contract:")
print(
    df.groupby("Contract")["EstimatedLTV"].mean()
)