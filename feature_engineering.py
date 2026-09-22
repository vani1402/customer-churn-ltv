import pandas as pd

# -----------------------------------
# 1. Load cleaned dataset
# -----------------------------------

input_file = "data-d1/cleaned-d1/cleaned_dataset.csv"

df = pd.read_csv(input_file)

print("Original shape:", df.shape)

# -----------------------------------
# 2. Create TenureGroup
# -----------------------------------

def tenure_group(tenure):
    if tenure <= 12:
        return "0-12"
    elif tenure <= 24:
        return "13-24"
    elif tenure <= 48:
        return "25-48"
    else:
        return "49+"

df["TenureGroup"] = df["tenure"].apply(tenure_group)

# -----------------------------------
# 3. Create MonthlyChargeGroup
# -----------------------------------

def monthly_charge_group(charge):
    if charge < 35:
        return "Low"
    elif charge < 70:
        return "Medium"
    else:
        return "High"

df["MonthlyChargeGroup"] = df["MonthlyCharges"].apply(
    monthly_charge_group
)

# -----------------------------------
# 4. Create TotalChargeGroup
# -----------------------------------

def total_charge_group(charge):
    if charge < 1000:
        return "Low"
    elif charge < 3000:
        return "Medium"
    else:
        return "High"

df["TotalChargeGroup"] = df["TotalCharges"].apply(
    total_charge_group
)

# -----------------------------------
# 5. Create ServiceCount
# -----------------------------------

service_columns = [
    "PhoneService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]

df["ServiceCount"] = 0

for column in service_columns:
    if column in df.columns:
        df["ServiceCount"] += (
            df[column].isin(["Yes"]).astype(int)
        )

# -----------------------------------
# 6. Display new features
# -----------------------------------

print("\nNew features created:")

print(
    df[
        [
            "TenureGroup",
            "MonthlyChargeGroup",
            "TotalChargeGroup",
            "ServiceCount"
        ]
    ].head()
)

# -----------------------------------
# 7. Check feature distributions
# -----------------------------------

print("\nTenureGroup:")
print(df["TenureGroup"].value_counts())

print("\nMonthlyChargeGroup:")
print(df["MonthlyChargeGroup"].value_counts())

print("\nTotalChargeGroup:")
print(df["TotalChargeGroup"].value_counts())

print("\nServiceCount:")
print(df["ServiceCount"].value_counts().sort_index())

# -----------------------------------
# 8. Save feature-engineered dataset
# -----------------------------------

output_file = (
    "data-d1/cleaned-d1/"
    "feature_engineered_dataset.csv"
)

df.to_csv(output_file, index=False)

print("\nFeature engineering completed!")
print("Final shape:", df.shape)
print("Saved to:", output_file)