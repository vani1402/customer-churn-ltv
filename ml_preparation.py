import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================================
# STAGE 4 - MACHINE LEARNING DATA PREPARATION
# ==========================================

# 1. Load Stage 3 feature-engineered dataset
df = pd.read_csv(
    "data-d1/cleaned-d1/feature_engineered_dataset.csv"
)

print("Original dataset shape:", df.shape)

# 2. Separate input features (X) and target (y)
X = df.drop("Churn", axis=1)
y = df["Churn"]

print("\nX shape:", X.shape)
print("y shape:", y.shape)

# 3. Convert Churn into numerical values
# Yes = 1, No = 0
y = y.map({"Yes": 1, "No": 0})

print("\nConverted Churn values:")
print(y.value_counts())

# 4. Convert categorical columns into numerical columns
X = pd.get_dummies(X, drop_first=True)

print("\nEncoded X shape:", X.shape)

print("\nEncoded columns:")
print(X.columns.tolist())

# 5. Split data into training and testing sets
# 80% = Training
# 20% = Testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 6. Display training and testing shapes
print("\nTraining data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# 7. Display churn distribution
print("\nTraining Churn distribution:")
print(y_train.value_counts())

print("\nTesting Churn distribution:")
print(y_test.value_counts())

print("\nStage 4 data preparation completed successfully!")
# Save prepared ML datasets

X_train.to_csv(
    "data-d1/ml-d1/X_train.csv",
    index=False
)

X_test.to_csv(
    "data-d1/ml-d1/X_test.csv",
    index=False
)

y_train.to_csv(
    "data-d1/ml-d1/y_train.csv",
    index=False
)

y_test.to_csv(
    "data-d1/ml-d1/y_test.csv",
    index=False
)

print("\nML datasets saved successfully!")