import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# STAGE 6 - MODEL EVALUATION
# ==========================================

# 1. Load test data
X_test = pd.read_csv(
    "data-d1/ml-d1/X_test.csv"
)

y_test = pd.read_csv(
    "data-d1/ml-d1/y_test.csv"
).squeeze()

# 2. Load trained model
model = joblib.load(
    "churn_model.pkl"
)

print("Test data loaded successfully!")

# 3. Make predictions
y_pred = model.predict(X_test)

# 4. Calculate accuracy
accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nModel Accuracy:")
print(accuracy)

print("\nModel Accuracy (%):")
print(accuracy * 100)

# 5. Confusion Matrix
cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

# 6. Classification Report
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)