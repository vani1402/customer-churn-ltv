import pandas as pd
from sklearn.linear_model import LogisticRegression

# ==========================================
# STAGE 5 - CUSTOMER CHURN PREDICTION MODEL
# ==========================================

# 1. Load training and testing data
X_train = pd.read_csv("data-d1/ml-d1/X_train.csv")
X_test = pd.read_csv("data-d1/ml-d1/X_test.csv")

y_train = pd.read_csv("data-d1/ml-d1/y_train.csv").squeeze()
y_test = pd.read_csv("data-d1/ml-d1/y_test.csv").squeeze()

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
# 2. Create the Logistic Regression model
model = LogisticRegression(
    max_iter=3000,
    random_state=42
)

# 3. Train the model
model.fit(X_train, y_train)

print("\nModel training completed successfully!")
# 4. Make predictions on test data
y_pred = model.predict(X_test)

print("\nFirst 20 predictions:")
print(y_pred[:20])# 5. Calculate model accuracy
accuracy = model.score(X_test, y_test)

print("\nModel Accuracy:", accuracy)
print("Model Accuracy (%):", accuracy * 100)
# 6. Evaluate the model
from sklearn.metrics import confusion_matrix, classification_report

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# 7. Save the trained model
import joblib

joblib.dump(model, "churn_model.pkl")

print("\nModel saved successfully as churn_model.pkl")
