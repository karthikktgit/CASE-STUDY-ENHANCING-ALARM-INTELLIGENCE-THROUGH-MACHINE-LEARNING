# ------------------ TASK 1: Load and Inspect Excel Data ------------------

import pandas as pd

# Load Excel file
file_path = "IM009B-XLS-ENG.xlsx"
xls = pd.ExcelFile(file_path)

# List sheet names
print("Available Sheets:", xls.sheet_names)

# Load the 'Training Data 20000' sheet
df = pd.read_excel(xls, sheet_name="Training Data 20000")

# Inspect basic info
print("\nShape of the dataset:", df.shape)
print("\nColumn names:", df.columns.tolist())

# Show first few rows
print("\nSample data:")
print(df.head())

# Show unique values in categorical columns
print("\nAlarm Tag Types:", df["Alarm Tag Type"].unique())
print("Hour Groups:", df["H"].unique())
print("Week Categories:", df["Week"].unique())

# ------------------ TASK 2: Encode Categorical Features ------------------

# Categorical columns to encode
columns_to_encode = ["Alarm Tag Type", "H", "Week"]

# Apply one-hot encoding using pandas
df_encoded = pd.get_dummies(df, columns=columns_to_encode, drop_first=True)

# Show shape and new columns created
print("\nShape after encoding:", df_encoded.shape)
print("\nNewly added encoded columns:")
for col in df_encoded.columns.difference(df.columns):
    print(col)

# Preview encoded data
print("\nSample of encoded dataset:")
print(df_encoded.head())

# ------------------ TASK 3: Drop Redundant Columns & Prepare Features ------------------

# Drop columns we no longer need
columns_to_drop = ['SO', 'Hour:0-6', 'Hour:7-12', 'Hour:13-18', 'Hour:19-24',
                   '1st Week', '2nd week', '3rd week', '4th week']
df_encoded = df_encoded.drop(columns=columns_to_drop, errors='ignore')  # ignore if missing

# Define target and features
X = df_encoded.drop(columns=['CHB'])  # Features
y = df_encoded['CHB']                 # Target

# Print results
print("\nFinal shape of X (features):", X.shape)
print("Shape of y (target):", y.shape)
print("\nSample feature columns:")
print(X.columns[:10].tolist())

# ------------------ TASK 4: Train & Evaluate Models ------------------

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 1: Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Train Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Step 3: Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Step 4: Evaluate both models
print("\n--- Logistic Regression ---")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("Classification Report:\n", classification_report(y_test, y_pred_lr))

print("\n--- Random Forest ---")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

# ------------------ TASK 5: Plot Feature Importance ------------------

import matplotlib.pyplot as plt

# Get feature importances from the Random Forest model
importances = rf_model.feature_importances_
feature_names = X.columns

# Create a DataFrame for sorting and visualization
feature_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot top 10 features
plt.figure(figsize=(10, 6))
plt.barh(feature_df['Feature'][:10][::-1], feature_df['Importance'][:10][::-1])
plt.xlabel("Feature Importance")
plt.title("Top 10 Important Features (Random Forest)")
plt.tight_layout()
plt.show()

# ------------------ Optional: Save Goal 1 Dataset ------------------

# Save encoded data to CSV for backup, sharing, or reuse
df_encoded.to_csv("goal1_encoded_training_data.csv", index=False)

print("✅ Encoded Goal 1 dataset saved as goal1_encoded_training_data.csv")
