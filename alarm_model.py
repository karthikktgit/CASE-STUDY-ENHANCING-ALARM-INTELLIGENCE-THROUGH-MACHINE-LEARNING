import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier, plot_importance

# Create output folder
output_dir = "goal1_plots"
os.makedirs(output_dir, exist_ok=True)

# ------------------ Load & Inspect ------------------
file_path = "IM009B-XLS-ENG.xlsx"
xls = pd.ExcelFile(file_path)
print("Available Sheets:", xls.sheet_names)

df = pd.read_excel(xls, sheet_name="Training Data 20000")

# Encode categorical features
columns_to_encode = ["Alarm Tag Type", "H", "Week"]
df_encoded = pd.get_dummies(df, columns=columns_to_encode, drop_first=True)

# Drop redundant columns
columns_to_drop = ['SO', 'Hour:0-6', 'Hour:7-12', 'Hour:13-18', 'Hour:19-24',
                   '1st Week', '2nd week', '3rd week', '4th week']
df_encoded = df_encoded.drop(columns=columns_to_drop, errors='ignore')

# Feature scaling
num_cols = ['ATD', 'M', 'Flow', 'Level', 'Pressure', 'Temperature', 'Others']
X = df_encoded.drop(columns=['CHB'])
y = df_encoded['CHB']
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# ------------------ Evaluation ------------------
print("\n--- Logistic Regression ---")
print(classification_report(y_test, y_pred_lr))
print("\n--- Random Forest ---")
print(classification_report(y_test, y_pred_rf))
print("\n--- XGBoost ---")
print(classification_report(y_test, y_pred_xgb))

# ------------------ Feature Importance Plots ------------------

# Logistic Regression
lr_coeff_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr_model.coef_[0]
}).nlargest(10, 'Coefficient', 'all')

plt.figure(figsize=(10, 6))
plt.barh(lr_coeff_df['Feature'][::-1], lr_coeff_df['Coefficient'][::-1])
plt.xlabel("Coefficient Value")
plt.title("🔍 Top 10 Features - Logistic Regression")
plt.tight_layout()
plt.savefig(f"{output_dir}/logistic_regression_importance.png")
plt.close()

# Random Forest
rf_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).nlargest(10, 'Importance', 'all')

plt.figure(figsize=(10, 6))
plt.barh(rf_importance_df['Feature'][::-1], rf_importance_df['Importance'][::-1])
plt.xlabel("Importance Score")
plt.title("🌲 Top 10 Features - Random Forest")
plt.tight_layout()
plt.savefig(f"{output_dir}/random_forest_importance.png")
plt.close()

# XGBoost
plt.figure(figsize=(10, 6))
plot_importance(xgb_model, max_num_features=10, importance_type='gain', height=0.5, grid=False)
plt.title("🚀 Top 10 Features - XGBoost (Gain)")
plt.tight_layout()
plt.savefig(f"{output_dir}/xgboost_importance.png")
plt.close()

# ------------------ CHB Trend Plots ------------------

# CHB Distribution
sns.countplot(data=df, x='CHB')
plt.title("CHB Distribution")
plt.xlabel("CHB Value (0 = No Alarm, 1 = Alarm)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{output_dir}/chb_distribution.png")
plt.close()

# CHB by Week
sns.countplot(data=df, x='Week', hue='CHB')
plt.title("CHB by Week")
plt.xlabel("Week")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{output_dir}/chb_by_week.png")
plt.close()

# CHB by Hour Group
sns.countplot(data=df, x='H', hue='CHB')
plt.title("CHB by Hour Group")
plt.xlabel("Hour Group")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{output_dir}/chb_by_hour.png")
plt.close()

# CHB by Alarm Tag Type
sns.countplot(data=df, x='Alarm Tag Type', hue='CHB')
plt.title("CHB by Alarm Tag Type")
plt.xlabel("Alarm Tag Type")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{output_dir}/chb_by_tag_type.png")
plt.close()

# ------------------ Save Dataset ------------------
df_encoded.to_csv("goal1_encoded_training_data.csv", index=False)
print(f"✅ All plots saved in folder: '{output_dir}'")
