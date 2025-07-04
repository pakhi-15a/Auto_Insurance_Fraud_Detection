import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import joblib

# Load data
df = pd.read_csv("insurance_claims.csv")

# Drop unnecessary columns
df.drop(columns=['_c39', 'incident_location', 'policy_number', 'insured_zip'], inplace=True, errors='ignore')

# Fill missing values
df['authorities_contacted'] = df['authorities_contacted'].fillna('Unknown')

# Cap outliers in numerical features
num_features = ['age', 'policy_annual_premium', 'umbrella_limit', 'total_claim_amount', 'property_claim']
for col in num_features:
    if col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        upper = Q3 + 1.5 * IQR
        lower = Q1 - 1.5 * IQR
        df[col] = np.where(df[col] > upper, upper, np.where(df[col] < lower, lower, df[col]))

# Encode categorical features
categorical_columns = [
    'insured_sex', 'insured_education_level', 'incident_type', 'collision_type', 'incident_severity',
    'authorities_contacted', 'incident_state', 'incident_city', 'auto_make', 'auto_model',
    'policy_bind_date', 'incident_date', 'insured_occupation', 'insured_hobbies', 'umbrella_limit',
    'property_damage', 'police_report_available', 'auto_year', 'witnesses', 'policy_state',
    'policy_csl', 'insured_relationship'
]

for col in categorical_columns:
    if col in df.columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Encode target variable
y = LabelEncoder().fit_transform(df['fraud_reported'])
X = df.drop('fraud_reported', axis=1)

# Save feature names
feature_names = X.columns.tolist()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Handle class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Train Decision Tree
model = DecisionTreeClassifier()
model.fit(X_train_scaled, y_train_resampled)

# Evaluate model (optional)
y_pred = model.predict(X_test_scaled)
print("Test Accuracy:", accuracy_score(y_test, y_pred))

# Save model, scaler, and feature names
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(feature_names, 'feature_names.pkl')

print("Saved model.pkl, scaler.pkl, and feature_names.pkl")
