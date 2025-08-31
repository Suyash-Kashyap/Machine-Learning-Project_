import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

# Define absolute directory paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Project root directory
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Create models directory if it doesn't exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Load dataset
data_path = os.path.join(BASE_DIR, 'data', 'Sleepdataset.csv')
df = pd.read_csv(data_path)

df.columns = df.columns.str.strip()

# Encode Gender once
le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])

# Encode Occupation once and save encoder for Flask
le_occ = LabelEncoder()
df['Occupation_enc'] = le_occ.fit_transform(df['Occupation'])
joblib.dump(le_occ, os.path.join(MODEL_DIR, 'occupation_label_encoder.pkl'))

# Encode Sleep Disorder target once
le_sleep = LabelEncoder()
df['Sleep Disorder_enc'] = le_sleep.fit_transform(df['Sleep Disorder'])
joblib.dump(le_sleep, os.path.join(MODEL_DIR, 'sleep_label_encoder.pkl'))  # Optional if needed

# Select features including encoded Occupation
X = df[['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
        'Stress Level', 'Heart Rate', 'Daily Steps', 'Gender', 'Occupation_enc']]

y = df['Sleep Disorder_enc']

# Scale numeric columns only
numeric_features = ['Age', 'Sleep Duration', 'Quality of Sleep',
                    'Physical Activity Level', 'Stress Level', 'Heart Rate', 'Daily Steps']

scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Train SVM
svm = SVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

# Save the best model and scaler for inference use
joblib.dump(lr, os.path.join(MODEL_DIR, 'logistic_regression_model.pkl'))
joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
