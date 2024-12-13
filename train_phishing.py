import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

# Load dataset
file_path = 'data/phishing/Dataset.csv'  # Replace with the actual file path
data = pd.read_csv(file_path)

# Explore dataset
print("Dataset Overview:")
print(data.info())
print("\nSample Data:")
print(data.head())

# Check for missing values
if data.isnull().sum().any():
    print("\nMissing Values Found! Handling them...")
    data = data.dropna()

# Split data into features (X) and target (y)
X = data.drop(columns=['Type'])
y = data['Type']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize Random Forest model
rf = RandomForestClassifier(random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

print("\nPerforming Grid Search for Hyperparameter Tuning...")
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters
best_params = grid_search.best_params_
print("\nBest Hyperparameters:")
print(best_params)

# Train the best model
best_rf = grid_search.best_estimator_

# Evaluate the model
print("\nEvaluating the Model...")
y_train_pred = best_rf.predict(X_train)
y_test_pred = best_rf.predict(X_test)

print("\n=== Training Set Performance ===")
print(f"Accuracy: {accuracy_score(y_train, y_train_pred):.2f}")
print("\n=== Test Set Performance ===")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.2f}")
print("\n=== Classification Report ===")
print(classification_report(y_test, y_test_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Legitimate', 'Phishing'], yticklabels=['Legitimate', 'Phishing'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('models/confusion_matrix.png')  # Save as an image file
plt.close()  # Close the plot to avoid blocking execution

# Save the trained model
output_model_path = 'models/phishing_model.pkl'
joblib.dump(best_rf, output_model_path)
print(f"\nTrained model saved to {output_model_path}")

# Save the scaler
output_scaler_path = 'models/phishing_scaler.pkl'
joblib.dump(scaler, output_scaler_path)
print(f"Scaler saved to {output_scaler_path}")

# Save feature names
feature_names_path = 'models/phishing_feature_names.pkl'
joblib.dump(X.columns.tolist(), feature_names_path)
print(f"Feature names saved to {feature_names_path}")