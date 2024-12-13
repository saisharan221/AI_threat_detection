import pandas as pd
import joblib
from feature_extraction import extract_features_from_url

# Load the trained model, scaler, and feature names
model_path = 'models/phishing_model.pkl'
scaler_path = 'models/phishing_scaler.pkl'
feature_names_path = 'models/phishing_feature_names.pkl'

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
training_features = joblib.load(feature_names_path)

def predict_url(url):
    """
    Predict if a URL is phishing or legitimate.
    """
    # Extract features from the URL
    features = extract_features_from_url(url)

    # Debug: Print the extracted features
    print("\nExtracted Features:")
    print(features)

    # Align extracted features with training features
    feature_df = pd.DataFrame([features])
    feature_df = feature_df.reindex(columns=training_features, fill_value=0)

    # Debug: Print the aligned DataFrame
    print("\nAligned Features for Prediction:")
    print(feature_df)

    # Scale the features
    features_scaled = scaler.transform(feature_df)

    # Predict using the loaded model
    prediction = model.predict(features_scaled)
    return 'Phishing' if prediction[0] == 1 else 'Legitimate'


if __name__ == "__main__":
    # Input URL from user
    input_url = input("Enter the URL: ")
    prediction = predict_url(input_url)
    print(f"The URL is classified as: {prediction}")
