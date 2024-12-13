# scripts/evaluate.py
import argparse
import pandas as pd
import joblib
from sklearn.metrics import classification_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--malware_data", type=str, default="data/malware/processed_malware_data.csv", help="Processed malware data")
    parser.add_argument("--phishing_data", type=str, default="data/phishing/processed_phishing_data.csv", help="Processed phishing data")
    parser.add_argument("--ddos_data", type=str, default="data/ddos/processed_ddos_data.csv", help="Processed ddos data")
    parser.add_argument("--malware_model", type=str, default="models/malware_model.pkl", help="Malware model path")
    parser.add_argument("--phishing_model", type=str, default="models/phishing_model.pkl", help="Phishing model path")
    parser.add_argument("--ddos_model", type=str, default="models/ddos_model.pkl", help="DDOS model path")
    args = parser.parse_args()

    # Example with Malware:
    malware_df = pd.read_csv(args.malware_data)
    X_m = malware_df.drop(columns=['label', 'file_path'])
    y_m = malware_df['label']
    malware_model = joblib.load(args.malware_model)
    y_pred_m = malware_model.predict(X_m)
    print("=== Malware Model Evaluation ===")
    print(classification_report(y_m, y_pred_m))

    # Similarly for phishing and ddos (when data & models are ready)
    # phishing_df = pd.read_csv(args.phishing_data)
    # ...
    # ddos_df = pd.read_csv(args.ddos_data)
    # ...

    # Print out results for each model
