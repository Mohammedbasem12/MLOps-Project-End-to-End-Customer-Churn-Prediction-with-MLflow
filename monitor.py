# src/monitor.py

import os
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, matthews_corrcoef, confusion_matrix

# ---- CONFIG ----
dir_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
mlruns_path = os.path.join(dir_root, '/Users/mohammedbasem/Desktop/MLops-completed/mlruns')
mlflow.set_tracking_uri(f"file://{mlruns_path}")

model_name = "ChurnClassifier"
data_path = os.path.join(dir_root, 'data', 'telecodata.csv')
# Example parameters (replace with your actual model parameters)
model_params = {
    "n_estimators": 100,  
    "max_depth": 10,
}
# ---- END CONFIG ----

def load_latest_production_model():
    client = MlflowClient()
    versions = client.get_latest_versions(name=model_name, stages=["Production"])
    if not versions:
        raise Exception("No model version in 'Production' stage.")
    model_uri = f"models:/{model_name}/Production"
    return mlflow.sklearn.load_model(model_uri)

def load_sample_data():
    df = pd.read_csv(data_path).dropna(subset=['Churn'])
    X = df.drop('Churn', axis=1)
    y = df['Churn'].map({'Yes': 1, 'No': 0}) if df['Churn'].dtype == object else df['Churn']
    return X, y

def main():
    model = load_latest_production_model()
    X, y = load_sample_data()
    preds = model.predict(X)
    preds_proba = model.predict_proba(X)[:, 1]  # For AUC and Log Loss

    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds)
    rec = recall_score(y, preds)
    f1 = f1_score(y, preds)
    auc = roc_auc_score(y, preds_proba)
    loss = log_loss(y, preds_proba)
    mcc = matthews_corrcoef(y, preds)
    cm = confusion_matrix(y, preds)

    # Log to MLflow
    with mlflow.start_run(run_name="Production Monitoring"):
        # Log model parameters
        for param, value in model_params.items():
            mlflow.log_param(param, value)
        
        # Log metrics
        mlflow.log_metric("monitor_accuracy", acc)
        mlflow.log_metric("monitor_precision", prec)
        mlflow.log_metric("monitor_recall", rec)
        mlflow.log_metric("monitor_f1", f1)
        mlflow.log_metric("monitor_auc", auc)
        mlflow.log_metric("monitor_log_loss", loss)
        mlflow.log_metric("monitor_mcc", mcc)
        

    print("\n📊 Monitoring Metrics (on full dataset):")
    print(f"Accuracy      : {acc:.4f}")
    print(f"Precision     : {prec:.4f}")
    print(f"Recall        : {rec:.4f}")
    print(f"F1 Score      : {f1:.4f}")
    print(f"AUC           : {auc:.4f}")
    print(f"Log Loss      : {loss:.4f}")
    print(f"MCC           : {mcc:.4f}")

if __name__ == '__main__':
    main()
