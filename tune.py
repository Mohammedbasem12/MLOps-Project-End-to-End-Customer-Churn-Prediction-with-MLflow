# src/tune.py

import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

# ---- CONFIGURATION ----
# Project root
dir_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# MLflow file‐store (so MLflow UI can read it)
mlruns_path = os.path.join(dir_root, '/Users/mohammedbasem/Desktop/MLops-completed/mlruns')
mlflow.set_tracking_uri(f"file://{mlruns_path}")

# Final experiment where only the best run will be logged
experiment_name = "RF_Hyperopt_BestOnly"
mlflow.set_experiment(experiment_name)

# Path to your cleaned, numeric CSV
data_path = os.path.join(dir_root, 'data', '/Users/mohammedbasem/Desktop/MLops-completed/data/telecodata.csv')
# ---- END CONFIG ----

def load_data():
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['Churn'])
    X = df.drop('Churn', axis=1)
    y = df['Churn'].map({'Yes': 1, 'No': 0}) if df['Churn'].dtype == object else df['Churn']
    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = load_data()

# Define objective for Hyperopt (we’ll *not* log these internal runs)
def objective(params):
    clf = RandomForestClassifier(**params, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    # We return negative accuracy because Hyperopt minimizes the loss
    return {'loss': -acc, 'status': STATUS_OK}

if __name__ == '__main__':
    # 1) Hyperparameter search (no MLflow logging inside objective)
    trials = Trials()
    space = {
        'n_estimators': hp.choice('n_estimators', [50, 100, 200]),
        'max_depth':    hp.choice('max_depth',    [None, 5, 10, 20]),
    }
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=20,
        trials=trials
    )

    # Convert choice indices back to actual hyperparameter values
    n_estimators = [50, 100, 200][best['n_estimators']]
    max_depth    = [None, 5, 10, 20]   [best['max_depth']]
    best_params  = {'n_estimators': n_estimators, 'max_depth': max_depth}

    print("Best hyperparameters:", best_params)

    # 2) Single MLflow run logging only the best parameters + final model
    with mlflow.start_run():
        # Log best hyperparameters
        mlflow.log_params(best_params)

        # Train and evaluate
        clf = RandomForestClassifier(**best_params, random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc   = accuracy_score(y_test, preds)
        mlflow.log_metric('accuracy', acc)

        # Log the final model artifact
        mlflow.sklearn.log_model(clf, artifact_path="best_model")

        print(f"Logged final model with accuracy {acc:.4f} under run ID {mlflow.active_run().info.run_id}")
