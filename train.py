# src/train_final.py
import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pickle
from imblearn.over_sampling import SMOTE

# ---- CONFIGURATION ----
# Project root and MLflow tracking store
dir_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
mlruns_path = os.path.join(dir_root, '/Users/mohammedbasem/Desktop/MLops-completed/mlruns')
mlflow.set_tracking_uri(f"file://{mlruns_path}")

# Experiment and registry names
experiment_name = "Final_RF_Model"        
registered_model_name = "ChurnClassifier"  
# ---- END CONFIGURATION ----

def load_data():
    # CSV path under data/
    data_path = os.path.join(dir_root, 'data', '/Users/mohammedbasem/Desktop/MLops-completed/data/telecodata.csv')
    df = pd.read_csv(data_path).dropna(subset=['Churn'])

    # All features are numeric—no encoding needed
    X = df.drop('Churn', axis=1)

    # Map target to numeric if still string
    y = df['Churn']
    if y.dtype == object:
        y = y.map({'Yes': 1, 'No': 0})

    # Check class distribution
    print("Train set class distribution:\n", y.value_counts())

    return train_test_split(X, y, test_size=0.2, random_state=42)

def main():
    # Ensure the experiment exists
    mlflow.set_experiment(experiment_name)

    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Apply SMOTE if there is a class imbalance
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Model initialization with simplified parameters
    clf = RandomForestClassifier(
        n_estimators=100,  # Default number of trees
        max_depth=10,      # Max depth of each tree
        random_state=42
    )

    # Train, log, and register
    with mlflow.start_run() as run:
        # Log hyperparameters
        mlflow.log_param('n_estimators', 100)
        mlflow.log_param('max_depth', 10)

        # Train the model
        clf.fit(X_train_res, y_train_res)

        preds = clf.predict(X_test)
        probas = clf.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class

        # Metrics
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probas)
        f1 = f1_score(y_test, preds)

        # Log metrics
        mlflow.log_metric('accuracy', acc)
        mlflow.log_metric('auc', auc)
        mlflow.log_metric('f1_score', f1)

        # Confusion Matrix and Classification Report
        conf_matrix = confusion_matrix(y_test, preds)
        clf_report = classification_report(y_test, preds)

        # Log confusion matrix and classification report as artifacts
        with open('confusion_matrix.txt', 'w') as f:
            f.write(str(conf_matrix))
        mlflow.log_artifact('confusion_matrix.txt')

        with open('classification_report.txt', 'w') as f:
            f.write(clf_report)
        mlflow.log_artifact('classification_report.txt')

        # Plot and log confusion matrix
        plt.matshow(conf_matrix, cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.colorbar()
        plt.savefig('confusion_matrix_plot.png')
        mlflow.log_artifact('confusion_matrix_plot.png')

        # Log and register model
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path='final_model',
            registered_model_name=registered_model_name
        )

        # Save model as pickle
        with open('model.pkl', 'wb') as f:
            pickle.dump(clf, f)
        mlflow.log_artifact('model.pkl')

        print(f"Run ID: {run.info.run_id}")
        print(f"Logged to experiment '{experiment_name}' with accuracy {acc:.4f}, AUC {auc:.4f}, F1 Score {f1:.4f}")
        print(f"Registered model under name '{registered_model_name}'")


if __name__ == '__main__':
    main()
