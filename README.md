"""
# MLOps Project: Customer Churn Prediction with MLflow

This project demonstrates the use of MLflow to manage the end-to-end machine learning lifecycle for a customer churn prediction task in the telecommunications domain.

## Project Objectives Met:

1.  **Experiment Tracking:** Implemented and demonstrated how MLflow can be used to track different experiments, including logging parameters, metrics, and outputs.
2.  **Model Training and Tuning:** Developed ML models and used MLflow to log different training sessions with varying parameters and hyperparameter tuning processes.
3.  **Model Deployment:** Packaged the trained model using MLflow’s model packaging tools and demonstrated how it can be deployed as a service.
4.  **Performance Monitoring:** Outlined mechanisms to monitor the deployed model's performance over time, utilizing MLflow to track drifts in model metrics.
5.  **Model Registry:** Utilized MLflow’s Model Registry to manage model versions and lifecycle including stage transitions like staging and production.

## Directory Structure

```
MLopsproject/
├── data/                     # Dataset files
├── docs/                     # Project report and presentation materials
├── notebooks/                # Jupyter notebooks for exploration and experimentation
├── scripts/                  # Python scripts for training, tuning, deployment, etc.
│   ├── 00_setup_env.sh       # Shell script for setting up environment (installing MLflow, etc.)
│   ├── 01_download_data.py   # Script to download the dataset
│   ├── 02_train_evaluate.py  # Script for model training, evaluation, and basic MLflow tracking
│   ├── 03_hyperparameter_tuning.py # Script for hyperparameter tuning with MLflow
│   ├── 04_register_model.py  # Script for registering model to MLflow Model Registry
│   └── 05_deploy_model.py    # Script/instructions for model deployment
├── mlruns/                   # MLflow tracking data (automatically generated)
├── README.md                 # This file
└── todo.md                   # Checklist for project tasks
```

## Solution Guide Implementation

1.  **Select Domain and Dataset:** 
    *   **Domain:** Telecommunications.
    *   **Task:** Customer Churn Prediction.
    *   **Dataset:** The "Telco Customer Churn" dataset is used. It includes customer information like services they subscribe to, account information, and demographic data. The goal is to predict whether a customer will churn (leave the company).

2.  **Setup MLflow:**
    *   MLflow is installed using pip (see `scripts/00_setup_env.sh`).
    *   A local MLflow tracking server is used, storing metadata in a local SQLite database (`mlruns.db`) and artifacts in the `./mlruns` directory by default when scripts are run from the `scripts` directory or if `mlflow.set_tracking_uri("file:./mlruns")` is used.

3.  **Model Development and Tracking:** 
    *   Implemented in `scripts/02_train_evaluate.py` and `notebooks/`.
    *   Uses Scikit-Learn for model development (e.g., Logistic Regression, Random Forest).
    *   MLflow is used to log parameters, metrics (accuracy, precision, recall, F1-score), and model artifacts for each experiment run.

4.  **Hyperparameter Tuning:**
    *   Implemented in `scripts/03_hyperparameter_tuning.py`.
    *   Demonstrates using a library like Hyperopt (or Scikit-learn's GridSearchCV) with MLflow to find optimal hyperparameters.
    *   All tuning trials and their results are logged to MLflow.

5.  **Model Deployment and Monitoring:**
    *   **Deployment:** The best model is packaged using MLflow and can be served locally using `mlflow models serve`. See `scripts/05_deploy_model.py` for an example.
    *   **Monitoring:** The project report (`docs/project_report.md`) discusses strategies for monitoring the deployed model's performance, such as periodically re-evaluating on new data and logging these metrics to MLflow to detect drift.

6.  **Use MLflow Model Registry:**
    *   Implemented in `scripts/04_register_model.py`.
    *   Demonstrates registering models, creating new versions, and transitioning models between stages (e.g., Staging, Production).

## Deliverables

1.  **Code Repository:** This repository, including all scripts, notebooks, and documentation.
2.  **Project Report:** A detailed report is available in `docs/project_report.md`.
3.  **Presentation:** Presentation materials/outline are available in `docs/presentation_outline.md`.

## Getting Started

1.  **Clone the repository .**
Github link : https://github.com/Mohammedbasem12/MLOps-Project-End-to-End-Customer-Churn-Prediction-with-MLflow
2.  **Set up the environment:** 
    ```bash
    cd scripts
    bash 00_setup_env.sh 
    ```
3.  **Download the data:**
    ```bash
    python 01_download_data.py
    ```
4.  **Run the MLflow UI (in a separate terminal, from the `MLopsproject` root directory):**
    ```bash
    mlflow ui
    ```
    Navigate to `http://localhost:5000` in your browser.

5.  **Run the scripts in the `scripts` directory sequentially or explore the notebooks in `notebooks/`:**
    *   `python 02_train_evaluate.py`
    *   `python 03_hyperparameter_tuning.py`
    *   `python 04_register_model.py`
    *   (For deployment, follow instructions in `05_deploy_model.py`)

"""
