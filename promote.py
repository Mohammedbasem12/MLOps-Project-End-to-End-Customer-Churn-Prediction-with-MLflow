# src/promote.py

import os
import mlflow
from mlflow.tracking import MlflowClient
import subprocess
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ---- CONFIG ----
dir_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
mlruns_path = os.path.join(dir_root, '/Users/mohammedbasem/Desktop/MLops-completed/mlruns')
mlflow.set_tracking_uri(f"file://{mlruns_path}")

model_name = "ChurnClassifier"  # Update if using another registered model name
log_file_path = os.path.join(dir_root, "promotion_log.txt")
run_monitor = True  # Set to False to skip monitor.py
# ---- END CONFIG ----

client = MlflowClient()

# Get latest version of the model
latest_versions = client.get_latest_versions(name=model_name, stages=["None"])
if not latest_versions:
    raise ValueError(f"No unregistered model version found under name: {model_name}")

new_version = latest_versions[0].version

# Archive older versions in Staging/Production
for stage in ["Staging", "Production"]:
    current = client.get_latest_versions(name=model_name, stages=[stage])
    for v in current:
        client.transition_model_version_stage(name=model_name, version=v.version, stage="Archived")

# Promote new version to both Staging and Production
client.transition_model_version_stage(name=model_name, version=new_version, stage="Staging")
client.transition_model_version_stage(name=model_name, version=new_version, stage="Production")

# Log promotion to file
with open(log_file_path, "a") as f:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    f.write(f"[{timestamp}] Promoted version {new_version} to Staging and Production for model '{model_name}'\n")

print(f"✅ Model version {new_version} promoted to Staging and Production.")

# Optional: Run monitor.py
if run_monitor:
    try:
        monitor_path = os.path.join(dir_root, 'src', 'monitor.py')
        subprocess.run(["python", monitor_path], check=True)
        print("📡 monitor.py executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ monitor.py failed: {e}")
