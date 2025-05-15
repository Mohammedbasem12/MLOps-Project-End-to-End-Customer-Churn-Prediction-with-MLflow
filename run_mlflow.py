# run_mlflow.py
from mlflow.server import app

if __name__ == "__main__":
    # threaded=True lets multiple requests; debug=False for production-like
    app.run(host="127.0.0.1", port=5000, threaded=True, debug=False)
