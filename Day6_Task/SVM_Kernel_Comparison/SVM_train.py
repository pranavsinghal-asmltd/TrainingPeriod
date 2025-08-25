import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv

load_dotenv()

experiment_base_name = os.environ.get("MLFLOW_EXPERIMENT_NAME")
artifact_location = os.environ.get("MLFLOW_ARTIFACT_LOCATION")
if not experiment_base_name:
    raise ValueError("Environment variable MLFLOW_EXPERIMENT_NAME is not set.")

experiment_name = f"train/{experiment_base_name}"
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))

mlflow.set_experiment(experiment_name)

df = np.loadtxt(r"data2d.txt")
df = pd.DataFrame(df, columns=["x1", "x2", "y"])
df.head()
X = df.drop(columns="y")
y = df["y"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
for kernel in ["rbf", "sigmoid", "linear"]:
    with mlflow.start_run(run_name=f"SVM_{kernel.upper()}"):
        # Train
        clf = SVC(kernel=kernel, gamma="scale")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        # Log parameters & metrics
        mlflow.log_param("kernel", kernel)
        mlflow.log_param("gamma", "scale")
        mlflow.log_metric("accuracy", acc)
        #log model
        mlflow.sklearn.log_model(clf, f"svm_{kernel}_model")