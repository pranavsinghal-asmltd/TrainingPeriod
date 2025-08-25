import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score,
    classification_report
)
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

# Load environment variables
load_dotenv()
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
experiment_base_name = os.environ.get("MLFLOW_EXPERIMENT_NAME")
artifact_location = os.environ.get("MLFLOW_ARTIFACT_LOCATION")
if not experiment_base_name:
    raise ValueError("Environment variable MLFLOW_EXPERIMENT_NAME is not set.")

experiment_name = f"train/{experiment_base_name}"

mlflow.set_experiment(experiment_name)

client = MlflowClient()

# Load dataset
df = np.loadtxt(r"data2d.txt")
df = pd.DataFrame(df, columns=["x1", "x2", "y"])
X = df.drop(columns="y")
y = df["y"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and log for different kernels
for kernel in ["rbf", "sigmoid", "linear"]:
    with mlflow.start_run(run_name=f"SVM_{kernel.upper()}") as run:
        # Train
        clf = SVC(kernel=kernel, gamma="scale")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)  
        prec = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        # Log parameters & metrics
        mlflow.log_param("kernel", kernel)
        mlflow.log_param("gamma", "scale")
        mlflow.log_metric("Accuracy", acc)
        mlflow.log_metric("Precision_weighted", prec)
        mlflow.log_metric("Recall_weighted", recall)
        mlflow.log_metric("F1_weighted", f1)

        # Save classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        pd.DataFrame(report).to_csv("classification_report.csv")
        mlflow.log_artifact("classification_report.csv")

        # Log model
        artifact_path = f"svm_{kernel}_model"
        mlflow.sklearn.log_model(clf, artifact_path)

        # Register model with absolute source path
        run_id = run.info.run_id
        run_info = client.get_run(run_id)
        artifact_root = run_info.info.artifact_uri  
        # Ensure forward slashes
        model_source = os.path.join(artifact_root, artifact_path).replace("\\", "/")
        print(f"Registering model from: {model_source}")
        try:
            model_name = f"svm_{kernel}_classifier"
            # Ensure model exists in registry
            try:
                client.get_registered_model(model_name)
            except Exception:
                client.create_registered_model(model_name)
            # Create new version
            client.create_model_version(
                name=model_name,
                source=model_source,
                run_id=run_id
            )
        except Exception as e:
            print(f"⚠️ Model registration failed for {kernel}: {e}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix ({kernel})")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        mlflow.log_figure(plt.gcf(), f"confusion_matrix_{kernel}.png")
        plt.close()

        # Decision boundary
        x_min, x_max = X_train.iloc[:, 0].min() - 1, X_train.iloc[:, 0].max() + 1
        y_min, y_max = X_train.iloc[:, 1].min() - 1, X_train.iloc[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.figure(figsize=(6, 5))
        plt.contourf(xx, yy, Z, alpha=0.3)
        plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_test, edgecolors='k')
        plt.title(f"Decision Boundary ({kernel})")
        mlflow.log_figure(plt.gcf(), f"decision_boundary_{kernel}.png")
        plt.close()

        # Sample dataset artifact
        df.head(20).to_csv("sample_data.csv", index=False)
        mlflow.log_artifact("sample_data.csv")
