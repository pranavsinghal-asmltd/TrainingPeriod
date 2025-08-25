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
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
import joblib


load_dotenv()
client = MlflowClient()
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
experiment_base_name = os.environ.get("MLFLOW_EXPERIMENT_NAME")
if not experiment_base_name:
    raise ValueError("Environment variable MLFLOW_EXPERIMENT_NAME is not set.")
artifact_location = os.environ.get("MLFLOW_ARTIFACT_LOCATION", None)
experiment_name = f"train/{experiment_base_name}"

exp = client.get_experiment_by_name(experiment_name)
if exp is None:
    exp_id = client.create_experiment(
        name=experiment_name,
        artifact_location=artifact_location
    )
else:
    exp_id = exp.experiment_id
mlflow.set_experiment(experiment_name)


df = np.loadtxt(r"data2d.txt")
df = pd.DataFrame(df, columns=["x1", "x2", "y"])
X = df.drop(columns="y")
y = df["y"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

for kernel in ["rbf", "sigmoid", "linear"]:
    with mlflow.start_run(run_name=f"SVM_{kernel.upper()}") as run:
        # Train
        clf = SVC(kernel=kernel, gamma="scale")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # Log params & metrics
        mlflow.log_param("kernel", kernel)
        mlflow.log_param("gamma", "scale")
        mlflow.log_metric("Accuracy", acc)
        mlflow.log_metric("Precision_weighted", prec)
        mlflow.log_metric("Recall_weighted", recall)
        mlflow.log_metric("F1_weighted", f1)

        # Save & log a pickle of the model (keeps it simple)
        model_filename = f"svm_{kernel}_model.pkl"
        joblib.dump(clf, model_filename)
        mlflow.log_artifact(model_filename)

        # Extra artifacts
        report = classification_report(y_test, y_pred, output_dict=True)
        pd.DataFrame(report).to_csv("classification_report.csv")
        mlflow.log_artifact("classification_report.csv")

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
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        plt.figure(figsize=(6, 5))
        plt.contourf(xx, yy, Z, alpha=0.3)
        plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_test, edgecolors='k')
        plt.title(f"Decision Boundary ({kernel})")
        mlflow.log_figure(plt.gcf(), f"decision_boundary_{kernel}.png")
        plt.close()

        # Sample data
        df.head(20).to_csv("sample_data.csv", index=False)
        mlflow.log_artifact("sample_data.csv")

        prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        mlflow.log_metric("Precision_macro", prec_macro)

        mlflow.sklearn.log_model(clf, artifact_path="model")

        # -------- Model Registration (robust to relative artifact roots) --------
        run_id = run.info.run_id
        run_info = client.get_run(run_id)
        artifact_root = run_info.info.artifact_uri 
        model_name = f"svm_{kernel}_classifier"
        model_uri = f"runs:/{run.info.run_id}/model"
        print(f"Registering model {model_name} from {model_uri}")
        try:
            # Ensure registry entry exists
            try:
                client.get_registered_model(model_name)
            except Exception:
                client.create_registered_model(model_name)
            client.create_model_version(
                name=model_name,
                source=model_uri,  
                run_id=run_id
        )
        except Exception as e:
            print(f"⚠️ Model registration failed for {kernel}: {e}")
