import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)
def run_model_report(X_train, X_test, y_train, y_test):
    # ---------------------------------------------------------
    # Create results folder if not exists
    # ---------------------------------------------------------
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    # ---------------------------------------------------------
    # Train Logistic Regression model
    # ---------------------------------------------------------
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    # Predictions
    y_pred = model.predict(X_test)
    # ---------------------------------------------------------
    # Collect metrics
    # ---------------------------------------------------------
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred)
    }
    # Convert to DataFrame
    df_metrics = pd.DataFrame(metrics, index=["Logistic Regression"])
    # ---------------------------------------------------------
    # Save metrics to CSV
    # ---------------------------------------------------------
    metrics_path = os.path.join(results_dir, "logistic_regression_metrics.csv")
    df_metrics.to_csv(metrics_path, index=True)
    print(f"[INFO] Metrics saved to: {metrics_path}")
    # ---------------------------------------------------------
    # Plot and save confusion matrix
    # ---------------------------------------------------------
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("Figure 3: Confusion Matrix")
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Confusion matrix saved to: {cm_path}")
    return df_metrics, model
# ---------------------------------------------------------
# Example usage (remove this block if importing elsewhere)
# ---------------------------------------------------------
if __name__ == "__main__":
    print("This script expects preprocessed X_train, X_test, y_train, y_test.")
    print("Import and call run_model_report(X_train, X_test, y_train, y_test) from another script.")