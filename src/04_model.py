import click
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline

@click.command()
@click.argument("input_data", type=str)
@click.argument("output", type=str)
def main(input_data, output):
    """
    Train Logistic Regression on preprocessed data and save metrics + confusion matrix.
    Expects a CSV with features and 'high_price' target column.
    """
    # -------------------------------
    # Load data
    # -------------------------------
    df = pd.read_csv(input_data)
    
    # Split into features and target
    X = df.drop(columns=["high_price", "price", "state_median_price"])
    y = df["high_price"]
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=123
    )
    
    # Identify categorical and numeric features
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ],
        remainder="passthrough"
    )
    
    # Full pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Predict
    y_pred = pipeline.predict(X_test)
    
    # Metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred)
    }
    df_metrics = pd.DataFrame(metrics, index=["Logistic Regression"])
    df_metrics.to_csv(f"{output}/logistic_regression_metrics.csv", index=True)
    print(f"[INFO] Metrics saved to: {output}/logistic_regression_metrics.csv")
    
    # Confusion matrix
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("Figure 3: Confusion Matrix")
    plt.savefig(f"{output}/confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Confusion matrix saved to: {output}/confusion_matrix.png")

if __name__ == "__main__":
    main()
