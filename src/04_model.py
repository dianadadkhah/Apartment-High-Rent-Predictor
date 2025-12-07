import click
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

@click.command()
@click.argument("x_train_file", type=str)
@click.argument("x_test_file", type=str)
@click.argument("y_train_file", type=str)
@click.argument("y_test_file", type=str)
@click.argument("output", type=str)
def main(x_train_file, x_test_file, y_train_file, y_test_file, output):
    """
    Train a Logistic Regression model on preprocessed train/test CSVs,
    encode categorical features, and save performance metrics and confusion matrix.
    """
    # Load data
    X_train = pd.read_csv(x_train_file)
    X_test = pd.read_csv(x_test_file)
    y_train = pd.read_csv(y_train_file).squeeze()
    y_test = pd.read_csv(y_test_file).squeeze()

    print("[INFO] Data loaded successfully.")

    # Identify numeric and categorical features
    numeric_features = ["square_feet", "bathrooms"]
    categorical_features = ["bedrooms", "state", "pets_allowed", "fee", "has_photo"]

    # Preprocessing: OneHotEncode categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)

    print("[INFO] Data preprocessing complete.")

    # Train Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_encoded, y_train)
    y_pred = model.predict(X_test_encoded)

    print("[INFO] Model training complete.")

    # Collect performance metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred)
    }
    df_metrics = pd.DataFrame(metrics, index=["Logistic Regression"])

    # Save metrics to CSV
    df_metrics.to_csv(f"{output}/logistic_regression_metrics.csv", index=True)
    print(f"[INFO] Metrics saved to: {output}/logistic_regression_metrics.csv")

    # Plot and save confusion matrix
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("Figure 3: Confusion Matrix")
    plt.savefig(f"{output}/confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Confusion matrix saved to: {output}/confusion_matrix.png")

if __name__ == "__main__":
    main()
