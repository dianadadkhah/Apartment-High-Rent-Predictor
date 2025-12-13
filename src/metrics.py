import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def classification_metrics(y_test, y_pred, model_name, output_path):
    """
    Creates and saves a table with a summary of the classification metrics. 

    Parameters
    ----------
    y_test : array
        True labels for testing the classification model. 
    y_pred : array
        Predicted targets from the model. 
    model_name: str
        Name of model that is used.
    output_path: str
        File path where the classification metrics will be saved.

    Returns
    -------
    dataframe:
        A dataframe that contains a summary of the resulting accuracy, precision, recall and f-1 score

    Examples
    --------
    classification_metrics(y_test, y_pred, "Logistic Regression", "results/tables/logistic_regression_metrics.csv")
    """
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred)
    }
    df_metrics = pd.DataFrame(metrics, index=[model_name])
   
    if output_path:
        df_metrics.to_csv(output_path, index=True)
        print(f"[INFO] Metrics saved to: {output_path}")

    return df_metrics
