import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from metrics import classification_metrics

def sample_test_labels():
    y_test = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 1]
    return y_test, y_pred

def test_classification_metrics_df():
    y_test, y_pred = sample_test_labels()
    df_metrics = classification_metrics(y_test, y_pred, "Model", output_path=None)

    assert isinstance(df_metrics, pd.DataFrame), "Should return a dataframe."

def test_classification_metrics_saves(tmp_path):
    y_test, y_pred = sample_test_labels()
    output_file = tmp_path / "test_metrics.csv"

    df_metrics = classification_metrics(
        y_test, y_pred, "Model", output_path=str(output_file))
    
    assert output_file.exists(), "CSV file is saved somewhere."
    
