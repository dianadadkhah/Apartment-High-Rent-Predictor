import pandas as pd
import matplotlib.pyplot as plt
from src.scatterplot import scatterplot_squarefeet_price

def sample_test_df():
    return pd.DataFrame({
        "square_feet": [400, 800, 1200, 1600],
        "price": [1500, 2000, 2500, 3000],
        "high_price": [0, 0, 1, 1]
    })

def test_scatterplot_figure():
    df = sample_test_df()
    fig = scatterplot_squarefeet_price(df, output_path=None)
    assert isinstance(fig, plt.Figure), "Function should return a matplotlib Figure"

def test_scatterplot_saves(tmp_path):
    df = sample_test_df()
    output_file = tmp_path / "scatter_test.png"  
    fig = scatterplot_squarefeet_price(df, output_path=str(output_file))
    
    assert output_file.exists(), "Scatterplot image was saved."
    
    plt.close(fig)

