import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def scatterplot_squarefeet_price(df, output_path):
    """
    Create and save a scatterplot of square feet vs. price, colored by high_price.

    Parameters
    ----------
    df : a pandas DataFrame
        Input DataFrame containing columns: square_feet, price, high_price.
    output_path : str
        File path where the scatterplot image will be saved.

    Returns
    -------
    figure: the created matplotlib figure.
    The figure is also saved if output_path is provided.

    Examples
    --------
    scatterplot(eda_df, "results/figures/scatterplot.png")
    """
    sample_df = df.sample(
        n=min(5000, len(df)),
        random_state=123)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=sample_df,
        x="square_feet",
        y="price",
        hue="high_price",
        alpha=0.4,
        ax=ax)

    ax.set_xlim(0, sample_df["square_feet"].quantile(0.99))
    ax.set_ylim(0, sample_df["price"].quantile(0.99))
    ax.set_xlabel("Square Feet")
    ax.set_ylabel("Price")
    ax.set_title("Size vs Price (Colored by High-Price Label)")

    if output_path:
        fig.savefig(output_path)
        plt.close(fig)

    return fig
