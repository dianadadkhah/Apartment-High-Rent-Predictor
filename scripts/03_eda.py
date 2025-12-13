import click
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scatterplot import scatterplot_squarefeet_price

@click.command()
@click.argument("input_data", type=str)
@click.argument("output", type=str)
def main(input_data, output):
    """Conducts EDA and produces two figures"""
    # Read data
    df = pd.read_csv(input_data)

    # Subset of needed columns
    df_subset = df[["price", "square_feet", "bathrooms", "high_price"]].copy()

    # Descriptive Stats:
    desc = df_subset[["price", "square_feet", "bathrooms"]].describe()
    desc.to_csv("results/tables/describe.csv")

    # Target Count: 
    target_count = df_subset["high_price"].value_counts(normalize=True)
    target_count.to_csv("results/tables/target_counts.csv")

    # Figure 1: Distribution of Rental Prices
    plt.figure(figsize=(8,5))
    sns.histplot(df_subset["price"], bins=50)
    plt.xlim(0, df_subset["price"].quantile(0.99))
    plt.xlabel("Price (USD)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Rental Prices")

    plt.savefig("results/figures/hist_price.png")
    plt.close()

    # Figure 2: Scatterplot of Size vs. Price
    scatterplot_squarefeet_price(df_subset, f"{output}/figures/scatter.png")

    click.echo("EDA complete! Files saved in the results folder.")

if __name__ == "__main__":
    main()
