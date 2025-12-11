import click
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    plt.title("Figure 1: Distribution of Rental Prices")

    plt.savefig("results/figures/hist_price.png")
    plt.close()

    # Figure 2: Scatterplot of Size vs. Price 
    sample_df = df_subset.sample(
        n=min(5000, len(df_subset)),
        random_state=123)

    plt.figure(figsize=(8,6))
    sns.scatterplot(
        data=sample_df,
        x="square_feet",
        y="price",
        hue="high_price",
        alpha=0.4)
    plt.xlim(0, sample_df["square_feet"].quantile(0.99))
    plt.ylim(0, sample_df["price"].quantile(0.99))
    plt.title("Figure 2: Size vs Price (Colored by High-Price Label)")

    plt.savefig("results/figures/scatter.png")
    plt.close()

    click.echo("EDA complete! Files saved in the results folder.")

if __name__ == "__main__":
    main()
