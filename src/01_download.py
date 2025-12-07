import click
import pandas as pd
from ucimlrepo import fetch_ucirepo

@click.command()
@click.option('--output_file', help='Path to save the raw dataset, e.g., data/raw.csv')
def main(output_file):

    # 1. Fetch dataset from UCI (Apartment for Rent dataset, ID = 555)
    apartment_ds = fetch_ucirepo(id=555)
    
    # 2. Extract the original dataframe
    df = apartment_ds.data.original.copy()

    # 3. Save to CSV
    df.to_csv(output_file, index=False)

    print(f"Raw dataset saved to: {output_file}")

if __name__ == "__main__":
    main()
