import pandas as pd


COLS_NEEDED = [
    "price",
    "square_feet",
    "bathrooms",
    "bedrooms",
    "state",
    "pets_allowed",
    "fee",
    "has_photo",
]


def select_and_clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select relevant columns and perform basic cleaning:
    - Keep required columns only
    - Coerce numeric columns to numeric
    - Remove rows with non-positive price or square_feet
    - Drop rows with missing values in required columns
    - Drop duplicates

    Parameters
    ----------
    df : pandas.DataFrame
        Raw input dataframe.

    Returns
    -------
    pandas.DataFrame
        Cleaned dataframe with required columns.
    """
    df_subset = df[COLS_NEEDED].copy()

    # numeric coercions
    for col in ["price", "square_feet", "bathrooms", "bedrooms"]:
        df_subset[col] = pd.to_numeric(df_subset[col], errors="coerce")

    # remove invalid
    df_subset = df_subset[(df_subset["price"] > 0) & (df_subset["square_feet"] > 0)]

    # drop missing in key cols + duplicates
    df_subset = df_subset.dropna(subset=COLS_NEEDED)
    df_subset = df_subset.drop_duplicates().reset_index(drop=True)

    return df_subset


def add_state_median_and_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add state-level median price and binary target `high_price`:
    - state_median_price: median(price) per state
    - high_price: 1 if price > state_median_price else 0

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing at least 'state' and 'price'.

    Returns
    -------
    pandas.DataFrame
        Copy of df with 'state_median_price' and 'high_price' columns added.
    """
    out = df.copy()

    state_medians = out.groupby("state")["price"].median()
    out["state_median_price"] = out["state"].map(state_medians)

    out["high_price"] = (out["price"] > out["state_median_price"]).astype(int)

    return out
