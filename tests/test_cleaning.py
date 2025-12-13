import pandas as pd
from src.cleaning import select_and_clean_columns, add_state_median_and_target


def sample_raw_df():
    """Create a small raw dataframe for testing cleaning functions."""
    return pd.DataFrame(
        {
            "price": ["1000", "2000", "-500", None],
            "square_feet": ["500", "1000", "700", "800"],
            "bathrooms": ["1", "2", "1", "1"],
            "bedrooms": ["1", "2", "1", "1"],
            "state": ["CA", "CA", "CA", "CA"],
            "pets_allowed": ["Cats", "Dogs", "Cats", "Cats"],
            "fee": ["No", "Yes", "No", "No"],
            "has_photo": ["Yes", "Yes", "Yes", "Yes"],
        }
    )


def test_select_and_clean_columns_filters_invalid_rows():
    df = sample_raw_df()
    cleaned = select_and_clean_columns(df)

    
    assert cleaned.shape[0] == 2

    # Columns should be numeric
    assert cleaned["price"].dtype.kind in "fi"
    assert cleaned["square_feet"].dtype.kind in "fi"



def test_add_state_median_and_target_creates_columns():
    df = sample_raw_df()
    cleaned = select_and_clean_columns(df)
    final = add_state_median_and_target(cleaned)

    assert "state_median_price" in final.columns
    assert "high_price" in final.columns

    
    assert set(final["high_price"].unique()).issubset({0, 1})
