import click
import pandas as pd
import pandera as pa
from pandera import DataFrameSchema, Column, Check
from sklearn.model_selection import train_test_split

from src.cleaning import select_and_clean_columns, add_state_median_and_target


@click.command()
@click.argument("input_data", type=str)
@click.argument("output", type=str)
def main(input_data, output):
    """
    Read raw data, clean + create target, run validation checks, split train/test,
    and write processed outputs to data/processed/.
    """
    df = pd.read_csv(input_data)

    # --- cleaning + target creation (abstracted to src/cleaning.py) ---
    df_subset = select_and_clean_columns(df)
    df_subset = add_state_median_and_target(df_subset)

    # --- Data Validation Checks ---
    numeric_cols = ["price", "square_feet", "bathrooms", "bedrooms", "state_median_price"]
    target = "high_price"
    threshold = 0.9  # correlation threshold

    allowed_states = [
        "CA",
        "VA",
        "NM",
        "CO",
        "WV",
        "WA",
        "TX",
        "IL",
        "MS",
        "OR",
        "FL",
        "MO",
        "PA",
        "IA",
        "WI",
        "NC",
        "GA",
        "OK",
        "RI",
        "NJ",
        "IN",
        "MD",
        "OH",
        "ND",
        "NE",
        "DC",
        "AZ",
        "MA",
        "MI",
        "SC",
        "ID",
        "MN",
        "KS",
        "TN",
        "UT",
        "KY",
        "SD",
        "LA",
        "AK",
        "AR",
        "AL",
        "CT",
        "NY",
        "NV",
        "HI",
        "WY",
        "VT",
        "NH",
        "MT",
        "DE",
        "ME",
    ]

    allowed_pets = ["Cats", "Cats,Dogs", "Dogs", "Cats,Dogs,None"]
    allowed_fee = ["No", "Yes"]
    allowed_has_photo = ["Thumbnail", "Yes", "No"]

    def check_no_outliers(df_):
        has_outliers = False
        for col in numeric_cols:
            q1 = df_[col].quantile(0.25)
            q3 = df_[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            outliers = df_[(df_[col] < lower) | (df_[col] > upper)]
            if not outliers.empty:
                has_outliers = True
                print(f"Outliers detected in column '{col}':")
                print(outliers[[col]])
                print("-" * 40)

        return not has_outliers  # True if no outliers, False otherwise

    def check_categories(df_):
        invalid_found = False
        category_mapping = {
            "state": allowed_states,
            "pets_allowed": allowed_pets,
            "fee": allowed_fee,
            "has_photo": allowed_has_photo,
        }

        for col, allowed in category_mapping.items():
            invalid_rows = df_[~df_[col].isin(allowed)]
            if not invalid_rows.empty:
                invalid_found = True
                print(f"Unexpected values in column '{col}':")
                print(invalid_rows[[col]])
                print("-" * 40)

        return not invalid_found  # True if all categories valid, False otherwise

    def check_target_distribution(df_):
        unique_vals = set(df_["high_price"].unique())
        if not unique_vals.issubset({0, 1}):
            print(f"Unexpected target values found: {unique_vals}")
            return False

        ratio = df_["high_price"].mean()  # proportion of 1s
        if ratio <= 0.05 or ratio >= 0.95:
            print(f"Target distribution is extremely imbalanced: {ratio:.2f} proportion of 1s")
            return False

        return True

    result = check_target_distribution(df_subset)
    print("Target distribution check passed?", result)

    def check_corr_target(df_):
        corr = df_[numeric_cols + [target]].corr()[target].drop(target)
        return (abs(corr) < threshold).all()

    def check_corr_feats(df_):
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                corr = df_[numeric_cols[i]].corr(df_[numeric_cols[j]])
                if abs(corr) >= threshold:
                    return False
        return True

    schema = DataFrameSchema(
        {
            "price": Column(
                pa.Float,
                nullable=False,
                checks=[Check.greater_than(0), Check(lambda s: s.isna().mean() <= 0.05)],
            ),
            "square_feet": Column(
                pa.Float,
                nullable=False,
                checks=[Check.greater_than(0), Check(lambda s: s.isna().mean() <= 0.05)],
            ),
            "bathrooms": Column(
                pa.Float,
                nullable=False,
                checks=[
                    Check.greater_than_or_equal_to(0),
                    Check(lambda s: s.isna().mean() <= 0.05),
                ],
            ),
            "bedrooms": Column(
                pa.Float,
                nullable=False,
                checks=[
                    Check.greater_than_or_equal_to(0),
                    Check(lambda s: s.isna().mean() <= 0.05),
                ],
            ),
            "state": Column(
                pa.String,
                nullable=False,
                checks=[Check.isin(allowed_states), Check.str_length(min_value=1)],
            ),
            "pets_allowed": Column(pa.String, nullable=True, checks=[Check.isin(allowed_pets)]),
            "fee": Column(pa.String, nullable=True, checks=[Check.isin(allowed_fee)]),
            "has_photo": Column(
                pa.String, nullable=False, checks=[Check.isin(allowed_has_photo)]
            ),
            "state_median_price": Column(
                pa.Float,
                nullable=False,
                checks=[Check.greater_than(0), Check(lambda s: s.isna().mean() <= 0.05)],
            ),
            "high_price": Column(pa.Int, nullable=False, checks=[Check.isin([0, 1])]),
        },
        checks=[
            Check(lambda df_: df_.duplicated().sum() == 0, error="Duplicate rows detected"),
            Check(lambda df_: ~df_.isna().all(axis=1)),
            Check(check_no_outliers, error="Outliers detected in numeric columns."),
            Check(check_categories, error="Unexpected categorical values."),
            Check(check_target_distribution, error="Target variable distribution is anomalous"),
            Check(check_corr_target, error="Anomalous correlation between target and features."),
            Check(check_corr_feats, error="Anomalous correlation between features"),
        ],
    )

    try:
        schema.validate(df_subset, lazy=True)
        print("All checks passed!")
    except pa.errors.SchemaErrors as e:
        print("Some checks failed:")
        print(e.failure_cases)

    # --- Splitting data ---
    numeric_features = ["square_feet", "bathrooms"]
    categorical_features = ["bedrooms", "state", "pets_allowed", "fee", "has_photo"]

    model_df = df_subset[numeric_features + categorical_features + ["high_price"]].copy()
    X = model_df[numeric_features + categorical_features]
    y = model_df["high_price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=123
    )

    # --- Saving cleaned and processed data ---
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)
    df_subset.to_csv("data/processed/full_cleaned_data.csv", index=False)

    print("Data cleaning and processing complete! Files saved in data/processed/.")


if __name__ == "__main__":
    main()
