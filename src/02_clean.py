import click
import pandas as pd
import pandera as pa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from pandera import DataFrameSchema, Column, Check

@click.command()
@click.argument("input_data", type=str)
@click.argument("output", type=str)
def main(input_data, output):
    """
    Script that reads the data from the first script and does the data
    cleaning, data transformation and data splitting and then outputs
    the cleaned and processed data ready for EDA and modelling. 
    """
    df = pd.read_csv(input_data)

    cols_needed = [
        "price", "square_feet", "bathrooms", "bedrooms",
        "state", "pets_allowed", "fee", "has_photo"
    ]
    df_subset = df[cols_needed].copy()

    # Convert price and square_feet to numeric (coerce invalid values to NaN)
    df_subset["price"] = pd.to_numeric(df_subset["price"], errors="coerce")
    df_subset["square_feet"] = pd.to_numeric(df_subset["square_feet"], errors="coerce")

    # Also convert bathrooms and bedrooms to numeric
    df_subset["bathrooms"] = pd.to_numeric(df_subset["bathrooms"], errors="coerce")
    df_subset["bedrooms"] = pd.to_numeric(df_subset["bedrooms"], errors="coerce")

    # Remove non-positive or missing price/square_feet
    df_subset = df_subset[
        (df_subset["price"] > 0) &
        (df_subset["square_feet"] > 0)
    ]
# Drop rows with missing values in other key areas
    df_subset = df_subset.dropna(subset=cols_needed)

# Compute median price per state
    state_medians = df_subset.groupby("state")["price"].median()
    df_subset["state_median_price"] = df_subset["state"].map(state_medians)

# Remove duplicates 
    df_subset = df_subset.drop_duplicates().reset_index(drop=True)

# Create binary target
    df_subset["high_price"] = (df_subset["price"] > df_subset["state_median_price"]).astype(int)

# Data Validation Checks

numeric_cols = ["price", "square_feet", "bathrooms", "bedrooms", "state_median_price"]

target = "high_price"
threshold = 0.9  # correlation threshold

allowed_states = [
    'CA','VA','NM','CO','WV','WA','TX','IL','MS','OR','FL','MO','PA','IA',
    'WI','NC','GA','OK','RI','NJ','IN','MD','OH','ND','NE','DC','AZ','MA',
    'MI','SC','ID','MN','KS','TN','UT','KY','SD','LA','AK','AR','AL','CT',
    'NY','NV','HI','WY','VT','NH','MT','DE','ME'
]

allowed_pets = ['Cats', 'Cats,Dogs', 'Dogs', 'Cats,Dogs,None']
allowed_fee = ['No', 'Yes']
allowed_has_photo = ['Thumbnail', 'Yes', 'No']

def check_no_outliers(df):
    has_outliers = False  

    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        if not outliers.empty:
            has_outliers = True
            print(f"Outliers detected in column '{col}':")
            print(outliers[[col]])
            print("-"*40)
    
    return not has_outliers  # True if no outliers, False otherwise

def check_categories(df):
    invalid_found = False  # track if any invalid category exists
    
    category_mapping = {
        "state": allowed_states,
        "pets_allowed": allowed_pets,
        "fee": allowed_fee,
        "has_photo": allowed_has_photo
    }
    
    for col, allowed in category_mapping.items():
        invalid_rows = df[~df[col].isin(allowed)]
        if not invalid_rows.empty:
            invalid_found = True
            print(f"Unexpected values in column '{col}':")
            print(invalid_rows[[col]])
            print("-"*40)
    
    return not invalid_found  # True if all categories valid, False otherwise

def check_target_distribution(df):
    # Only 0 or 1 allowed
    unique_vals = set(df["high_price"].unique())
    if not unique_vals.issubset({0, 1}):
        print(f"Unexpected target values found: {unique_vals}")
        return False
    
    # Check class balance
    ratio = df["high_price"].mean()  # proportion of 1s
    if ratio <= 0.05 or ratio >= 0.95:
        print(f"Target distribution is extremely imbalanced: {ratio:.2f} proportion of 1s")
        return False
    
    return True

result = check_target_distribution(df_subset)
print("Target distribution check passed?", result)

def check_corr_target(df):
    corr = df[numeric_cols + [target]].corr()[target].drop(target)
    return (abs(corr) < threshold).all()

def check_corr_feats(df):
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            corr = df[numeric_cols[i]].corr(df[numeric_cols[j]])
            if abs(corr) >= threshold:
                return False
    return True

schema = DataFrameSchema(
    {
        "price": Column(pa.Float, nullable=False,
            checks=[Check.greater_than(0),
                    Check(lambda s: s.isna().mean() <= 0.05)]),

        "square_feet": Column(pa.Float, nullable=False,
            checks=[Check.greater_than(0),
                    Check(lambda s: s.isna().mean() <= 0.05)]),

        "bathrooms": Column(pa.Float, nullable=False,
            checks=[Check.greater_than_or_equal_to(0),
                    Check(lambda s: s.isna().mean() <= 0.05)]),

        "bedrooms": Column(pa.Float, nullable=False,
            checks=[Check.greater_than_or_equal_to(0),
                    Check(lambda s: s.isna().mean() <= 0.05)]),

        "state": Column(pa.String, nullable=False,
            checks=[Check.isin(allowed_states),
                    Check.str_length(min_value=1)]),

        "pets_allowed": Column(pa.String, nullable=True,
            checks=[Check.isin(allowed_pets)]),

        "fee": Column(pa.String, nullable=True,
            checks=[Check.isin(allowed_fee)]),

        "has_photo": Column(pa.String, nullable=False,
            checks=[Check.isin(allowed_has_photo)]),

        "state_median_price": Column(pa.Float, nullable=False,
            checks=[Check.greater_than(0),
                    Check(lambda s: s.isna().mean() <= 0.05)]),

        "high_price": Column(pa.Int, nullable=False,
            checks=[Check.isin([0, 1])]),
    },

    checks=[
        Check(lambda df: df.duplicated().sum() == 0, error="Duplicate rows detected"),
        Check(lambda df: ~df.isna().all(axis=1)),
        Check(check_no_outliers, error="Outliers detected in numeric columns."),
        Check(check_categories, error="Unexpected categorical values."),
        Check(check_target_distribution, error="Target variable distribution is anomalous"),
        Check(check_corr_target, error="Anomalous correlation between target and features."),
        Check(check_corr_feats, error="Anomalous correlation between features")
    ])

try:
    validated_df = schema.validate(df_subset, lazy=True)
    print("All checks passed!")
except pa.errors.SchemaErrors as e:
    print("Some checks failed:")
    print(e.failure_cases)

# Splitting data 
numeric_features = ["square_feet", "bathrooms"]
categorical_features = ["bedrooms", "state", "pets_allowed", "fee", "has_photo"]

model_df = df_subset[numeric_features + categorical_features + ["high_price"]].copy()

X = model_df[numeric_features + categorical_features]
y = model_df["high_price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=123
)

# Saving cleaned and processed data
X_train.to_csv("results/X_train.csv", index=False)
X_test.to_csv("results/X_test.csv", index=False)
y_train.to_csv("results/y_train.csv", index=False)
y_test.to_csv("results/y_test.csv", index=False)
df_subset.to_csv("results/full_cleaned_data.csv", index=False)
print("Data cleaning and processing complete! Files saved in the results folder.")

if __name__ == "__main__":
    main()
