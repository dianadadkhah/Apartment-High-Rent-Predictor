# Apartment High-Rent Predictor

Contributors / Authors

-   Diana Dadkhah Tirani  
-   Shanze Khemani
-   Ssemakula Peter Wasswa
-   Grigory Artazyan

## Project Summary

This project investigates whether a machine learning model can accurately predict if an apartment in the U.S. is high-priced relative to the median rent within its state. The model leverages features such as location, size, and the number of bedrooms and bathrooms to make these predictions. With rental affordability becoming an increasingly important issue in America, we were interested in understanding how rental prices vary across states and how this variation affects the ability of renters, landlords, and policymakers to identify what determines a “high-priced” apartment in a given area.

## Findings Summary and Limitations

Our logistic regression model got an accuracy of approximately 70 percent with precision and recall around 0.72 and 0.66 indicating that the model performs reasonably well at identifying high-priced apartment listings relative to each state median rent. However, there are some limitations, the model still misclassifies many high-priced listings, suggesting that additional features (such as neighborhood characteristics or text-based listing information) and more advanced models could further improve performance.

## How to Run the Data Analysis

1.  Clone the repository
2.  Create the conda environment using the provided environment.yml file
3.  Activate the environment
4.  Navigate to the root of this project on your computer using the command line and enter the following command:

```         
docker compose up
```

Copy and paste this URL into your browser: `http://127.0.0.1:8888/lab?token=mds522`

To run the analysis, open a terminal and run the following commands:

```         
python src/01_download.py

python src/02_clean.py results/data.csv results

python src/03_eda.py results/full_cleaned_data.csv results

python src/04_model.py results/full_cleaned_data.csv results


quarto render notebooks/apartment_pricing_ml_analysis.qmd --to html
quarto render notebooks/apartment_pricing_ml_analysis.qmd --to pdf
```

### Clean up

To shut down the container and clean up the resources, type `Cntrl` + `C` in the terminal where you launched the container, and then type `docker compose rm`

### Updating Docker Container Image

If the Dockerfile or environment.yml file is updated, you must rebuild the image using the following command:

```         
docker compose build
```

## Dependencies

This project uses the following software dependencies, all of which are specified in the environment.yml file:

-   Python 3.11
-   numpy 1.26.4
-   pandas 2.2.2
-   scikit-learn 1.4.2
-   altair 5.3.0
-   matplotlib 3.8.4
-   seaborn 0.13.2
-   jupyterlab 4.2.5
-   click 8.1.7
-   pandera 0.20.3 (installed via pip)
-   ucimlrepo
-   pointblank

To install all dependencies, create the conda environment using: 1. conda env create -f environment.yml 2. then activate it: conda activate Group_26

## License(s)

This project contains the following license(s):

-   MIT License
