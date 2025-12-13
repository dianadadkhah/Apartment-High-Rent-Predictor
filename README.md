### Apartment High-Rent Predictor

Contributors / Authors

Diana Dadkhah Tirani

Shanze Khemani

Ssemakula Peter Wasswa

Grigory Artazyan

### Project Summary

This project investigates whether a machine learning model can accurately predict if an apartment in the U.S. is high-priced relative to the median rent within its state. The model leverages features such as apartment size, number of bedrooms and bathrooms, pet policies, and location to make these predictions. With rental affordability becoming an increasingly important issue in the United States, this project aims to better understand how rental prices vary across states and what factors contribute to an apartment being classified as high-priced.

### Findings Summary and Limitations

Our logistic regression model achieved an accuracy of approximately 70 percent, with precision and recall values around 0.72 and 0.66, respectively. These results indicate that the model performs reasonably well at identifying high-priced apartment listings relative to each state’s median rent. However, the model still misclassifies a number of high-priced listings, suggesting that additional features (such as neighborhood-level characteristics or text-based listing information) and more advanced modeling approaches could further improve performance.

### How to Run the Data Analysis

Clone the repository

Create the conda environment using the provided environment.yml file

Activate the environment

Create the conda environment using:

conda env create -f environment.yml

To activate it: conda activate Group_26

Register the environment as a Jupyter/Quarto kernel (required for Quarto rendering):

python -m ipykernel install --user --name group_26 --display-name "Python (group_26)"


Navigate to the root of this project on your computer using the command line.

To run the entire analysis pipeline (data download, cleaning, EDA, modeling, and report generation), run:

make all


To run the unit tests for reusable functions:

make test


To remove all generated data, figures, models, and reports:

make clean

### Dependencies

 This project uses the following software dependencies, all of which are specified in the environment.yml file:

Python 3.11

numpy 1.26.4

pandas 2.2.2

scikit-learn 1.4.2

altair 5.3.0

matplotlib 3.8.4

seaborn 0.13.2

jupyterlab 4.2.5

click 8.1.7

pytest

pandera 0.20.3 (installed via pip)

ucimlrepo

pointblank

To install all dependencies, create the conda environment using:

conda env create -f environment.yml

conda activate Group_26

### License(s)

 This project contains the following license(s):

MIT License
