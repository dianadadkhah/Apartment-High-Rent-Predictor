

PYTHON := python



RAW_DATA        := results/data.csv
CLEAN_DATA      := results/full_cleaned_data.csv

EDA_HIST        := results/hist_price.png
MODEL_METRICS   := results/logistic_regression_metrics.csv

REPORT_QMD      := notebooks/apartment_pricing_ml_analysis.qmd
REPORT_HTML     := notebooks/apartment_pricing_ml_analysis.html



.PHONY: all clean data eda model report

# Main target: run everything
all: $(REPORT_HTML)



# 1. Download raw data from UCI
$(RAW_DATA): src/01_download.py
	$(PYTHON) src/01_download.py --output_file $(RAW_DATA)

# 2. Clean data, validate, create train/test splits & full_cleaned_data
$(CLEAN_DATA): $(RAW_DATA) src/02_clean.py
	$(PYTHON) src/02_clean.py $(RAW_DATA) results

data: $(CLEAN_DATA)


# 3. Run EDA (script produces multiple files inside results/)
$(EDA_HIST): $(CLEAN_DATA) src/03_eda.py
	$(PYTHON) src/03_eda.py $(CLEAN_DATA) results

eda: $(EDA_HIST)



# 4. Train logistic regression & save metrics + confusion matrix
$(MODEL_METRICS): $(CLEAN_DATA) src/04_model.py
	$(PYTHON) src/04_model.py $(CLEAN_DATA) results

model: $(MODEL_METRICS)



# 5. Render Quarto report (depends on cleaned data, EDA & model)
$(REPORT_HTML): $(REPORT_QMD) $(CLEAN_DATA) $(EDA_HIST) $(MODEL_METRICS)
	quarto render $(REPORT_QMD) --to html

report: $(REPORT_HTML)



clean:
	rm -f results/*.csv
	rm -f results/*.png
	rm -f $(REPORT_HTML)

