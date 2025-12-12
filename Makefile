# Makefile for Apartment High-Rent Predictor
# Run full pipeline: make all
# Clean generated files: make clean

PYTHON := python



RAW_DATA    := data/raw/data.csv
CLEAN_DATA  := data/processed/full_cleaned_data.csv

EDA_HIST        := results/figures/hist_price.png
MODEL_METRICS   := results/models/logistic_regression_metrics.csv

REPORT_QMD      := reports/apartment_pricing_ml_analysis.qmd
REPORT_HTML     := reports/apartment_pricing_ml_analysis.html



.PHONY: all clean data eda model report dirs

# Main target: run everything
all: $(REPORT_HTML)


dirs:
	mkdir -p data/raw
	mkdir -p data/processed
	mkdir -p results/tables
	mkdir -p results/models
	mkdir -p results/figures



# 1. Download raw data from UCI
$(RAW_DATA): src/01_download.py | dirs
	$(PYTHON) src/01_download.py --output_file $(RAW_DATA)

# 2. Clean data, validate, create train/test splits & full_cleaned_data
$(CLEAN_DATA): $(RAW_DATA) src/02_clean.py | dirs
	$(PYTHON) src/02_clean.py $(RAW_DATA) data/processed

data: $(CLEAN_DATA)



# 3. Run EDA (produces tables + figures in results/)
$(EDA_HIST): $(CLEAN_DATA) src/03_eda.py | dirs
	$(PYTHON) src/03_eda.py $(CLEAN_DATA) results

eda: $(EDA_HIST)



# 4. Train logistic regression & save metrics + confusion matrix
$(MODEL_METRICS): $(CLEAN_DATA) src/04_model.py | dirs
	$(PYTHON) src/04_model.py $(CLEAN_DATA) results

model: $(MODEL_METRICS)



# 5. Render Quarto report (depends on cleaned data, EDA & model)
$(REPORT_HTML): $(REPORT_QMD) $(CLEAN_DATA) $(EDA_HIST) $(MODEL_METRICS)
	quarto render $(REPORT_QMD) --to html

report: $(REPORT_HTML)



clean:
	rm -f $(RAW_DATA)
	rm -f data/processed/*.csv
	rm -f results/tables/*.csv
	rm -f results/models/*.csv
	rm -f results/figures/*.png
	rm -f $(REPORT_HTML)
