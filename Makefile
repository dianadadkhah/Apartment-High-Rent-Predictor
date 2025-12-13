PYTHON := python

RAW_DATA    := data/raw/data.csv
CLEAN_DATA  := data/processed/full_cleaned_data.csv

EDA_HIST        := results/figures/hist_price.png
MODEL_METRICS   := results/models/logistic_regression_metrics.csv

REPORT_QMD      := reports/apartment_pricing_ml_analysis.qmd
REPORT_HTML     := reports/apartment_pricing_ml_analysis.html

# EDA plotting functions
SRC_EDA_FUNCS := src/scatterplot.py

.PHONY: all clean data eda model report dirs test

all: $(REPORT_HTML)

dirs:
	mkdir -p data/raw
	mkdir -p data/processed
	mkdir -p results/tables
	mkdir -p results/models
	mkdir -p results/figures

$(RAW_DATA): scripts/01_download.py | dirs
	$(PYTHON) scripts/01_download.py --output_file $(RAW_DATA)

$(CLEAN_DATA): $(RAW_DATA) scripts/02_clean.py | dirs
	$(PYTHON) scripts/02_clean.py $(RAW_DATA) data/processed

data: $(CLEAN_DATA)

$(EDA_HIST): $(CLEAN_DATA) scripts/03_eda.py $(SRC_EDA_FUNCS) | dirs
	$(PYTHON) scripts/03_eda.py $(CLEAN_DATA) results

eda: $(EDA_HIST)

$(MODEL_METRICS): $(CLEAN_DATA) scripts/04_model.py | dirs
	$(PYTHON) scripts/04_model.py $(CLEAN_DATA) results

model: $(MODEL_METRICS)

$(REPORT_HTML): $(REPORT_QMD) $(CLEAN_DATA) $(EDA_HIST) $(MODEL_METRICS)
	quarto render $(REPORT_QMD) --to html

report: $(REPORT_HTML)

test:
	PYTHONPATH=. pytest -q

clean:
	rm -f $(RAW_DATA)
	rm -f data/processed/*.csv
	rm -f results/tables/*.csv
	rm -f results/models/*.csv
	rm -f results/figures/*.png
	rm -f $(REPORT_HTML)

