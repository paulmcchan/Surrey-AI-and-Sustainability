README.txt
==========
AI & Sustainability (EEEM075) — UK Electricity Demand Forecasting
URN: 6926172

OVERVIEW
--------
This project forecasts daily UK electricity demand one day ahead, compares
an XGBoost model against an LSTM model, applies SHAP explainability to
both, and applies model compression to both. All notebooks were developed
and run in Google Colab.

Six notebooks are provided. They must be run in the order listed below,
since each one depends on files saved by the notebook(s) before it.

RAW DATASETS
------------
All raw datasets are stored in this project's GitHub repository under
data/raw/ (cloned automatically at the top of Notebook 1); cleaned
outputs are saved to data/processed/. Full source links are also given
in the notebooks and in the report's Appendix.

Primary dataset:
  Electricity demand (historic_demand_2009_2024.csv)
  Source: https://www.kaggle.com/datasets/albertovidalrod/electricity-consumption-uk-20092022

Supplementary datasets:
  (1) Weather, daily (all_weather_data.parquet)
      Source: https://www.kaggle.com/datasets/jakewright/2m-daily-weather-history-uk
      Note: 2021 regional population weights used to aggregate to a
      national daily figure 
      (source: https://www.nomisweb.co.uk/datasets/pestsyoala)
      (Office for National Statistics, "Total population" dataset, via NOMIS — published 26/09/2025)
      file: region_location_lookup.csv, summarised from the source data)

  (2) PMI (united-kingdom.markit-manufacturing-pmi.csv)
      Source: https://www.mql5.com/en/economic-calendar/united-kingdom/markit-manufacturing-pmi

  (3) RSI (retail-sales-index-time-series-v45-filtered-2026-06-30T15-15-28Z.csv)
      Source: https://www.ons.gov.uk/datasets/retail-sales-index/editions/time-series/versions/45

  (4) COVID-19 stringency (OxCGRT_compact_national_v1.csv)
      Source: https://github.com/OxCGRT/covid-policy-dataset/tree/main

All raw files are mirrored at:
  https://raw.githubusercontent.com/paulmcchan/Surrey-AI-and-Sustainability/main/data/raw

Raw datasets are included with this submission; Notebook 1 will also work
standalone by pulling directly from the GitHub links above if the local
files are not present.

NOTEBOOK 1: EEEM075-6926172-Data_cleaning.ipynb
-----------------------------------------------------
Purpose: Cleans and merges five raw data sources (electricity demand,
weather, PMI, RSI, COVID-19 stringency index) into a single daily dataset.

Input:  Raw source files (demand, weather, PMI, RSI, OxCGRT stringency —
        see Appendix Table 1 in the report for full source list/URLs).
Output: merged_energy_demand_dataset.csv (5,641 rows x 13 columns,
        1 Jan 2009 - 11 June 2024, no missing values, no duplicate dates)


NOTEBOOK 2: EEEM075-6926172-EDA.ipynb
-----------------------------------------
Purpose: Exploratory data analysis (distributions, calendar effects,
weather relationships, economic indicators, COVID-19 disruption,
cross-predictor correlation/VIF checks) and feature engineering
(cyclical encoding, lag/rolling-window features, interaction term).

Input:  merged_energy_demand_dataset.csv (from Notebook 1)
Output: model_ready_dataset.csv 
        (5,611 rows x 21 columns, after removing
        the first 30 rows with lag/rolling-window NaNs)

	split_train.csv, split_val.csv, split_test.csv
	(Train set, Validation set and Test set)


NOTEBOOK 3: EEEM075-6926172-Modelling_XGBoost.ipynb
--------------------------------------
Purpose: Trains and tunes the XGBoost model. Covers baseline diagnostic,
150-combination random hyperparameter search, boundary check, final
training on combined train+validation data, test-set evaluation, residual/
bias analysis, and SHAP explainability (TreeExplainer).

Input:  model_ready_dataset.csv (from Notebook 2)
Output: xgboost_final_model.joblib, xgboost_test_predictions.csv,
        xgboost_model_comparison.csv, xgboost_model_metadata.json


NOTEBOOK 4: EEEM075-6926172-Modelling_LSTM.ipynb
------------------------------------
Purpose: Trains and tunes the LSTM model. Covers baseline diagnostic,
feature/target scaling, a two-phase hyperparameter search (Phase 1:
20-sample random search over a 75-combination space; Phase 2: multi-seed
confirmation of the top 3 candidates, to average out GPU non-determinism),
final training, test-set evaluation, residual/bias analysis, and SHAP
explainability (GradientExplainer).

Input:  model_ready_dataset.csv (from Notebook 2)
Output: lstm_final_model.keras, lstm_test_predictions.csv,
        lstm_model_comparison.csv, lstm_model_metadata.json

Requires a GPU runtime in Colab (Runtime > Change runtime type > GPU).


NOTEBOOK 5: EEEM075-6926172-Model_Compression_XGBoost.ipynb
-----------------------------------------------
Purpose: Applies two compression techniques to the tuned XGBoost model —
tree reduction (equivalent of pruning) and precision reduction
(equivalent of quantization) — plus their combination, and compares
size/accuracy trade-offs against the uncompressed baseline.

Input:  xgboost_final_model.joblib (from Notebook 3)
Output: xgboost_final_compressed_model.json, xgboost_compression_summary.csv,
        xgboost_compression_recommendation.txt


NOTEBOOK 6: EEEM075-6926172-Model_Compression_LSTM.ipynb
---------------------------------------------
Purpose: Applies three compression techniques to the tuned LSTM model —
pruning, quantization (TFLite int8), and distillation (smaller student
network, four sizes tested) — plus a combination, and compares
size/accuracy trade-offs against the uncompressed baseline.

Input:  lstm_final_model.keras (from Notebook 4)
Output: lstm_final_max_compression.tflite, lstm_final_best_accuracy.tflite,
        lstm_compression_summary.csv, lstm_compression_recommendation.txt

Requires a GPU runtime in Colab.


RUN ORDER
---------
1. EEEM075-6926172-Data_cleaning.ipynb
2. EEEM075-6926172-EDA.ipynb
3. EEEM075-6926172-Modelling_XGBoost.ipynb
4. EEEM075-6926172-Modelling_LSTM.ipynb
5. EEEM075-6926172-Model_Compression_XGBoost.ipynb
6. EEEM075-6926172-Model_Compression_LSTM.ipynb

Notebooks 3 and 4 can be run in either order relative to each other (both
depend only on Notebook 2's output), but each compression notebook (5, 6)
must run after its corresponding modelling notebook (3, 4).


NOTES
-----
- Random seeds (SEED = 42) are fixed throughout for reproducibility. Some
  GPU/cuDNN operations in the LSTM notebooks remain non-deterministic
  despite this (see Section 4.4.3 of the report); this is documented and
  addressed via multi-seed averaging in the tuning process, not treated
  as an error.
- Reported wall-clock timings (tuning/training time) may vary slightly
  between runs due to shared Colab hardware; this does not affect model
  accuracy, size, or configuration results, which are reproducible exactly.
- All notebooks were re-run in full from a clean kernel prior to
  submission, with zero execution errors.