# Investigation of the Factors Influencing Soil Biota Populations and Soil Properties in the McMurdo Dry Valley

This repository contains a collection of R scripts for generating Random Forest models, creating diagnostics, and computing Predictive Power Scores (PPS) on soil sample data from the McMurdo Dry Valleys Long-Term Ecological Research (LTER) project. The workflow cleans the data, fits classification and regression models, and creates a variety of plots and metrics.

## Contents

| Script | Purpose |
|-------|---------|
| `RandomForestTreesBetterModeling.R` | Build Random Forest models, split each dataset into training/validation/test sets, and save outputs under `processed/`, `classification/`, and `regression/`. |
| `checkAndCreateMissing.R` | Most comprehensive graph file. Checks if the other graph files made something then makes it if it is missing.  | 
| `classificationGraphs.R` | Produce additional classification model graphs such as PDP, ICE, calibration, and interaction heatmaps. |
| `regressionGraphs.R` | Generate PDP/ICE and ROC curves for regression and classification models. |
| `PDPNew.R` | Fixes errors in Classification and Regression graph|
| `PPSheatEncoding.R` | Compute Predictive Power Scores and plot heatmaps. |
| `combineAndKeepCatergorical.R` | Preprocess CSV files and compute PPS while retaining categorical variables. |
| `statsMetrics.R` | Helper functions for summarising model metrics. |
| `Validation.R` | Any file with the word validation returns valuable validation metrics that help the data to be useful |
| `Leakage.R` | Returns leakabge metrics |


## Dependencies
These scripts rely on several R packages including **tidyverse**, **caret**, **randomForest**, **pROC**, **PRROC**, **fastDummies**, **pdp**, **viridis**, and **fs**.

## Basic Usage
1. Edit the file paths near the top of each script. They currently reference Windows directories under `C:/Users/rache/...`.
2. Run `RandomForestTreesBetterModeling.R` to create training/validation/test splits and fit models. Outputs are written to a chosen `output_root` directory with the subfolders:
   - `processed/` &ndash; clean data splits
   - `classification/` &ndash; Random Forest classification models and metrics
   - `regression/` &ndash; regression models and metrics
3. Run `checkAndCreateMissing.R` to generate any missing metrics and diagnostic plots for the above models.
4. Use the remaining scripts (`classificationGraphs.R`, `regressionGraphs.R`, `PDPNew.R`, etc.) to produce additional plots such as PDP/ICE curves and PPS heatmaps.

The scripts expect CSV files with numeric and categorical features. Outputs include CSV files with metrics and `.png` images for each plot.

## Directory Structure
After running the workflow the directory tree resembles:

```
output_root/
├── processed/
│   ├── <dataset>_train.csv
│   ├── <dataset>_validation.csv
│   └── <dataset>_test.csv
├── classification/
│   └── <dataset>/<target>/
│       ├── model.rds
│       ├── importance.csv
│       ├── *.png
│       └── ...
├── regression/
│   └── <dataset>/<target>/
│       ├── model.rds
│       ├── importance.csv
│       ├── *.png
│       └── ...
└── pps_output/
    └── pps_heatmap_<dataset>.png
```

## Notes
- File paths are hard coded for the original author's environment; modify them to match your own directory layout.
- Some scripts read intermediate outputs produced by others, so run them in the order shown above.
- The repository does not include the CSV input data or the generated result directories. Links are provided in the paper. 
