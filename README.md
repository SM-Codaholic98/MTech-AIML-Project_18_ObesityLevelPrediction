# Obesity Level Classification: Model Explainability with SHAP

## Project objective

Train at least two classifiers on the same multiclass prediction problem, explain their predictions with SHAP, and compare whether models with comparable predictive performance produce similar feature explanations.

This repository follows the uploaded `test.ipynb` experiment. The target is `NObeyesdad`; the classifiers are Random Forest and Extra Trees.

## Repository structure

```text
├── README.md
├── requirements.txt
├── data/
│   ├── README.md
│   └── ObesityDataSet_raw_and_data_sinthetic.csv
├── notebooks/
│   └── experiment.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── model.py
│   └── evaluation.py
├── results/
│   ├── results.csv
│   ├── comparison_table.csv
│   ├── feature_importance_comparison.csv
│   └── figures/
│       └── grouped_shap_comparison.png
└── presentation/
```

`notebooks/experiment.ipynb` is the uploaded notebook copied into the repository with its dataset path changed to the relative `data/` location. 

## Dataset

The experiment uses the Estimation of Obesity Levels Based on Eating Habits and Physical Condition dataset. The target column is `NObeyesdad`. After removing 24 duplicate rows, the uploaded notebook reports 2,087 rows and 17 columns, with no missing values.

Download the original CSV and place it at:

```text
data/ObesityDataSet_raw_and_data_sinthetic.csv
```

See [`data/README.md`](data/README.md) for the expected columns and data handling notes.

## Environment setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Run the experiment

From the repository root:

```bash
python -m src.evaluation --data data/ObesityDataSet_raw_and_data_sinthetic.csv --results results
```

The script:

1. Loads the CSV and removes exact duplicate rows.
2. Creates stratified 70% training, 15% validation, and 15% test partitions using random state 42.
3. Fits ordinal and one-hot encoders on the training partition only and applies them to validation and test data.
4. Trains Random Forest and Extra Trees with 200 trees each.
5. Reports accuracy, macro-F1, weighted-F1, macro-precision, and macro-recall for each split.
6. Calculates grouped SHAP importance on the same held-out test set.
7. Compares top-five feature overlap and Spearman rank correlation.
8. Writes CSV files and a grouped SHAP comparison figure to `results/`.

## Explainability comparison

The one-hot encoded columns belonging to `CAEC`, `CALC`, and `MTRANS` are grouped back under their original feature names. For each original feature, encoded-column SHAP contributions are summed first; the absolute grouped contributions are then averaged across test samples and output classes.

The pairwise comparison uses:

- **Top-5 feature overlap:** percentage of the two models' five highest-ranked features that are shared.
- **Spearman rank correlation:** correlation between the full feature-importance rankings.

These are measures of global feature-importance similarity. They do not establish that individual predictions or causal effects are the same.

## Important note about the initially included results

The uploaded notebook's final cell contains hardcoded test scores (Random Forest: accuracy 0.9521 and macro-F1 0.9515; Extra Trees: accuracy 0.8978 and macro-F1 0.8972). The notebook also contains two different SHAP aggregation calculations: the earlier encoded-feature aggregation and the later grouped-contribution-first calculation. The supplied `comparison_table.csv` and SHAP figure use the later calculation from cells 28–30 (40% top-five overlap and Spearman correlation 0.8000, p-value 0.000199).

The hardcoded scores are preserved as notebook-reported values in the starter `results.csv`; they are **not independently re-computed from the uploaded notebook**. Run the Python pipeline with the full dataset to generate fresh, internally consistent performance and explanation results. The script overwrites the starter CSVs with calculated results.

The two classifiers' reported macro-F1 scores differ by 5.43 percentage points, so they should not automatically be described as having similar predictive performance. The project should explicitly discuss this limitation or add further classifiers and identify a pair with comparable held-out performance.

## Reproducibility

- Random state: `42`
- Number of trees: `200`
- Split: stratified 70/15/15
- SHAP evaluation partition: test set
- Primary predictive metric: macro-F1

## Outputs

- `results/results.csv`: model metrics by train, validation, and test split.
- `results/comparison_table.csv`: pairwise explanation-similarity metrics.
- `results/feature_importance_comparison.csv`: grouped SHAP values and ranks per feature.
- `results/figures/grouped_shap_comparison.png`: comparison plot.