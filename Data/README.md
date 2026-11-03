# Dataset directory

## Required dataset

Place the original obesity dataset CSV in this directory with the following filename:

```text
ObesityDataSet_raw_and_data_sinthetic.csv
```

The project expects these 17 columns:

```text
Gender, Age, Height, Weight, family_history_with_overweight, FAVC,
FCVC, NCP, CAEC, SMOKE, CH2O, SCC, FAF, TUE, CALC, MTRANS, NObeyesdad
```

`NObeyesdad` is the seven-class target variable. The uploaded notebook reports 2,111 raw rows, 24 duplicate rows, and 2,087 rows after duplicate removal. It reports no missing values.

## Data handling

The preprocessing script removes exact duplicate rows, then makes stratified training, validation, and test partitions. Encoders are fitted only on the training partition to avoid fitting preprocessing steps on held-out data.

Do not commit a full dataset to a public repository unless its license and redistribution terms permit it. Keep any local copy out of version control if necessary.