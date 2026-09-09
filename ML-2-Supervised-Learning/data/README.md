# Dataset: Adult Census Income

**File:** `adult.csv` (included in this repo, ~48,842 rows, ~3.8 MB)

**Source:** UCI Machine Learning Repository — "Adult" / "Census Income"
dataset (Kohavi, 1996), retrieved via the public GitHub mirror
[jbrownlee/Datasets](https://github.com/jbrownlee/Datasets)
(`adult-all.csv`), which combines the original UCI train+test files with
no header row.

**Why this dataset:**
- Real-world, non-trivial: predicting whether a person's income exceeds
  $50K/year from census attributes — classes are **not** linearly separable
  and overlap significantly.
- Large enough (48,842 rows, 14 features) for a meaningful comparison, with
  a genuine mix of **numerical** and **categorical** features, real
  **missing values**, and real **class imbalance** (~76% / 24%) — all of
  which this project's preprocessing and evaluation are specifically
  designed to handle.
- Publicly available and reproducible without authentication.

**Columns:** age, workclass, fnlwgt, education, education-num,
marital-status, occupation, relationship, race, sex, capital-gain,
capital-loss, hours-per-week, native-country, income (target: `<=50K` /
`>50K`).

**No download step is required** — `src/data_preprocessing.py` reads
`data/adult.csv` directly, so the project works immediately after cloning.
