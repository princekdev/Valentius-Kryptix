"""
generate_data.py
----------------
Generates data/Mall_Customers.csv — a realistic customer-segmentation dataset
structured exactly like the well-known "Mall Customer Segmentation" dataset
(CustomerID, Genre, Age, Annual Income (k$), Spending Score (1-100)).

Because this environment has no general internet access to Kaggle/UCI, the
200 customer records are generated with numpy using a fixed random seed and
five realistic underlying customer archetypes (the same archetypes the
original dataset is famous for revealing). This keeps every downstream
number in the project (k, silhouette score, PCA variance, cluster stats)
100% genuine and reproducible, while preserving the real-world structure and
business meaning of the classic dataset.
"""
import numpy as np
import pandas as pd

np.random.seed(42)

archetypes = [
    # (n, age_mean, age_sd, income_mean, income_sd, score_mean, score_sd)
    (40, 45, 8, 25, 6, 20, 10),   # older, low income, low spenders
    (40, 25, 4, 25, 6, 80, 8),    # young, low income, high spenders (impulsive)
    (40, 42, 7, 55, 8, 50, 8),    # middle income, average spenders
    (40, 32, 6, 88, 10, 82, 8),   # high income, high spenders (target segment)
    (40, 48, 8, 88, 10, 18, 8),   # high income, low spenders (cautious/wealthy)
]

rows = []
cid = 1
for n, age_m, age_s, inc_m, inc_s, sc_m, sc_s in archetypes:
    ages = np.clip(np.random.normal(age_m, age_s, n), 18, 70).round().astype(int)
    incomes = np.clip(np.random.normal(inc_m, inc_s, n), 15, 140).round().astype(int)
    scores = np.clip(np.random.normal(sc_m, sc_s, n), 1, 100).round().astype(int)
    genres = np.random.choice(["Male", "Female"], n)
    for a, i, s, g in zip(ages, incomes, scores, genres):
        rows.append([cid, g, a, i, s])
        cid += 1

df = pd.DataFrame(rows, columns=["CustomerID", "Genre", "Age",
                                  "Annual Income (k$)", "Spending Score (1-100)"])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df["CustomerID"] = range(1, len(df) + 1)
df.to_csv("data/Mall_Customers.csv", index=False)
print("Saved data/Mall_Customers.csv with shape", df.shape)
