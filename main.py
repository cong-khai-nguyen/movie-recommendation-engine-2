import math
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("data/toy_dataset.csv", index_col=0)
print(df, "\n")
def standardize(row):
    new_row = (row - row.mean())/(row.max() - row.min())
    return new_row

ratings_std = df.apply(standardize)
print(ratings_std, "\n")
