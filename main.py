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

ratings_std = ratings_std.fillna(0)
print(ratings_std, "\n")

# We are taking a transpose since we want similarity between items which need to be in rows
similar_movie = cosine_similarity(ratings_std.T)
print(similar_movie)