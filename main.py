import math
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
df = pd.read_csv("data/toy_dataset.csv", index_col=0)
print(df, "\n")

# Standardize data using Standard Scaler
std = StandardScaler()
# def standardize(row):
#     new_row = (row - row.mean())/(row.max() - row.min())
#     return new_row

ratings_std = std.fit_transform(df)
# ratings_std = df.apply(standardize)
print(ratings_std, "\n")

# ratings_std = ratings_std.fillna(0)
# print(ratings_std, "\n")
ratings_std = np.nan_to_num(ratings_std)

# We are taking a transpose since we want similarity between items which need to be in rows
similar_movie = cosine_similarity(ratings_std.T)
print(similar_movie)

similar_movie_df = pd.DataFrame(similar_movie, index=df.columns, columns=df.columns)
# similar_movie = pd.DataFrame(similar_movie, index=ratings_std.columns, columns=ratings_std.columns)
print(similar_movie_df)


