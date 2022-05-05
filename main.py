import math
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("data/toy_dataset.csv", index_col=0)
print(df, "\n")
