import math
import pandas as pd
import numpy as np
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
# After, fill the NaN with 0. We do this after because we don't want the model to think that user has rated the movie with 0
ratings_std = np.nan_to_num(ratings_std)

# We are taking a transpose since we want similarity between items which need to be in rows
similar_movie = cosine_similarity(ratings_std.T)
print(similar_movie)

# similar_movie_df = pd.DataFrame(similar_movie, index=df.columns, columns=df.columns)
# # similar_movie = pd.DataFrame(similar_movie, index=ratings_std.columns, columns=ratings_std.columns)
# print(similar_movie_df)
#
# def get_similar_movies(movie_name, user_rating):
#     # We take user_rating subtracting 2.5 because 2.5 is a mean a for a rating. That would help us to push down those related movie to that bad rating toward the bottom
#     similar_score = similar_movie_df[movie_name] * (user_rating - 2.5)
#     similar_score = similar_score.sort_values(ascending = False)
#     return similar_score
#
# # print(get_similar_movies("action1",2))
#
# # However, in real life, we are always given a list of ratings from user
# movies_ratings = [("action1", 5), ("romantic2", 5), ("romantic3", 1)]
#
# related_movies = pd.DataFrame()
# print("\n\n")
# for movie, ratings in movies_ratings:
#     related_movies = related_movies.append(get_similar_movies(movie, ratings), ignore_index=True)
#
# print(related_movies.head())
# print(related_movies.sum().sort_values(ascending=False))


# Now I want to test my recommendation algorithm with real life data sets
ratings = pd.read_csv("data/ratings.csv")
# print(ratings.head(2))
movies = pd.read_csv("data/movies.csv")
# print(movies.head(2))
# We can see that two data sets have column 'movieId' in common, so we can merge them together
movies_ratings = pd.merge(ratings, movies)
# print(movies_ratings.head(2))
# I drop some columns since they are not really important for our model
movies_ratings = movies_ratings.drop(['timestamp', 'genres'], axis = 1)
print(movies_ratings.head(2))

user_ratings = movies_ratings.pivot_table(index = ['userId'], columns = ['title'], values = 'rating')
print(user_ratings.head(2))

# Remove movies which have less than 10 ratings to remove irrelevancy due to not having enough data acquired
user_ratings = user_ratings.dropna(thresh = 10, axis = 1)
# print(user_ratings.head(2))
# user_ratings_std = std.fit_transform(user_ratings)

# Get pearson correlation between data
# Since applying pearson method already adjust the mean, I don't have to scale the data
similar_movie_df = user_ratings.fillna(0).corr(method="pearson")
print(similar_movie_df)
related_movies = pd.DataFrame()
# Recreate the helper method to find the similar movies
def get_similar_movies(movie_name, user_rating):
    # We take user_rating subtracting 2.5 because 2.5 is a mean a for a rating. That would help us to push down those related movie to that bad rating toward the bottom
    similar_score = similar_movie_df[movie_name] * (user_rating - 2.5)
    similar_score = similar_score.sort_values(ascending = False)
    return similar_score

movies_ratings = [("2 Fast 2 Furious (Fast and the Furious 2, The) (2003)", 5), ("12 Years a Slave (2013)", 4), ("2012 (2009)", 3), ("(500) Days of Summer (2009)", 2)]

for movie, ratings in movies_ratings:
    related_movies = related_movies.append(get_similar_movies(movie, ratings), ignore_index=True)

# print(related_movies.head())
print(related_movies.sum().sort_values(ascending=False).head(20))