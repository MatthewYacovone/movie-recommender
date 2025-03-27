# Building a Movie Recommender with Naive Bayes

import numpy as np
import pandas as pd

# Load data
ratings_path = 'ml-1m/ratings.dat'
movies_path = 'ml-1m/movies.dat'

df_ratings = pd.read_csv(ratings_path, header=None, sep='::', engine='python')
df_movies = pd.read_csv(movies_path, header=None, sep='::', engine='python', encoding='latin1')

df_ratings.columns = ['user_id', 'movie_id', 'rating', 'timestamp']
df_movies.columns = ['movie_id', 'title', 'genres']

"""
Merge ratings with movie genres
    - Allows us to incorporate genre info into our NB Classifier.
    - Instead of treating movies as isolated entities, we provide more context about their content based on genres.
"""
df_combined = pd.merge(df_ratings, df_movies[['movie_id', 'genres']], on='movie_id', how='left')
df_combined['genres'] = df_combined['genres'].str.split('|')
df_exploded = df_combined.explode('genres') # create multiple rows for each movie where each row contains only one genre
genres_encoded = pd.get_dummies(df_exploded['genres'], prefix='genre') # convert categorical genre names into binary columns
df_with_genres = pd.concat([df_exploded, genres_encoded], axis=1)

"""
Get unique counts
    - Help define the dimensions our our ratings and genre matrices
    - Ensure subsequent matrices use the correct sizes
"""
n_users = df_with_genres['user_id'].nunique() # number of unique users
n_movies = df_with_genres['movie_id'].nunique() # number of unique movies
n_genres = genres_encoded.shape[1] # number of unique genres
print(f'Number of users: {n_users}')
print(f'Number of movies: {n_movies}')
print(f'Number of genres: {n_genres}')

# Create a genre matrix for movies (each movie's genre profile)
movie_genres = df_with_genres.drop_duplicates(subset=['movie_id']).set_index('movie_id')[genres_encoded.columns]

def load_user_rating_data(df, n_users, n_movies):
    """
    Create a user-movie rating matrix and a mapping for movie IDs.
        - Converts sparse rating data into a structured user-item matrix
        - Standardizes the format for model training
        - Provides a movie index mapping to later extract specific movies
    """
    rating_data = np.zeros((n_users, n_movies), dtype=np.intc)
    movie_id_mapping = {}

    for user_id, movie_id, rating in zip(df['user_id'], df['movie_id'], df['rating']):
        user_id = int(user_id) - 1
        if movie_id not in movie_id_mapping:
            movie_id_mapping[movie_id] = len(movie_id_mapping)
        rating_data[user_id, movie_id_mapping[movie_id]] = rating
    
    return rating_data, movie_id_mapping

def load_movie_genre_data(df, n_movies, n_genres):
    """
    Create movie-genre matrix from one-hot encoded genre data.
        - One-hot encoded genre data is now structured for further computations
        - Enables future genre-based features in our model
    """
    movie_genre_data = np.zeros((n_movies, n_genres))
    movie_ids = list(df.index)

    for i, movie_id in enumerate(movie_ids):
        movie_genre_data[i, :] = df.loc[movie_id].values
    
    return movie_genre_data

# Load data matrices
rating_data, movie_id_mapping = load_user_rating_data(df_with_genres, n_users, n_movies)
movie_genre_data = load_movie_genre_data(movie_genres, n_movies, n_genres)

# Analyze the data distribution to identify potential class imbalance issues
values, counts = np.unique(rating_data, return_counts=True)
for value, count in zip(values, counts):
    print(f"Number of rating {value}: {count}")

# Identify a target movie with the most known ratings for easier prediction validation
print(df_ratings['movie_id'].value_counts())
target_movie_id = 2858
X_ratings = np.delete(rating_data, movie_id_mapping[target_movie_id], axis=1)
Y_ratings = rating_data[:, movie_id_mapping[target_movie_id]]

X_genres = np.delete(movie_genre_data, movie_id_mapping[target_movie_id], axis=0)
Y_genres = movie_genre_data[movie_id_mapping[target_movie_id], :]

# Define the recommendation threshold (ratings above this are considered "liked")
recommend = 3
# Compute a binary indicator of positive ratings (1 if rating > recommend, else 0)
positive_ratings = (X_ratings > recommend).astype(int)

# For each user, count the number of positively rated movies per genre using matrix multiplication
# This yields a user-genre count matrix where each entry represents the count of "liked" movies in that genre.
user_genre_count = positive_ratings @ X_genres # Shape: (n_users, n_genres)

# Combine the original rating features with the count-based genre features.
X_combined = np.concatenate([X_ratings, user_genre_count], axis=1)
print('Shape of combined feature matrix:', X_combined.shape)

# Filter users to only those who rated the target movie
X_filtered = X_combined[Y_ratings > 0]
Y = Y_ratings[Y_ratings > 0]
print("Shape of X after filering:", X_filtered.shape)
print("Shape of Y:", Y.shape)

# Convert ratings to binary labels
recommend = 3 # consider ratings greater than 3 as being liked
Y_binary = np.where(Y > recommend, 1, 0)
n_pos = (Y_binary == 1).sum()
n_neg = (Y_binary == 0).sum()
print(f'There are {n_pos} positive samples and {n_neg} negative samples.')

# Split into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_filtered, Y_binary, test_size=0.2, random_state=42)
print(f'There are {len(Y_train)} training samples and {len(Y_test)} test samples.')

# Train a Naive Bayes classifier
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(alpha=1.0, fit_prior=True)
clf.fit(X_train, Y_train)

# Prediction probability of each class
prediction_prob = clf.predict_proba(X_test)
print(prediction_prob[0:10])

# Predicted class
prediction = clf.predict(X_test)
print(prediction[:10])

# Evaluating for Performance
# Accuracy
accuracy = clf.score(X_test, Y_test)
print(f'The accuracy is: {accuracy*100:.1f}%')

# Precision, Recall, and F1-Score
from sklearn.metrics import classification_report
report = classification_report(Y_test, prediction)
print(report)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test, prediction, labels=[0,1]))

# Area Under the Curve (AUC) of the Receiver Operating Characteristic (ROC)
pos_prob = prediction_prob[:, 1]
from sklearn.metrics import roc_auc_score
print("AUC before tuning with cross validation:", roc_auc_score(Y_test, pos_prob))

# ## Tuning models with cross-validation

from sklearn.model_selection import StratifiedKFold
k = 5
k_fold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

smoothing_factor_option = [1, 2, 3, 4, 5, 6]
fit_prior_option = [True, False]
auc_record = {}

for train_indices, test_indices in k_fold.split(X_filtered, Y_binary):
    X_train_k, X_test_k = X_filtered[train_indices], X_filtered[test_indices]
    Y_train_k, Y_test_k = Y_binary[train_indices], Y_binary[test_indices]
    for alpha in smoothing_factor_option:
        if alpha not in auc_record:
            auc_record[alpha] = {}
        for fit_prior in fit_prior_option:
            clf = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
            clf.fit(X_train_k, Y_train_k)
            prediction_prob = clf.predict_proba(X_test_k)
            pos_prob = prediction_prob[:, 1]
            auc = roc_auc_score(Y_test_k, pos_prob)
            auc_record[alpha][fit_prior] = auc + auc_record[alpha].get(fit_prior, 0.0)

print('smoothing  fit prior  auc')
for smoothing, smoothing_record in auc_record.items():
    for fit_prior, auc in smoothing_record.items():
        print(f'    {smoothing}        {fit_prior}        {auc/k:.5f}')

clf = MultinomialNB(alpha=5.0, fit_prior=False)
clf.fit(X_train, Y_train)
pos_prob = clf.predict_proba(X_test)[:, 1]
print('AUC with the best model:', roc_auc_score(Y_test, pos_prob))
