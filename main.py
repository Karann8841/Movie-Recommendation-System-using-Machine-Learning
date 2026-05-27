import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("movies.csv")

# Select important columns
movies = movies[['title', 'genres', 'keywords', 'overview']]

# Remove missing values
movies.dropna(inplace=True)

# Combine columns into one
movies['tags'] = movies['genres'] + movies['keywords'] + movies['overview']

# New dataframe
new_data = movies[['title', 'tags']]

# Convert to lowercase
new_data['tags'] = new_data['tags'].apply(lambda x: x.lower())

# Convert text into vectors
cv = CountVectorizer(max_features=5000, stop_words='english')

vectors = cv.fit_transform(new_data['tags']).toarray()

# Calculate similarity
similarity = cosine_similarity(vectors)

# Recommendation function
def recommend(movie):

    movie = movie.lower()

    new_data['title_lower'] = new_data['title'].str.lower()

    matched_movies = new_data[new_data['title_lower'] == movie]

    if matched_movies.empty:
        print("\nMovie not found in dataset.")
        return

    movie_index = matched_movies.index[0]

    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    print("\nRecommended Movies:\n")

    for i in movies_list:
        print(new_data.iloc[i[0]].title)

# User input
movie_name = input("Enter Movie Name: ")

recommend(movie_name)