import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("movies.csv")

movies = movies[['title', 'genres', 'keywords', 'overview']]

movies.dropna(inplace=True)

movies['tags'] = movies['genres'] + movies['keywords'] + movies['overview']

new_data = movies[['title', 'tags']]

new_data['tags'] = new_data['tags'].apply(lambda x: x.lower())

cv = CountVectorizer(max_features=5000, stop_words='english')

vectors = cv.fit_transform(new_data['tags']).toarray()

similarity = cosine_similarity(vectors)

# Recommendation function
def recommend(movie):

    # convert user input to lowercase
    movie = movie.lower()

    # create lowercase titles column
    new_data['title_lower'] = new_data['title'].str.lower()

    # check movie existence
    if movie not in new_data['title_lower'].values:
        return ["Movie not found in database"]

    movie_index = new_data[new_data['title_lower'] == movie].index[0]

    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []

    for i in movies_list:
        recommended_movies.append(new_data.iloc[i[0]].title)

    return recommended_movies

# Streamlit UI
st.title("Movie Recommendation System")

movie_name = st.selectbox(
    "Select a Movie",
    new_data['title'].values
)

if st.button("Recommend"):

    recommendations = recommend(movie_name)

    st.write("Recommended Movies:")

    for movie in recommendations:
        st.write(movie)