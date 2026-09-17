import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the clustered data
try:
    df = pd.read_csv("final_df.csv")
except FileNotFoundError:
    st.error("Data file 'clustered_df.csv' not found. Please ensure it is in the same directory.")
    st.stop()

# Define numerical features used for clustering and similarity
numerical_features = [
    "valence", "danceability", "energy", "tempo",
    "acousticness", "liveness", "speechiness", "instrumentalness"
]

def recommend_songs(song_name, df, num_recommendations=5):
    """
    Recommends songs based on cosine similarity of numerical features 
    within the same cluster.
    """
    if song_name not in df["track_name"].values:
        return None, f"Song '{song_name}' not found in the database."

    # Extract target song data
    input_song_row = df[df["track_name"] == song_name].iloc[0]
    input_song_cluster = input_song_row["Cluster"]

    # Focus purely on the cluster pool to prioritize numerical traits
    cluster_pool = df[df["Cluster"] == input_song_cluster].copy()
    
    # FIX: Deduplicate based on name, artist, and language instead of just track_id
    cluster_pool = cluster_pool.drop_duplicates(
        subset=['track_name', 'artist_name', 'language']
    ).reset_index(drop=True)

    if len(cluster_pool) < num_recommendations + 1:
        return None, "Not enough unique similar songs in the database to generate recommendations."

    # Locate target song index in the clean pool
    target_song_matches = cluster_pool[cluster_pool["track_name"] == song_name]
    if target_song_matches.empty:
        # Fallback for slight case differences if the exact match was dropped
        target_song_matches = cluster_pool[cluster_pool["track_name"].str.lower() == song_name.lower()]
        
    if target_song_matches.empty:
        return None, "System error evaluating the target song after deduplication."
    
    target_idx = target_song_matches.index[0]

    # Calculate cosine similarity purely on numerical features
    cluster_features = cluster_pool[numerical_features]
    similarity_matrix = cosine_similarity(cluster_features)
    
    # Extract similarity scores for the target song
    song_similarities = similarity_matrix[target_idx]
    
    # Sort indices by highest similarity score
    similar_indices = np.argsort(song_similarities)[::-1]
    
    # Filter out the target song itself and grab the requested amount
    similar_indices = [idx for idx in similar_indices if idx != target_idx][:num_recommendations]

    recommendations = cluster_pool.iloc[similar_indices][
        ["track_name", "year", "artist_name", "language"]
    ]

    return recommendations, None

# --- UI Configuration ---
st.set_page_config(page_title="Audio Analytics & Recommendation", layout="wide")

st.title("Audio Analytics & Recommendation System")
st.markdown("Analyze track metrics and generate feature-based recommendations.")
st.divider()

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("Settings")
    num_recs = st.slider("Recommendation Count", min_value=1, max_value=20, value=5)
    min_popularity = st.slider("Minimum Popularity (Search)", min_value=0, max_value=100, value=0)

# --- Main Layout (Tabs) ---
tab1, tab2, tab3 = st.tabs(["Recommendations", "Keyword Search", "Artist Directory"])

with tab1:
    st.subheader("Feature-Based Recommendations")
    st.write("Generates recommendations prioritizing audio characteristics (tempo, energy, danceability, etc.).")
    
    input_song_name = st.text_input("Target Track Name:")
    
    if st.button("Generate Recommendations", type="primary"):
        if input_song_name:
            recommended_songs_df, error = recommend_songs(input_song_name, df, num_recs)

            if recommended_songs_df is not None:
                st.success("Recommendations generated successfully.")
                st.dataframe(recommended_songs_df, hide_index=True, use_container_width=True)
            else:
                st.error(error)
        else:
            st.warning("Please enter a track name.")

with tab2:
    st.subheader("Database Search")
    search_keyword = st.text_input("Search by Track or Artist:")
    
    if search_keyword:
        search_results = df[
            (df["track_name"].str.contains(search_keyword, case=False, na=False) |
             df["artist_name"].str.contains(search_keyword, case=False, na=False)) &
            (df["popularity"] >= min_popularity)
        ][["track_name", "artist_name", "year", "language", "popularity"]].drop_duplicates(
            subset=['track_name', 'artist_name']
        ).head(20)

        if not search_results.empty:
            st.dataframe(search_results, hide_index=True, use_container_width=True)
        else:
            st.info("No matching records found based on current criteria.")

with tab3:
    st.subheader("Artist Popularity Analysis")
    artist_name_input = st.text_input("Artist Name:")

    if artist_name_input:
        artist_songs = df[df["artist_name"].str.contains(artist_name_input, case=False, na=False)]
        
        if not artist_songs.empty:
            popular_artist_songs = artist_songs.sort_values(by="popularity", ascending=False)
            st.dataframe(
                popular_artist_songs[["track_name", "year", "popularity", "language"]].drop_duplicates(
                    subset=['track_name', 'language']
                ).head(10), 
                hide_index=True, 
                use_container_width=True
            )
        else:
            st.info("No records found for the specified artist.")
