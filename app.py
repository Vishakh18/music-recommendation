import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the clustered data
try:
    df = pd.read_csv("final_df.csv")
except FileNotFoundError:
    st.error("Data file 'final_df.csv' not found. Please ensure it is in the same directory.")
    st.stop()

# Define numerical features used for clustering and similarity
numerical_features = [
    "valence", "danceability", "energy", "tempo",
    "acousticness", "liveness", "speechiness", "instrumentalness"
]
def recommend_songs(track_name, artist_name, df, num_recommendations=5):
    """
    Recommends songs based on cluster match and cosine similarity,
    prioritizing:
      1. Tracks in the same language as the target song
      2. Other languages (excluding Tamil & Telugu)
      3. Tamil & Telugu tracks least
    """
    # 1. Exact match based on the dropdown selection
    target_song_matches = df[(df["track_name"] == track_name) & (df["artist_name"] == artist_name)]
    
    if target_song_matches.empty:
        return None, "Target song not found in the database."
        
    input_song_row = target_song_matches.iloc[0]
    input_song_cluster = input_song_row["Cluster"]
    target_language = str(input_song_row["language"]).strip().lower()

    # 2. Focus purely on the cluster pool to prioritize numerical traits
    cluster_pool = df[df["Cluster"] == input_song_cluster].copy()
    
    # Deduplicate based on name, artist, and language
    cluster_pool = cluster_pool.drop_duplicates(
        subset=['track_name', 'artist_name', 'language']
    ).reset_index(drop=True)

    if len(cluster_pool) < num_recommendations + 1:
        return None, "Not enough unique similar songs in the database to generate recommendations."

    # 3. Locate target song index in the clean pool
    pool_target = cluster_pool[
        (cluster_pool["track_name"] == track_name) & 
        (cluster_pool["artist_name"] == artist_name)
    ]
    
    if pool_target.empty:
        return None, "System error evaluating the target song after deduplication."
    
    target_idx = pool_target.index[0]

    # 4. Calculate cosine similarity purely on numerical features
    cluster_features = cluster_pool[numerical_features]
    similarity_matrix = cosine_similarity(cluster_features)
    
    # Attach similarity scores to the pool
    cluster_pool["similarity"] = similarity_matrix[target_idx]
    
    # Exclude the target song itself
    candidates = cluster_pool.drop(index=target_idx).copy()

    # 5. Define language priority rank
    def get_lang_priority(lang):
        l = str(lang).strip().lower()
        if l == target_language:
            return 0  # Priority 1: Same language
        elif l in ["tamil", "telugu"]:
            return 2  # Priority 3: Tamil and Telugu ranked last
        else:
            return 1  # Priority 2: All other languages

    candidates["lang_priority"] = candidates["language"].apply(get_lang_priority)

    # 6. Sort by language priority (ascending) and similarity (descending)
    recommendations = (
        candidates.sort_values(by=["lang_priority", "similarity"], ascending=[True, False])
        .head(num_recommendations)[["track_name", "year", "artist_name", "language"]]
    )

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
tab1, tab2 = st.tabs(["Recommendations", "Artist Directory"])

with tab1:
    st.subheader("Feature-Based Recommendations")
    st.write("Generates recommendations prioritizing audio characteristics (tempo, energy, danceability, etc.).")
    
    # Search input (User must press Enter to trigger the search)
    search_query = st.text_input("Search for a song (Case-sensitive, e.g., 'Such Keh Rha'):")
    
    if search_query:
        # changed case=False to case=True to enforce the explicit camel case requirement
        matches = df[df["track_name"].str.contains(search_query, case=False, na=False, regex=False)]
        
        if not matches.empty:
            # Deduplicate matches to keep the dropdown clean
            matches = matches.drop_duplicates(subset=["track_name", "artist_name"])
            
            # Map formatted display strings to row data
            options = {f"{row['track_name']} (by {row['artist_name']})": row for _, row in matches.iterrows()}
            
            # Dropdown containing the matches
            selected_display = st.selectbox("Select the exact track:", list(options.keys()))
            
            if st.button("Generate Recommendations", type="primary"):
                # Extract exact track and artist from the user's selection
                exact_track = options[selected_display]["track_name"]
                exact_artist = options[selected_display]["artist_name"]
                
                # Pass exact track and artist to the updated function
                recommended_songs_df, error = recommend_songs(exact_track, exact_artist, df, num_recs)

                if recommended_songs_df is not None:
                    st.success(f"Showing recommendations based on: **{selected_display}**")
                    st.dataframe(recommended_songs_df, hide_index=True, use_container_width=True)
                else:
                    st.error(error)
        else:
            st.warning(f"No tracks found named '{search_query}' in dataset.")



with tab2:
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
