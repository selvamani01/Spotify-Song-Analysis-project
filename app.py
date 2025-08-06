import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("dataset.csv")

df = load_data()

# Sidebar filters
st.sidebar.title("Filter Options")

genres = df['track_genre'].unique()
selected_genre = st.sidebar.selectbox("Select Genre", sorted(genres))

min_pop, max_pop = int(df['popularity'].min()), int(df['popularity'].max())
popularity_range = st.sidebar.slider("Select Popularity Range", min_pop, max_pop, (30, 70))

filtered_df = df[(df['track_genre'] == selected_genre) & 
                 (df['popularity'].between(popularity_range[0], popularity_range[1]))]

# App title
st.title("Spotify Song Analysis Dashboard")

# Show data info
st.subheader("Dataset Overview")
st.write(f"Total Songs: {filtered_df.shape[0]}")
st.dataframe(filtered_df[['track_name', 'artists', 'popularity', 'track_genre']].head(10))

# Correlation Heatmap
st.subheader("Correlation Heatmap")
numeric_cols = filtered_df.select_dtypes(include=['float64', 'int64'])
corr = numeric_cols.corr()

fig1, ax1 = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax1)
st.pyplot(fig1)

# Top 10 songs bar chart
st.subheader("Top 10 Songs by Popularity")
top_songs = filtered_df.sort_values(by='popularity', ascending=False).head(10)
fig2 = px.bar(top_songs, x='track_name', y='popularity', color='artists', title='Top Tracks')
st.plotly_chart(fig2)

# Radar Chart for Average Audio Features
st.subheader("Average Audio Features Radar Chart")
features = ['danceability', 'energy', 'acousticness', 'liveness', 'valence']
avg_features = filtered_df[features].mean().reset_index()
avg_features.columns = ['Feature', 'Value']

fig3 = px.line_polar(avg_features, r='Value', theta='Feature', line_close=True,
                     title='Audio Feature Profile', color_discrete_sequence=['#2ca02c'])
st.plotly_chart(fig3)

# Optional: K-Means Clustering
st.subheader("K-Means Clustering Visualization (2D PCA)")
X = filtered_df[features]
if len(X) >= 4:
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(X)

    pca = PCA(n_components=2)
    components = pca.fit_transform(X)

    filtered_df['cluster'] = clusters
    filtered_df['PC1'] = components[:, 0]
    filtered_df['PC2'] = components[:, 1]

    fig4 = px.scatter(filtered_df, x='PC1', y='PC2', color='cluster',
                      hover_data=['track_name', 'artists'], title="Song Clusters (PCA Projection)")
    st.plotly_chart(fig4)
else:
    st.warning("Not enough data points to form 4 clusters.")
