import pandas as pd
import plotly.express as px

def feat(x):
    if "feat." in x:
        return True
    else:
        return False

df = pd.read_csv("data/SpotifySongPolularityAPIExtract.csv", low_memory=False)
df.drop_duplicates(subset=['track_id'], keep='first', inplace=True)
df.dropna(inplace=True)

def matrix(df):
    df["feat"] = df["track_name"].apply(lambda x: feat(x))

    df_matrix = df.drop(["track_id", "track_name", "artist_name"], axis=1)
    px.scatter_matrix(df_matrix.sample(1000)).show()

def artist_scatter(df):
    df.sort_values("artist_name", ascending=False, inplace=True)

    df_artist_scatter = df
    px.scatter(df_artist_scatter, x="artist_name", y="popularity").show()