import pandas as pd
import plotly.express as px

def feat(x):
    if "feat." in x:
        return True
    else:
        return False

df = pd.read_csv("data/SpotifySongPolularityAPIExtract.csv", low_memory=False)



df["feat"] = df["track_name"].apply(lambda x: feat(x))

df_matrix = df.drop(["track_id", "track_name", "artist_name"], axis=1)
px.scatter_matrix(df_matrix.sample(1000)).show()

df.sort_values("artist_name", ascending=False, inplace=True)
name_dict={}
for i, name in enumerate(df["artist_name"].unique()):
    name_dict[name] = i

df_artist_scatter = df
df_artist_scatter["artist_name"] = df_artist_scatter["artist_name"].map(name_dict)
px.scatter(df_artist_scatter, x="artist_name", y="popularity", color="feat").show()