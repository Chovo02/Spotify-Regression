import pandas as pd
from SpotifyCustomTransformer import FeatTransformer, ArtistPopularityTransformer
from sklearn.pipeline import Pipeline


df = pd.read_csv("data\\SpotifySongPolularityAPIExtract.csv")
df.drop_duplicates(subset=['track_id'], keep='first', inplace=True)

X = df.drop(["popularity"], axis=1)
y = df["popularity"]

pipeline = Pipeline([
    ("feat_transformer", FeatTransformer()),
    ("artist_popularity_transformer", ArtistPopularityTransformer())
])

print(pipeline.fit_transform(X, y).head()) 