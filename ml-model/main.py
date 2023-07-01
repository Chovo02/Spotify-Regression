import pandas as pd
import json

def calculate_average_popularity(df, artist_col, popularity_col):
    avg_popularity = df.groupby(artist_col)[popularity_col].mean().to_dict()
    return avg_popularity

def average_popularity_for_features(avg_artist_pop, feats_json):
    feats_averaged = {}

    for key, artists in feats_json.items():
        pop_list = [avg_artist_pop.get(artist, 0) for artist in artists]

        feats_averaged[key] = sum(pop_list) / len(pop_list) if pop_list else 0

    return feats_averaged

df = pd.read_csv("data\\SpotifySongPolularityAPIExtract.csv")
df.drop_duplicates(subset=['track_id'], keep='first', inplace=True)
avg_artist_pop = calculate_average_popularity(df, "artist_name", "popularity")

with open("data\\feats.json","r") as f:
    feats_json = json.load(f)

feats_averaged = average_popularity_for_features(avg_artist_pop, feats_json)

df["feats_avg_popularity"] = df["track_id"].map(feats_averaged)

print(df.head())


