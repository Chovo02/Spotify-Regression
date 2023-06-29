import os
import dotenv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import spotipy
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import json

renames = {
    "valence": "positiveness",
    "duration_ms": "duration",
}

def ms_to_min(df, column):
    df[column] = df[column] / 60000
    return df

dotenv.load_dotenv()
MY_SQL_CONNECTION = os.getenv("MY_SQL_CONNECTION")

df = pd.read_csv("data\\SpotifySongPolularityAPIExtract.csv")
df.dropna(inplace=True)
df.drop_duplicates(subset="track_id", inplace=True)

df = ms_to_min(df, "duration_ms")
df.rename(columns=renames, inplace=True)
df.drop(columns=["track_id"], inplace=True)

y = df["popularity"]
X = df.drop(columns=["popularity"])





















