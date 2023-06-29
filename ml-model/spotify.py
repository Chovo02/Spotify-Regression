import base64
import requests
import json
import dotenv
import os
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

dotenv.load_dotenv()

client_id = os.getenv("SPOTIFY_CLIENT_ID")
client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

def get_token():
    auth_string = f"{client_id}:{client_secret}"
    auth_bytes = auth_string.encode("utf-8")
    auth_base64 = str(base64.b64encode(auth_bytes), "utf-8")

    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization": f"Basic {auth_base64}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "grant_type": "client_credentials"
        }
    result = requests.post(url, headers=headers, data=data)
    json_result = json.loads(result.text)
    return json_result["access_token"]

def get_auth_header(token):
    return {"Authorization": f"Bearer {token}"}

def controlla_proposte_simili(descrizione1,descrizone2):
    try:
        stop_words = stopwords.words('italian')
        vectorizer = CountVectorizer(stop_words=stop_words)
        vectorizer.fit([descrizione1,descrizone2])
        vector = vectorizer.transform([descrizione1,descrizone2])
        return cosine_similarity(vector)[0][1]
    except:
        return 0

def get_feat(track_name, token,track_id):
    url = f"https://api.spotify.com/v1/search?q={track_name}&type=track&limit=10"
    result = requests.get(url, headers=get_auth_header(token))
    json_result = json.loads(result.content)
    feat = []
    
    for item in json_result["tracks"]["items"]:
        if item["id"] == track_id or (controlla_proposte_simili(item["name"].lower(),track_name.lower())>0.9 and item["artists"][0]["name"].lower() == artist_name.lower()):
            for artist in item["artists"][1:]:
                feat.append(artist["name"])
            return feat
        elif " feat" in track_name.lower() or " (feat" in track_name.lower():
            return ["feat"]
    return feat

token = get_token()

df = pd.read_csv("data\\SpotifySongPolularityAPIExtract.csv")
df.drop_duplicates(subset=['track_id'], keep='first', inplace=True)

dictionary = {}

with open("data\\feats.json") as f:
    d = json.load(f)
    if len(d) > 0:
        dictionary = d
        df = df[~df['track_id'].isin(d)]

    for track_id,track_name,artist_name in tqdm(zip(df["track_id"], df["track_name"],df["artist_name"]), total=len(df)):
        try:
            feats = get_feat(track_name, token, track_id)
            dictionary[track_id] = feats
        except Exception as e:
            with open ("data\\feats.json","w") as f:
                json.dump(dictionary, f, indent=4)
            print(e)
            if KeyboardInterrupt:
                break