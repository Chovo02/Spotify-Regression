import json
import dotenv
import os
import pandas as pd
from alive_progress import alive_bar
import tekore as tk

dotenv.load_dotenv()

def get_connections():
    client_ids = os.environ['CLIENT_ID_LIST'].split(",")
    client_secrets = os.environ['CLIENT_SECRET_LIST'].split(",")
    connections = []
    for client_id, client_secret in zip(client_ids, client_secrets):
        connections.append(tk.Spotify(tk.request_client_token(client_id, client_secret)))
    return connections

def save_data(path:str = "data\\feats.json", data:dict = {}):
    try:
        with open(path,"r") as f:
            current_data = json.load(f)
        current_data.update(data)
        with open(path,"w") as f:
            json.dump(current_data, f, indent=4)
        return True
    except Exception as e:
        print(e)
        return False
    
def get_feats_tekkore(track_id,current_connection,connections):
    feat = []
    connection = connections[current_connection]
    result = connection.track(track_id).artists
    for artist in result:
        feat.append(artist.name)
    return feat

df = pd.read_csv("data\\SpotifySongPolularityAPIExtract.csv")
df.drop_duplicates(subset=['track_id'], keep='first', inplace=True)

dictionary = {}

with open("data\\feats.json") as f:
    d = json.load(f)
    if len(d) > 0:
        dictionary = d
        print(f"Caricati {len(dictionary)} elementi dal file")

i=0
current_connection = 0
connections = get_connections()
with alive_bar(len(df["track_id"])) as bar:
    for track_id,track_name,artist_name in zip(df["track_id"], df["track_name"],df["artist_name"]):
        if track_id not in d.keys():
            i += 1
            try:
                feats = get_feats_tekkore(track_id,current_connection,connections)
                dictionary[track_id] = feats[1:]
                if current_connection == len(connections)-1:
                    current_connection = 0
                else:
                    current_connection += 1
            except KeyboardInterrupt:
                if save_data(data=dictionary):
                    dictionary = {}
                break
            except Exception as e:
                print(e)
                if save_data(data=dictionary):
                    dictionary = {}
            if i%1000 == 0 or track_id == df["track_id"].iloc[-1]:
                print(i)
                if save_data(data=dictionary):
                    dictionary = {}
                else:
                    break
        bar()