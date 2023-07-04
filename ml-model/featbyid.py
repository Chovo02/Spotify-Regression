import json
import dotenv
import os
from alive_progress import alive_bar
import tekore as tk
import pandas as pd
import logging

def get_env(key:str):
    dotenv.load_dotenv()
    return os.environ[key]

def get_connections():
    client_ids = get_env("CLIENT_ID_LIST").split(",")
    client_secrets = get_env("CLIENT_SECRET_LIST").split(",")
    return [tk.Spotify(tk.request_client_token(client_id, client_secret)) for client_id, client_secret in zip(client_ids, client_secrets)]

def save_data(path:str = get_env("JSON_PATH"), data:dict = {}):
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
    
def get_feats_tekkore(track_id, current_connection, connections):
    connection = connections[current_connection]
    result = connection.track(track_id)
    artists = result.artists
    markets = result.available_markets
    return [artist.name for artist in artists],0 if len(markets) == 0 else 1

def load_json(X:pd.DataFrame, verbose:int = 1, path:str = get_env("JSON_PATH")):
    with open(path) as f:
        d = json.load(f)
        not_present = X[X["track_id"].apply(lambda x: x not in d.keys())]
        not_present.drop_duplicates(subset=['track_id'], keep='first', inplace=True)

        if not_present.shape[0] == 0:
            if verbose == 1:
                print("Tutte le canozni sono già in cache")
            return d, True
        elif not_present.shape[0] != X.shape[0]:
            if verbose == 1:
                print(f"Caricati {X.shape[0]-not_present.shape[0]} canzoni ")
        else:
            if verbose == 1:
                print("Nessun dato presente, inizio caricamento")
        return d, False
    
def feat_by_id_bar(X:pd.DataFrame, json_file:dict):
    i=0
    dictionary = {}
    connections = get_connections()
    with alive_bar(len(X["track_id"])) as bar:
        for track_id in X["track_id"]:
            if track_id not in json_file.keys():
                i += 1
                try:
                    feats,markets = get_feats_tekkore(track_id,i%len(connections),connections)
                    dictionary[track_id] = {"feats":
                        feats[1:],
                        "markets":markets}
                except KeyboardInterrupt:
                    save_data(data=dictionary)
                    dictionary = {}
                    break
                except Exception as e:
                    print(e)
                    save_data(data=dictionary)
                    dictionary = {}
                if i%1000 == 0:
                    save_data(data=dictionary)
                    dictionary = {}
            bar()
    save_data(data=dictionary)

def feat_by_id(X:pd.DataFrame, json_file:dict):
    i=0
    dictionary = {}
    connections = get_connections()
    for track_id in X["track_id"]:
        if track_id not in json_file.keys():
            i += 1
            try:
                feats,markets = get_feats_tekkore(track_id,i%len(connections),connections)
                dictionary[track_id] = {"feats":
                    feats[1:],
                    "markets":markets}
            except KeyboardInterrupt:
                save_data(data=dictionary)
                dictionary = {}
                break
            except Exception as e:
                print(e)
                save_data(data=dictionary)
                dictionary = {}
            
            if i%1000 == 0:
                save_data(data=dictionary)
                dictionary = {}
    save_data(data=dictionary)


def feat_dict(X:pd.DataFrame, verbose:int = 1, path:str = get_env("JSON_PATH")):

    json_file, all_fill = load_json(X, verbose, path)
    if all_fill:
        return json_file

    if verbose == 0:
        feat_by_id(X, json_file)
    else:
        feat_by_id_bar(X, json_file)


   
    with open(get_env("JSON_PATH")) as f:
        d = json.load(f)
    return d