import json
import dotenv
import os
from alive_progress import alive_bar
import tekore as tk
import pandas as pd
import logging

def get_env(key:str):
    """
    Takes a key as string and returns the value of the environment variable with that key

    Parameters
    ----------
    key : str
             The key associated with the content of the environment variable

    Returns
    -------
    String value of the environment variable
    """
    dotenv.load_dotenv()
    return os.environ[key]

def get_connections():
    """
    Returns
    -------
    A list containing the connections to the Spotify API
    """
    client_ids = get_env("CLIENT_ID_LIST").split(",")
    client_secrets = get_env("CLIENT_SECRET_LIST").split(",")
    return [tk.Spotify(tk.request_client_token(client_id, client_secret)) for client_id, client_secret in zip(client_ids, client_secrets)]

def save_data(path:str = get_env("JSON_PATH"), data:dict = {}):
    """
    Saves the data in the json file

    Parameters
    ----------
    path : str, optional
              Path of the json file (to insert in the env file)\n
              Default value is the path of the json file in the environment variable JSON_PATH

    data : dict, optional
              Dictionary containing the data to save\n
              Default value is an empty dictionary
    
    Returns
    -------
    True if the data has been saved correctly
    False if an error occurred

    """
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
    """
    Returns the features of a track

    Parameters
    ----------
    track_id : str
            The id of the track

    current_connection : int
                    The index of the current connection to the Spotify API

    connections : list
                    The list of the connections to the Spotify API

    Returns
    -------
    A list containing the names of the artists of the track and\n 
    a boolean value indicating if the track is at the moment\n 
    available in any market
    """
    connection = connections[current_connection]
    result = connection.track(track_id)
    artists = result.artists
    markets = result.available_markets
    return [artist.name for artist in artists],0 if len(markets) == 0 else 1

def load_json(X:pd.DataFrame, verbose:int = 1, path:str = get_env("JSON_PATH")):
    """
    Checks how many tracks are already present in the cache

    Parameters
    ----------
    X : pd.DataFrame
            The dataframe containing the tracks

    verbose : int, optional
                    Sets the verbosity of the function\n
                    if 1 prints are enabled\n
                    if 0 prints are disabled\n

    path : str, optional
                Path of the json file (to insert in the env file)\n
                Default value is the path of the json file in the environment variable JSON_PATH

    Returns
    -------
    A dictionary containing the tracks already present in the cache\n
    A boolean value indicating if all the tracks are already present in the cache
    """
    with open(path) as f:
        d = json.load(f)
        not_present = X[X["track_id"].apply(lambda x: x not in d.keys())]

        if not_present.shape[0] == 0:
            if verbose == 1:
                print("Tutte le canozni sono già presenti in cache")
            return d, True
        elif not_present.shape[0] != X.shape[0]:
            if verbose == 1:
                print(f"Caricate {X.shape[0]-not_present.shape[0]} canzoni dalla cache, inizio caricamento delle restanti")
        else:
            if verbose == 1:
                print("Nessuna canzone presente in cache, inizio caricamento")
        return d, False
    
def feat_by_id_bar(X:pd.DataFrame, json_file:dict):
    """
    Performs the extraction of the feats of the tracks in the dataframe X\n
    and saves them in the json file that acts as a cache 

    Used if verbose of feat_dict is set to 1 so it shows a progress bar

    Parameters
    ----------
    X : pd.DataFrame
            The dataframe containing the tracks

    json_file : dict
                The dictionary containing the tracks already present in the cache

    """
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
    """
    Performs the extraction of the feats of the tracks in the dataframe X\n
    and saves them in the json file that acts as a cache

    Used if verbose of feat_dict is set to 0 so it doesn't show a progress bar

    Parameters
    ----------
    X : pd.DataFrame
            The dataframe containing the tracks

    json_file : dict
                The dictionary containing the tracks already present in the cache

    """
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
    """
    Calls the function that extracts the features of the tracks\n

    Parameters
    ----------
    X : pd.DataFrame
            The dataframe containing the tracks

    verbose : int, optional
                Sets the verbosity of the function\n
                if 1 prints are enabled\n
                if 0 prints are disabled\n

    path : str, optional
            Default value is the path of the json file in the environment variable JSON_PATH

    Returns
    -------
    A dictionary containing the features of the tracks associated with their id
    """

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