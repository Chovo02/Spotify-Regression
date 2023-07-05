import json
import dotenv
import os
from alive_progress import alive_bar
import tekore as tk
import pandas as pd
import logging

def get_env(key:str):
    '''The function `get_env` retrieves the value of an environment variable specified by the `key`
    parameter.
    
    Parameters
    ----------
    key : str
        The `key` parameter is a string that represents the name of the environment variable that you want
    to retrieve.
    
    Returns
    -------
        the value of the environment variable specified by the key parameter.
    
    '''
    dotenv.load_dotenv()
    return os.environ[key]

def get_connections():
    '''The function `get_connections` retrieves client IDs and secrets from environment variables and
    returns a list of Spotify connections using those credentials.
    
    Returns
    -------
        a list of Spotify client connections.
    
    '''
    client_ids = get_env("CLIENT_ID_LIST").split(",")
    client_secrets = get_env("CLIENT_SECRET_LIST").split(",")
    return [tk.Spotify(tk.request_client_token(client_id, client_secret)) for client_id, client_secret in zip(client_ids, client_secrets)]

def save_data(path:str = get_env("JSON_PATH"), data:dict = {}):
    '''The function `save_data` takes a path and a dictionary as input, loads the existing data from the
    file at the given path, updates it with the new data, and saves the updated data back to the file.
    
    Parameters
    ----------
    path : str
        The `path` parameter is a string that represents the file path where the JSON data will be saved.
    It is optional and has a default value of `get_env("JSON_PATH")`. This suggests that the function is
    trying to retrieve the file path from an environment variable named "JSON_PATH". If
    data : dict
        The `data` parameter is a dictionary that contains the data that you want to save. It will be
    merged with the existing data in the file specified by the `path` parameter. If the file does not
    exist, a new file will be created with the specified `path` and the `data
    
    Returns
    -------
        a boolean value. It returns True if the data is successfully saved to the specified path, and False
    if there is an exception or error during the process.
    
    '''
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
    '''The function "get_feats_tekkore" takes a track ID, a current connection, and a dictionary of
    connections as input, and returns a list of artist names and a flag indicating if the track is
    available in any markets.
    
    Parameters
    ----------
    track_id
        The track ID is a unique identifier for a specific track in the Spotify database. It is used to
    retrieve information about that track, such as its artists and available markets.
    current_connection
        The current_connection parameter is the index of the current connection being used. It is used to
    access the correct connection object from the connections list.
    connections
        The `connections` parameter is a dictionary that contains different Spotify connections. Each
    connection is identified by a key, and the value is the actual connection object.
    
    Returns
    -------
        a list of artist names and a flag indicating whether the track is available in any markets.
    
    '''
    connection = connections[current_connection]
    result = connection.track(track_id)
    artists = result.artists
    markets = result.available_markets
    return [artist.name for artist in artists],0 if len(markets) == 0 else 1

def load_json(X:pd.DataFrame, verbose:int = 1, path:str = get_env("JSON_PATH")):
    '''The function `load_json` loads a JSON file and checks if all the track IDs in a given DataFrame are
    present in the JSON file, returning the loaded JSON data and a boolean indicating if all track IDs
    were already present in the JSON file.
    
    Parameters
    ----------
    X : pd.DataFrame
        The parameter X is a pandas DataFrame that contains the track_id column. This column is used to
    check if the track is already present in the JSON cache file.
    verbose : int, optional
        The `verbose` parameter is used to control the level of output messages during the execution of the
    function.
    path : str
        The `path` parameter is a string that represents the file path to the JSON file that needs to be
    loaded. It is set to the value of the `JSON_PATH` environment variable by default.
    
    Returns
    -------
        a tuple containing two elements. The first element is the dictionary `d`, which is loaded from a
    JSON file. The second element is a boolean value indicating whether all the songs in the input
    DataFrame `X` are already present in the cache.
    
    '''
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
    '''The function `feat_by_id_bar` takes a DataFrame `X` and a JSON file `json_file` as input, and
    iterates over the track IDs in `X` to retrieve features and markets for each track ID, storing the
    results in a dictionary.
    
    Parameters
    ----------
    X : pd.DataFrame
        X is a pandas DataFrame containing the track IDs.
    json_file : dict
        The `json_file` parameter is a dictionary that contains track IDs as keys and their corresponding
    features as values. It is used to check if a track ID already exists in the dictionary before
    fetching its features.
    
    '''
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
    '''The function `feat_by_id` takes a DataFrame `X` and a JSON file `json_file` as input, and iterates
    through the `track_id` column of `X` to retrieve features and markets for each track ID, storing the
    results in a dictionary.
    
    Parameters
    ----------
    X : pd.DataFrame
        X is a pandas DataFrame containing the track IDs.
    json_file : dict
        The `json_file` parameter is a dictionary that contains information about tracks. It is used to
    check if a track ID already exists in the dictionary before processing it.
    
    '''
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
    '''The function `feat_dict` loads a JSON file, performs feature engineering on a DataFrame, and returns
    the resulting dictionary.
    
    Parameters
    ----------
    X : pd.DataFrame
        The parameter `X` is a pandas DataFrame that contains the data for which we want to create a
    feature dictionary.
    verbose : int, optional
        The `verbose` parameter is used to control the level of detail in the output. If `verbose` is set
    to 0, the function will use the `feat_by_id` function to generate the feature dictionary. If
    `verbose` is set to any other value, the function will use the
    path : str
        The "path" parameter is a string that represents the path to the JSON file. It is used to load the
    JSON file in the "load_json" function. If the "path" parameter is not provided, it will use the
    value returned by the "get_env" function with the argument "
    
    Returns
    -------
        a dictionary object.
    
    '''
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