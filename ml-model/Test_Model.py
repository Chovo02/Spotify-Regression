import optuna
import pandas as pd
import os
import dotenv
from sklearn.pipeline import Pipeline
import pickle
import tekore
import featbyid 
import argparse
from argparse import RawTextHelpFormatter

dotenv.load_dotenv()

original_df = pd.read_csv("data\\SpotifySongPolularityAPIExtract.csv")


def tempo_classifier(x, percentile_33, percentile_66):
    if x < percentile_33:
        return 1
    elif x > percentile_33 and x < percentile_66:
        return 2
    elif x > percentile_66:
        return 3

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

def get_connection():
    return featbyid.get_connections()

def predict(link,model,score_type,with_feat):
    model_name = {
       "linear": "Linear Regression",
       "lasso": "Lasso Regression",
       "svr": "Linear SVR",
       "rf": "Random Forest Regressor",
       "ridge": "Ridge Regression",
       "sgd": "SGD Regressor",
       "xgb": "XGBoost Regressor"
    }[model]

    if with_feat:
        study_name = f"{model_name} with Feat ({score_type})"
    else:
        study_name = f"{model_name} ({score_type})"

    if score_type == "R2":
        score_type = "r2"
    elif score_type == "MAE":
        score_type = "neg_mean_absolute_error"
    elif score_type == "MSE":
        score_type = "neg_mean_squared_error"

    study_name=study_name.replace(" ","_")
    
    track_id=link.split("?")[0].split("/")[-1]
    connections = get_connection()
    artist,market = featbyid.get_feats_tekkore(track_id=track_id,current_connection=1,connections=connections)
    audio_features = featbyid.get_track_audio_features(track_id=track_id,current_connection=1,connections=connections)
    acousticness = audio_features.acousticness
    danceability = audio_features.danceability
    duration_ms = audio_features.duration_ms / 60000
    energy = audio_features.energy
    instrumentalness = audio_features.instrumentalness
    liveness = audio_features.liveness
    loudness = audio_features.loudness
    speechiness = audio_features.speechiness
    tempo = audio_features.tempo
    valence = audio_features.valence
    popularity,name = featbyid.get_track_popularity(track_id=track_id,current_connection=1,connections=connections)

    df = pd.DataFrame([[artist[0],acousticness,danceability,duration_ms,energy,instrumentalness,liveness,loudness,speechiness,tempo,valence,track_id,name]],columns=['artist_name','acousticness','danceability','duration_min','energy','instrumentalness','liveness','loudness','speechiness','tempo','valence','track_id','track_name'])
    tempo = df["tempo"].values[0]
    percentile_33 = original_df["tempo"].quantile(0.33)
    percentile_66 = original_df["tempo"].quantile(0.66)
    df["tempo"] = tempo_classifier(tempo, percentile_33, percentile_66)
    try:
        loaded_model = pickle.load(open(f"data\\{study_name}.sav", 'rb'))
    except:
        raise Exception("Modello non trovato")
    

    print(f"Predizione della canzone {name} di {artist[0]} in corso...")
    prediction = loaded_model.predict(df)
    print("Popolarità predetta: ",prediction[0])
    print("Popolarità reale: ",popularity)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict',formatter_class=RawTextHelpFormatter)
    parser.add_argument('--link', type=str, required=True, help='Link della canzone da predire')
    parser.add_argument('--model', type=str, required=True, choices=['lasso','linear','svr','rf','ridge','sgd','xgb'], 
                        help="""\nSeleziona il modello da utilizzare per predirre la popolarità\n\n--model lasso: Lasso Regression\n--model linear: Linear Regression\n--model svr: Linear SVR\n--model rf: Random Forest Regressor\n--model ridge: Ridge Regression\n--model sgd: SGD Regressor\n--model xgb: XGBoost Regressor """)
    parser.add_argument('--score', type=str, required=True, choices=['R2', 'MAE', 'MSE'], help='\nTipo di score utilizzato nel modello\n\n--score R2: R2\n--score MAE: Mean Absolute Error\n--score MSE: Mean Squared Error')
    parser.add_argument('--feat', dest='feat', action='store_true', help='Usa feat transformer')
    parser.add_argument('--nofeat', dest='feat', action='store_false', help='Non usa feat transformer')
    parser.set_defaults(feat=True)

    args = parser.parse_args()
    predict(args.link,args.model,args.score, args.feat)





