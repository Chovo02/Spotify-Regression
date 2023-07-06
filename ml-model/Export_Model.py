import optuna
import os
import dotenv
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import sys
from SpotifyCustomTransformer import FeatTransformer, ArtistPopularityTransformer
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
from argparse import RawTextHelpFormatter
from sklearn.linear_model import Lasso
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error


df = pd.read_csv("data\\SpotifySongPolularityAPIExtract.csv")
df.drop_duplicates(subset=['track_id'], keep='first', inplace=True)
df.dropna(inplace=True)

keywords = ["podcast", "mix", "rain", "intro", "outro", "dj", "sleep"]
mask = df['track_name'].str.lower().str.contains('|'.join(keywords))
df = df[~mask]

df = df[df["time_signature"] != 0]
df = df[df["tempo"] != 0]

X = df.drop(["popularity","mode","key","time_signature","tempo"], axis=1)
y = df["popularity"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sys.path.append("ml-model")

dotenv.load_dotenv()

def get_best_pipeline(model_name, best_params, with_feat):
    MODELS = {
        "linear": LinearRegression,
        "lasso": Lasso,
        "svr": LinearSVR,
        "rf": RandomForestRegressor,
        "ridge": Ridge,
        "sgd": SGDRegressor,
        "xgb": XGBRegressor
    }
    model = MODELS[model_name]()
    if with_feat:
        steps = [
            ("feat_transformer", FeatTransformer()),
            ("artist_popularity_transformer", ArtistPopularityTransformer()),
            ("imputer", SimpleImputer(strategy=best_params.pop("strategy"))),
            ("scaler", StandardScaler(with_mean=best_params.pop("with_mean"), with_std=best_params.pop("with_std"))),
            (model_name, model)
        ]
    else:
        steps = [
            ("artist_popularity_transformer", ArtistPopularityTransformer()),
            ("imputer", SimpleImputer(strategy=best_params.pop("strategy"))),
            ("scaler", StandardScaler(with_mean=best_params.pop("with_mean"), with_std=best_params.pop("with_std"))),
            (model_name, model)
        ]
    pipeline = Pipeline(steps)

    model_params = {f"{model_name}__{key}": value for key, value in best_params.items()}
    pipeline.set_params(**model_params)
    return pipeline

def export_model(model,score_type, with_feat):
    dotenv.load_dotenv()
    if model == "linear":
          model_name = "Linear Regression"
    elif model == "lasso":
            model_name = "Lasso Regression"
    elif model == "svr":
            model_name = "Linear SVR"
    elif model == "rf":
            model_name = "Random Forest Regressor"
    elif model == "ridge":
            model_name = "Ridge Regression"
    elif model == "sgd":
            model_name = "SGD Regressor"
    elif model == "xgb":
            model_name = "XGBoost Regressor"

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

    try:
        study = optuna.load_study(study_name=study_name, storage=os.getenv("MY_SQL_CONNECTION_PROVA"))
    except:
        print(f"Studio ({study_name}) non trovato su {os.getenv('MY_SQL_CONNECTION_PROVA')}")
        return

    best_params = study.best_params
    pipeline = get_best_pipeline(model_name=model, best_params=best_params,with_feat=with_feat)
    pipeline.fit(X_train, y_train)
    study_name=study_name.replace(" ","_")
    pickle.dump(pipeline, open(f"data\\{study_name}.sav", "wb"))
    print(f"Modello {study_name} esportato correttamente")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Selection manager',formatter_class=RawTextHelpFormatter)
    parser.add_argument('--model', type=str, required=True, choices=['lasso','linear','svr','rf','ridge','sgd','xgb'], 
                        help="""\nSeleziona il modello da esportare\n\n--model lasso: Lasso Regression\n--model linear: Linear Regression\n--model svr: Linear SVR\n--model rf: Random Forest Regressor\n--model ridge: Ridge Regression\n--model sgd: SGD Regressor\n--model xgb: XGBoost Regressor """)
    parser.add_argument('--score', type=str, required=True, choices=['R2', 'MAE', 'MSE'], help='\nTipo di score utilizzato nel modello\n\n--score R2: R2\n--score MAE: Mean Absolute Error\n--score MSE: Mean Squared Error')
    parser.add_argument('--feat', dest='feat', action='store_true', help='Usa feat transformer')
    parser.add_argument('--nofeat', dest='feat', action='store_false', help='Non usa feat transformer')
    parser.set_defaults(feat=True)

    args = parser.parse_args()
    export_model(args.model,args.score, args.feat)





