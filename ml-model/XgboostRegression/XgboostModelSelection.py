import argparse
import pandas as pd
import numpy as np
import sys
sys.path.append("ml-model")
from SpotifyCustomTransformer import FeatTransformer, ArtistPopularityTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import optuna
import dotenv
import os
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

df = pd.read_csv("data\\SpotifySongPolularityAPIExtract.csv")
df.drop_duplicates(subset=['track_id'], keep='first', inplace=True)

X = df.drop(["popularity"], axis=1)
y = df["popularity"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def objective(trial, score_type, with_feat):
    with_mean = trial.suggest_categorical("with_mean", [True, False])
    with_std = trial.suggest_categorical("with_std", [True, False])
    strategy = trial.suggest_categorical("strategy", ["mean", "median", "most_frequent"])
    validate_parameters = trial.suggest_categorical("validate_parameters", [True, False])
    n_estimators = trial.suggest_int("n_estimators", 100, 1000, step=100)
    learnig_rate = trial.suggest_float("learnig_rate", 0.01, 0.2, step=0.01)
    max_depth = trial.suggest_int("max_depth", 3, 10, step=1)
    min_child_weight = trial.suggest_int("min_child_weight", 1, 10, step=1)
    gamma = trial.suggest_float("gamma", 0.0, 1, step=0.1)
    subsample = trial.suggest_float("subsample", 0.5, 1.0, step=0.1)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0, step=0.1)
    reg_alpha = trial.suggest_float("reg_alpha", 0.0, 1.0, step=0.1)
    num_parallel_tree = trial.suggest_int("num_parallel_tree", 1, 10, step=1)
    tree_method = trial.suggest_categorical("tree_method", ["auto", "exact", "approx", "hist"])
    
    return f(
        with_mean=with_mean,
        with_std=with_std,
        strategy=strategy,
        validate_parameters=validate_parameters,
        n_estimators=n_estimators,
        learnig_rate=learnig_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        gamma=gamma,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        num_parallel_tree=num_parallel_tree,
        tree_method=tree_method,
        score_type=score_type,
        with_feat=with_feat
    )

def f(with_mean,
      with_std,
      strategy,
      validate_parameters,
      n_estimators,
      learnig_rate,
      max_depth,
      min_child_weight,
      gamma,
      subsample,
      colsample_bytree,
      reg_alpha,
      num_parallel_tree,
      tree_method,
      score_type,
      with_feat
      ):
    
    if with_feat:
        pipeline = Pipeline([
            ("feat_transformer", FeatTransformer()),
            ("artist_popularity_transformer", ArtistPopularityTransformer()),
            ("imputer", SimpleImputer(strategy=strategy)),
            ("scaler", StandardScaler(with_mean=with_mean, with_std=with_std)),
            ("xgb", XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=learnig_rate,
                max_depth=max_depth,
                min_child_weight=min_child_weight,
                gamma=gamma,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_alpha=reg_alpha,
                num_parallel_tree=num_parallel_tree,
                tree_method=tree_method,
                validate_parameters=validate_parameters
            ))
        ])
    else:
        pipeline = Pipeline([
            ("artist_popularity_transformer", ArtistPopularityTransformer()),
            ("imputer", SimpleImputer(strategy=strategy)),
            ("scaler", StandardScaler(with_mean=with_mean, with_std=with_std)),
            ("xgb", XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=learnig_rate,
                max_depth=max_depth,
                min_child_weight=min_child_weight,
                gamma=gamma,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_alpha=reg_alpha,
                num_parallel_tree=num_parallel_tree,
                tree_method=tree_method,
                validate_parameters=validate_parameters
            ))
        ])
    print(pipeline) 
    print(f"Scoring metric used: {score_type}") 

    return np.mean(cross_val_score(
        estimator=pipeline,
        X=X_train,
        y=y_train,
        cv=5,
        scoring=score_type
    ))

def run_study(score_type, with_feat):
    dotenv.load_dotenv()

    if with_feat:
        study_name = f"Xgboost tree with Feat ({score_type})"
    else:
        study_name = f"Xgboost tree ({score_type})"

    if score_type == "R2":
        score_type = "r2"
    elif score_type == "MAE":
        score_type = "neg_mean_absolute_error"
    elif score_type == "MSE":
        score_type = "neg_mean_squared_error"


    study = optuna.create_study(storage=os.getenv("MY_SQL_CONNECTION_PROVA"),
                                study_name=study_name,
                                direction="maximize",
                                load_if_exists=True)

    study.optimize(lambda trial: objective(trial, score_type, with_feat), n_trials=100,n_jobs=-1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Selection.')
    parser.add_argument('--score', type=str, required=True, choices=['R2', 'MAE', 'MSE'], help='The score type: R2, MAE or MSE.')
    parser.add_argument('--feat', dest='feat', action='store_true', help='Use feat transformer.')
    parser.add_argument('--nofeat', dest='feat', action='store_false', help='Do not use feat transformer.')
    parser.set_defaults(feat=True)

    args = parser.parse_args()
    run_study(args.score, args.feat)
