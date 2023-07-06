import argparse
from argparse import RawTextHelpFormatter
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
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor

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

def objective(trial, model, score_type, with_feat):
    with_mean = trial.suggest_categorical("with_mean", [True, False])
    with_std = trial.suggest_categorical("with_std", [True, False])
    strategy = trial.suggest_categorical("strategy", ["mean", "median", "most_frequent"])

    if model == "lasso":
        with_alpha = trial.suggest_float("with_alpha", 0.1, 1.0)
        fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])
        return f(
            model=model,
            with_mean=with_mean,
            with_std=with_std,
            strategy=strategy,
            with_alpha = with_alpha,
            fit_intercept=fit_intercept,
            score_type=score_type,
            with_feat=with_feat
        )
    elif model == "linear":
        fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])
        return f(
            model=model,
            with_mean=with_mean,
            with_std=with_std,
            strategy=strategy,
            fit_intercept=fit_intercept,
            score_type=score_type,
            with_feat=with_feat)
    elif model == "svr":
        epsilon = trial.suggest_float("epsilon", 0.0, 1.0)
        tol = trial.suggest_float("tol", 0.0, 1.0)
        max_iter = trial.suggest_int("max_iter", 100, 1000)
        return f(
            model=model,
            with_mean=with_mean,
            with_std=with_std,
            strategy=strategy,
            epsilon=epsilon,
            tol=tol,
            max_iter=max_iter,
            score_type=score_type,
            with_feat=with_feat)
    elif model == "rf":
        n_estimators = trial.suggest_int("n_estimators", 100, 10000)
        criterion = trial.suggest_categorical("criterion", ["squared_error", "friedman_mse", "poisson"])
        max_depth = trial.suggest_int("max_depth", 10, 300)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 100)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 200)
        min_weight_fraction_leaf = trial.suggest_float("min_weight_fraction_leaf", 0, 0.5)
        max_features = trial.suggest_categorical("max_features", [None, "sqrt", "log2"])
        max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 10, 300)
        bootstrap = trial.suggest_categorical("bootstrap", [True, False])
        return f(
            model=model,
            with_mean=with_mean,
            with_std=with_std,
            strategy=strategy,
            score_type=score_type,
            with_feat=with_feat,
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            bootstrap=bootstrap)
    elif model == "ridge":
        alpha = trial.suggest_float("alpha", 0 , 25)
        fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])
        return f(
            model=model,
            with_mean=with_mean,
            with_std=with_std,
            strategy=strategy,
            alpha=alpha,
            fit_intercept=fit_intercept,
            score_type=score_type,
            with_feat=with_feat)
    elif model == "sgd":
        loss = trial.suggest_categorical("loss", ["squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"])
        penalty = trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"])
        alpha = trial.suggest_float("alpha", 1e-8, 1e-1, log=True)
        l1_ratio = trial.suggest_float("l1_ratio", 0.1, 1.0)
        fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])
        max_iter = trial.suggest_int("max_iter", 100, 2000)
        tol = trial.suggest_float("tol", 1e-8, 1e-1, log=True)
        return f(
            model=model,
            with_mean=with_mean,
            with_std=with_std,
            strategy=strategy,
            loss=loss,
            penalty=penalty,
            alpha=alpha,
            l1_ratio=l1_ratio,
            score_type=score_type,
            with_feat=with_feat)
    elif model == "xgb":
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
            model=model,
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
            with_feat=with_feat)

def f(model,
      with_mean,
      with_std,
      strategy,
      score_type,
      with_feat,
      fit_intercept=None,
      with_alpha=None,
      epsilon=None,
      tol=None,
      max_iter=None,
      n_estimators=None,
      criterion=None,
      max_depth=None,
      min_samples_split=None,
      min_samples_leaf=None,
      min_weight_fraction_leaf=None,
      max_features=None,
      max_leaf_nodes=None,
      bootstrap=None,
      alpha=None,
      loss=None,
      penalty=None,
      l1_ratio=None,
      validate_parameters=None,
      learnig_rate=None,
      min_child_weight=None,
      gamma=None,
      subsample=None,
      colsample_bytree=None,
      reg_alpha=None,
      num_parallel_tree=None,
      tree_method=None
      ):
    
    if with_feat:
        if model == "linear":
            pipeline = Pipeline([
                ("feat_transformer", FeatTransformer()),
                ("artist_popularity_transformer", ArtistPopularityTransformer()),
                ("imputer", SimpleImputer(strategy=strategy)),
                ("scaler", StandardScaler(with_mean=with_mean, with_std=with_std)),
                ("linear_regression", LinearRegression(n_jobs=-1, fit_intercept=fit_intercept))
            ])
        elif model == "lasso":
            pipeline = Pipeline([
                ("feat_transformer", FeatTransformer()),
                ("artist_popularity_transformer", ArtistPopularityTransformer()),
                ("imputer", SimpleImputer(strategy=strategy)),
                ("scaler", StandardScaler(with_mean=with_mean, with_std=with_std)),
                ("lasso_regression", Lasso(alpha=with_alpha, fit_intercept=fit_intercept))
            ])
        elif model == "svr":
            pipeline = Pipeline([
                ("feat_transformer", FeatTransformer()),
                ("artist_popularity_transformer", ArtistPopularityTransformer()),
                ("imputer", SimpleImputer(strategy=strategy)),
                ("scaler", StandardScaler(with_mean=with_mean, with_std=with_std)),
                ("Linear_SVR", LinearSVR(epsilon=epsilon, tol=tol, max_iter=max_iter))
            ])
        elif model == "rf":
            pipeline = Pipeline([
                ("feat_transformer", FeatTransformer()),
                ("artist_popularity_transformer", ArtistPopularityTransformer()),
                ("imputer", SimpleImputer(strategy=strategy)),
                ("scaler", StandardScaler(with_mean=with_mean, with_std=with_std)),
                ("random_forest_regressor", RandomForestRegressor(n_estimators=n_estimators, 
                                                      criterion=criterion, 
                                                      max_depth=max_depth, 
                                                      min_samples_split=min_samples_split,
                                                      min_samples_leaf=min_samples_leaf,
                                                      min_weight_fraction_leaf=min_weight_fraction_leaf,
                                                      max_features=max_features,
                                                      max_leaf_nodes=max_leaf_nodes,
                                                      bootstrap=bootstrap,
                                                      n_jobs=-1))   
            ])
        elif model == "ridge":
                pipeline = Pipeline([
                    ("feat_transformer", FeatTransformer(verbose=0)),
                    ("artist_popularity_transformer", ArtistPopularityTransformer()),
                    ("standard_scaler", StandardScaler(with_mean=with_mean, with_std=with_std)),
                    ("imputer", SimpleImputer(strategy=strategy)),  
                    ("ridge_regression", Ridge(alpha=alpha, fit_intercept=fit_intercept))
                    ])
        elif model == "sgd":
                 pipeline = Pipeline([
                    ("feat_transformer", FeatTransformer(verbose=0)),
                    ("artist_popularity_transformer", ArtistPopularityTransformer()),
                    ("standard_scaler", StandardScaler(with_mean=with_mean, with_std=with_std)),
                    ("imputer", SimpleImputer(strategy=strategy)),
                    ("sgd_regressor", SGDRegressor(loss=loss,
                                                penalty=penalty,
                                                alpha=alpha,
                                                l1_ratio=l1_ratio,
                                                fit_intercept=fit_intercept,
                                                max_iter=max_iter,
                                                tol=tol))    
                    ])
        elif model == "xgb":
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
        if model == "linear":
            pipeline = Pipeline([
                ("artist_popularity_transformer", ArtistPopularityTransformer()),
                ("imputer", SimpleImputer(strategy=strategy)),
                ("scaler", StandardScaler(with_mean=with_mean, with_std=with_std)),
                ("linear_regression", LinearRegression(n_jobs=-1, fit_intercept=fit_intercept))
            ])
        elif model == "lasso":
            pipeline = Pipeline([
                ("artist_popularity_transformer", ArtistPopularityTransformer()),
                ("imputer", SimpleImputer(strategy=strategy)),
                ("scaler", StandardScaler(with_mean=with_mean, with_std=with_std)),
                ("lasso_regression", Lasso(alpha=with_alpha, fit_intercept=fit_intercept))
            ])
        elif model == "svr":
            pipeline = Pipeline([
                ("artist_popularity_transformer", ArtistPopularityTransformer()),
                ("imputer", SimpleImputer(strategy=strategy)),
                ("scaler", StandardScaler(with_mean=with_mean, with_std=with_std)),
                ("Linear_SVR", LinearSVR(epsilon=epsilon, tol=tol, max_iter=max_iter))
            ])
        elif model == "rf":
                        pipeline = Pipeline([
                ("artist_popularity_transformer", ArtistPopularityTransformer()),
                ("imputer", SimpleImputer(strategy=strategy)),
                ("scaler", StandardScaler(with_mean=with_mean, with_std=with_std)),
                ("random_forest_regressor", RandomForestRegressor(n_estimators=n_estimators, 
                                                      criterion=criterion, 
                                                      max_depth=max_depth, 
                                                      min_samples_split=min_samples_split,
                                                      min_samples_leaf=min_samples_leaf,
                                                      min_weight_fraction_leaf=min_weight_fraction_leaf,
                                                      max_features=max_features,
                                                      max_leaf_nodes=max_leaf_nodes,
                                                      bootstrap=bootstrap,
                                                      n_jobs=-1))   
            ])
        elif model == "ridge":
                pipeline = Pipeline([
                    ("artist_popularity_transformer", ArtistPopularityTransformer()),
                    ("standard_scaler", StandardScaler(with_mean=with_mean, with_std=with_std)),
                    ("imputer", SimpleImputer(strategy=strategy)),  
                    ("ridge_regression", Ridge(alpha=alpha, fit_intercept=fit_intercept))
                    ])
        elif model == "sgd":
                              pipeline = Pipeline([
                    ("artist_popularity_transformer", ArtistPopularityTransformer()),
                    ("standard_scaler", StandardScaler(with_mean=with_mean, with_std=with_std)),
                    ("imputer", SimpleImputer(strategy=strategy)),
                    ("sgd_regressor", SGDRegressor(loss=loss,
                                                penalty=penalty,
                                                alpha=alpha,
                                                l1_ratio=l1_ratio,
                                                fit_intercept=fit_intercept,
                                                max_iter=max_iter,
                                                tol=tol))    
                    ])
        elif model == "xgb":
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

    return np.mean(cross_val_score(
        estimator=pipeline,
        X=X_train,
        y=y_train,
        cv=5,
        scoring=score_type
    ))

def run_study(model,score_type, with_feat):
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

    study = optuna.create_study(storage=os.getenv("MY_SQL_CONNECTION_PROVA"),
                                study_name=study_name,
                                direction="maximize",
                                load_if_exists=True)

    study.optimize(lambda trial: objective(trial, model, score_type, with_feat), n_trials=100)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Selection manager',formatter_class=RawTextHelpFormatter)
    parser.add_argument('--model', type=str, required=True, choices=['lasso','linear','svr','rf','ridge','sgd','xgb'], 
                        help="""\nSeleziona il modello da utilizzare\n\n--model lasso: Lasso Regression\n--model linear: Linear Regression\n--model svr: Linear SVR\n--model rf: Random Forest Regressor\n--model ridge: Ridge Regression\n--model sgd: SGD Regressor\n--model xgb: XGBoost Regressor """)
    parser.add_argument('--score', type=str, required=True, choices=['R2', 'MAE', 'MSE'], help='\nTipo di score da utilizzare\n\n--score R2: R2\n--score MAE: Mean Absolute Error\n--score MSE: Mean Squared Error')
    parser.add_argument('--feat', dest='feat', action='store_true', help='Usa feat transformer')
    parser.add_argument('--nofeat', dest='feat', action='store_false', help='Non usa feat transformer')
    parser.set_defaults(feat=True)

    args = parser.parse_args()
    run_study(args.model,args.score, args.feat)
