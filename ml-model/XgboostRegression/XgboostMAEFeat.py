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
      tree_method
      ):
    
    pipeline = Pipeline([
    ("feat_transformer", FeatTransformer(verbose=0)),
    ("artist_popularity_transformer", ArtistPopularityTransformer()),
    ("standard_scaler", StandardScaler(with_mean=with_mean, with_std=with_std)),
    ("imputer", SimpleImputer(strategy=strategy)),  
    ("xgboost", XGBRegressor(
        booster="gbtree",
        validate_parameters=validate_parameters,
        n_estimators=n_estimators,
        learning_rate=learnig_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        gamma=gamma,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        num_parallel_tree=num_parallel_tree,
        tree_method=tree_method
        )),
    ])
    return np.mean(cross_val_score(
        estimator=pipeline,
        X=X_train,
        y=y_train,
        cv=5,
        scoring="neg_mean_absolute_error"
    ))

def objective(trial):
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
        tree_method=tree_method
      )


dotenv.load_dotenv()

study = optuna.create_study(storage=os.getenv("MY_SQL_CONNECTION"),
                            study_name="Xgboost tree with Feat (MAE)",
                            direction="maximize",
                            load_if_exists=True)

study.optimize(objective, n_trials=100,n_jobs=-1)