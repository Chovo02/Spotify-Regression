import pandas as pd
import numpy as np
import sys
sys.path.append("ml-model")
from SpotifyCustomTransformer import FeatTransformer, ArtistPopularityTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import optuna
import dotenv
import os
from sklearn.impute import SimpleImputer

df = pd.read_csv("data\\SpotifySongPolularityAPIExtract.csv")
df.drop_duplicates(subset=['track_id'], keep='first', inplace=True)

X = df.drop(["popularity"], axis=1)
y = df["popularity"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def f(with_mean,
      with_std,
      strategy,
      n_estimators,
      criterion,
      max_depth,
      min_samples_split,
      min_samples_leaf,
      min_weight_fraction_leaf,
      max_features,
      max_leaf_nodes,
      bootstrap
      ):
    
    pipeline = Pipeline([
    ("feat_transformer", FeatTransformer(verbose=0)),
    ("artist_popularity_transformer", ArtistPopularityTransformer()),
    ("standard_scaler", StandardScaler(with_mean=with_mean, with_std=with_std)),
    ("imputer", SimpleImputer(strategy=strategy)),
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
    return np.mean(cross_val_score(
        estimator=pipeline,
        X=X_train,
        y=y_train,
        cv=5,
        scoring="r2"
    ))

def objective(trial):
    with_mean = trial.suggest_categorical("with_mean", [True, False])
    with_std = trial.suggest_categorical("with_std", [True, False])

    strategy = trial.suggest_categorical("strategy", ["mean", "median", "most_frequent"])

    n_estimators = trial.suggest_int("n_estimators", 100, 10000)
    criterion = trial.suggest_categorical("criterion", ["squared_error", "friedman_mse", "poisson"])
    max_depth = trial.suggest_int("max_depth", 10, 300)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 100)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 200)
    min_weight_fraction_leaf = trial.suggest_float("min_weight_fraction_leaf", 0, 0.5)
    max_features = trial.suggest_categorical("max_features", [None, "sqrt", "log2"])
    max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 10, 300)
    bootstrap = trial.suggest_categorical("bootstrap", [True, False])
    return f(with_mean,
      with_std,
      strategy,
      n_estimators,
      criterion,
      max_depth,
      min_samples_split,
      min_samples_leaf,
      min_weight_fraction_leaf,
      max_features,
      max_leaf_nodes,
      bootstrap
      )


dotenv.load_dotenv()

study = optuna.create_study(storage=os.getenv("MY_SQL_CONNECTION"),
                            study_name="Random Forest Regressor with Feat (R2)",
                            direction="maximize",
                            load_if_exists=True)

study.optimize(objective, n_jobs=-1, n_trials=100)