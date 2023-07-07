import pandas as pd
import numpy as np
import sys
sys.path.append("ml-model")
from SpotifyCustomTransformer import FeatTransformer, ArtistPopularityTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import optuna
import dotenv
import os
from sklearn.impute import SimpleImputer
from sklearn.svm import LinearSVR
df = pd.read_csv("data\\SpotifySongPolularityAPIExtract.csv")
df.drop_duplicates(subset=['track_id'], keep='first', inplace=True)

X = df.drop(["popularity"], axis=1)
y = df["popularity"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def f(with_mean,
      with_std,
      epsilon,
      tol,
      max_iter,
      strategy
      ):
    
    pipeline = Pipeline([
    ("feat_transformer", FeatTransformer(verbose=0)),
    ("artist_popularity_transformer", ArtistPopularityTransformer()),
    ("standard_scaler", StandardScaler(with_mean=with_mean, with_std=with_std)),
    ("imputer", SimpleImputer(strategy=strategy)),  
    ("Linear_SVR", LinearSVR(epsilon=epsilon, tol=tol, max_iter=max_iter))
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
    epsilon = trial.suggest_float("epsilon", 0.0, 1.0)
    tol=trial.suggest_float("tol", 0.0, 1.0)
    max_iter=trial.suggest_int("max_iter", 100, 1000)
    strategy = trial.suggest_categorical("strategy", ["mean", "median", "most_frequent"])

    return f(
        with_mean=with_mean,
        with_std=with_std,
        epsilon=epsilon,
        tol=tol,
        max_iter=max_iter,
        strategy=strategy
      )


dotenv.load_dotenv()

study = optuna.create_study(storage=os.getenv("MY_SQL_CONNECTION"),
                            study_name="Linear SVR with feat (R2)",
                            direction="maximize",
                            load_if_exists=True)

study.optimize(objective, n_jobs=-1, n_trials=100)