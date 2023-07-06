import pandas as pd
import numpy as np
import sys
sys.path.append("ml-model")
from SpotifyCustomTransformer import FeatTransformer, ArtistPopularityTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import optuna
import dotenv
import os
from sklearn.impute import SimpleImputer

df = pd.read_csv("../../data/SpotifySongPolularityAPIExtract.csv")
df.drop_duplicates(subset=['track_id'], keep='first', inplace=True)

X = df.drop(["popularity"], axis=1)
y = df["popularity"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def f(with_mean,
      with_std,
      fit_intercept,
      strategy,
      with_alpha
      ):
    
    pipeline = Pipeline([
    ("artist_popularity_transformer", ArtistPopularityTransformer()),
    ("standard_scaler", StandardScaler(with_mean=with_mean, with_std=with_std)),
    ("imputer", SimpleImputer(strategy=strategy)),  
    ("lasso_regression", Lasso(alpha=with_alpha, fit_intercept=fit_intercept))
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
    fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])
    strategy = trial.suggest_categorical("strategy", ["mean", "median", "most_frequent"])
    with_alpha = trial.suggest_float("with_alpha", 0.1, 1.0)

    return f(
        with_mean=with_mean,
        with_std=with_std,
        fit_intercept=fit_intercept,
        strategy=strategy,
        with_alpha = with_alpha
      )


dotenv.load_dotenv()

study = optuna.create_study(storage=os.getenv("MY_SQL_CONNECTION"),
                            study_name="Lasso Regression (MAE)",
                            direction="maximize",
                            load_if_exists=True)

study.optimize(objective, n_jobs=-1, n_trials=100)