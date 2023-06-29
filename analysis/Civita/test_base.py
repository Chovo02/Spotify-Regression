import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv("data/SpotifySongPolularityAPIExtract.csv", low_memory=False)
df.drop_duplicates(subset=['track_id'], keep='first', inplace=True)
df.dropna(inplace=True)
df.drop(["track_id"], axis=1, inplace=True)

df_group = df.groupby(["artist_name"])["popularity"].mean()

X_train, X_test, y_train, y_test = train_test_split(df["artist_name"], df["popularity"], test_size=0.2, random_state=42)

y_hat = []
for X in X_train:
    y_hat.append(df_group[X])

print(f"MSE: {mean_squared_error(y_train, y_hat)}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_train, y_hat))}")
print(f"MAE: {mean_absolute_error(y_train, y_hat)}")