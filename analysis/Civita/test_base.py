import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("data/SpotifySongPolularityAPIExtract.csv", low_memory=False)
df.drop_duplicates(subset=['track_id'], keep='first', inplace=True)
df.dropna(inplace=True)

X_train, X_test, y_train, y_test = train_test_split(df[["artist_name"]], df["popularity"], test_size=0.2, random_state=42)

X_train["popularity"] = y_train

X_group = X_train.groupby(["artist_name"])["popularity"].mean()

X_train.drop(["popularity"], axis=1, inplace=True)


y_hat = []
for X in X_test["artist_name"]:
    try:
        y_hat.append(X_group[X])
    except KeyError:
        y_hat.append(X_group.values.mean())

print(f"MAE: {mean_absolute_error(y_test, y_hat)}")
print(f"MSE: {mean_squared_error(y_test, y_hat)}")
print(f"R2: {r2_score(y_test, y_hat)}")