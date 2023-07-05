import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import  mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn import set_config
set_config(transform_output="pandas")
pd.set_option('display.max_columns', None)
df = pd.read_csv(r"C:\Users\DavideSoltys\OneDrive - ITS Angelo Rizzoli\Desktop\ML progetto\Spotify-Regression\data\SpotifySongPolularityAPIExtract.csv")
df.drop_duplicates(subset=['track_id'], keep='first', inplace=True)


df_target = df["popularity"]
df.drop("popularity",axis= 1, inplace=True)


X_train, X_test, y_train, y_test = train_test_split(df, df_target, test_size=0.2, random_state=42)
#print(X_train)
# X train lavoro
X_train["popularity"] = y_train
# df_nuovo = X_train.groupby("artist_name")["popularity"].mean()
# X_train["singola_popolarita"] = X_train["artist_name"].map(df_nuovo)
X_train.drop(["track_id", "track_name", "artist_name", "popularity"], axis=1, inplace=True)
#print(X_train)


# X test

X_test["popularity"] = y_test
# df_nuovo = X_test.groupby("artist_name")["popularity"].mean()
# X_test["singola_popolarita"] = X_test["artist_name"].map(df_nuovo)
X_test.drop(["track_id", "track_name", "artist_name", "popularity"], axis=1, inplace=True)




#transfrom
print(X_test)
standard_scaler = StandardScaler()
standard_scaler.fit(X_train)
X_test = standard_scaler.transform(X_test)
X_train = standard_scaler.transform(X_train)

print(X_train)
print(X_test)


#linear regression
linear_regressor = LinearRegression()
reg = linear_regressor.fit(X_train, y_train)
y_hat = (reg.predict(X_test))
print(reg.score(X_test,y_test))

print(mean_absolute_error(y_test, y_hat))




