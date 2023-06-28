import pandas as pd
from sklearn.model_selection import train_test_split
import plotly.express as px
from sklearn.preprocessing import FunctionTransformer

def feat(x):
    if "feat." in x:
        return True
    else:
        return False

df = pd.read_csv("data/SpotifySongPolularityAPIExtract.csv", low_memory=False)



df["feat"] = df["track_name"].apply(lambda x: feat(x))

df_matrix = df.drop(["track_id", "track_name", "artist_name"], axis=1)
px.scatter_matrix(df_matrix.sample(1000)).show()