#!/usr/bin/env python
# coding: utf-8
import pandas as pd
from ydata_profiling import ProfileReport
import plotly.express as px

df = pd.read_csv("SpotifySongPolularityAPIExtract.csv")
print(df.head(10))
print(df.info())
profile = ProfileReport(df,title="Profile Report")
profile.to_file("report.html")


import plotly.io as pio
pio.renderers.default = 'iframe' # or 'colab' or 'iframe' or 'iframe_connected' or 'sphinx_gallery'

# Calculate the percentage of NaN values in each column
nan_percentages = df.isnull().sum() / len(df) * 100
print(nan_percentages)
nan_percentages = nan_percentages.reset_index()
nan_percentages.columns = ['Column', 'Percentage']

# Plot the percentage of NaN values
fig = px.bar(nan_percentages, x='Column', y='Percentage', title='Percentage of NaN Values in Columns',
             labels={'Column': 'Columns', 'Percentage': 'Percentage'})
fig.update_layout(xaxis_tickangle=-45)
fig.show()


## Baseline

from sklearn.model_selection import train_test_split

features = df.drop(columns=["popularity","artist_name","track_id","track_name"]) # drop categorical var
target = df['popularity']
X = features
y = target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


### 1. linear regression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the Spotify dataset into a pandas DataFrame
df = pd.read_csv('SpotifySongPolularityAPIExtract.csv')

# Initialize and train the baseline algorithm (Linear Regression)
baseline_model = LinearRegression()
baseline_model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = baseline_model.predict(X_test)

# Evaluate the performance of the baseline algorithm
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
print(f'R-squared (R²): {r2:.2f}')


#### 2. dummyRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# create a dummy regressor
dummy_reg = DummyRegressor(strategy='mean')
# fit it on the training set
dummy_reg.fit(X_train, y_train)
# make predictions on the test set
y_pred = dummy_reg.predict(X_test)

# Evaluate the performance of the algorithm
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
print(f'R-squared (R²): {r2:.2f}')


## Models
from sklearn.model_selection import train_test_split

features = df.drop(columns=["popularity","artist_name","track_id","track_name"]) # drop categorical var
target = df['popularity']
X = features
y = target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def performance(y_test, y_pred):
    # measure the performance
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error (MSE): {mse:.2f}')
    print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
    print(f'R-squared (R²): {r2:.2f}')


# TODO: 
# + encoding categorical var
# + feature importance (ex. Lasso Reg)
# + polinomial regression
# + pipeline
# + optuna

### KNN

# libraries
import numpy as np
from sklearn.preprocessing import StandardScaler
#from sklearn.dummy import DummyRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

pipe = make_pipeline(
    StandardScaler(), KNeighborsRegressor(), verbose=True
)
# apply all the transformation on the training set and train an knn model
pipe.fit(X_train, y_train)
# apply all the transformation on the test set and make predictions
y_pred = pipe.predict(X_test)

# measure the performance
performance(y_test, y_pred)

### DummyRegressor
# libraries
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
#from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score


pipe = make_pipeline(
    StandardScaler(), DummyRegressor()
)
# apply all the transformation on the training set and train an knn model
pipe.fit(X_train, y_train)
# apply all the transformation on the test set and make predictions
y_pred = pipe.predict(X_test)

# measure the performance
performance(y_test, y_pred)