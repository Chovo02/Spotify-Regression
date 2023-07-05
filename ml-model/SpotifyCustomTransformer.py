from sklearn.base import BaseEstimator, TransformerMixin
from featbyid import feat_dict

class FeatTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, verbose:int = 0) -> None:
        '''Constructor that initializes an instance variable "verbose" with a
        default value of 0.
        
        Parameters
        ----------
        verbose : int, optional
            The `verbose` parameter is an optional integer parameter that determines the level of verbosity
        of the code. It is used to control the amount of output or information that is displayed during
        the execution of the code.
        
        '''
        self.verbose = verbose

    def fit(self, X, y=None):
        '''The fit function calculates the average popularity of each artist in the dataset and stores it
        in a dictionary.
        
        Parameters
        ----------
        X
            The parameter X is a pandas DataFrame that contains the input features for training the model.
        Each row represents a sample, and each column represents a feature.
        y
            The parameter "y" represents the target variable or the dependent variable in the dataset. It
        is the variable that we are trying to predict or model. In this case, it seems to be the
        popularity of an artist.
        
        Returns
        -------
            the instance of the class itself (self).
        
        '''
        self.y = y
        X["popularity"] = self.y
        self.avg_popularity = X.groupby("artist_name")["popularity"].mean().to_dict()
        X.drop("popularity", axis=1, inplace=True)
        return self
    
    def transform(self, X, y=None):
        '''The `transform` function takes in a dataset `X` and adds a new column `feats_avg_popularity`
        based on the average popularity of the artists in each row.
        
        Parameters
        ----------
        X
            X is a pandas DataFrame containing the input data for the transformation. It has columns such
        as "popularity", "track_id", and other features.
        y
            The parameter "y" is a variable that represents the target variable or the dependent variable
        in a machine learning model. In this case, it seems to be the popularity of a track.
        
        Returns
        -------
            the modified DataFrame X.
        
        '''
        X["popularity"] = self.y
        self.feat = feat_dict(X, self.verbose)
        market_list = {}
        feats_averaged = {}
        for key, items in self.feat.items():
            pop_list = [self.avg_popularity.get(artist, 0) for artist in items["feats"]]
            feats_averaged[key] = sum(pop_list) / len(pop_list) if pop_list else 0
        X["feats_avg_popularity"] = X["track_id"].map(feats_averaged)
        X.drop("popularity", axis=1, inplace=True)
        return X
    
    def fit_transform(self, X, y=None):
        '''The fit_transform function fits the data to a model and then transforms it.
        
        Parameters
        ----------
        X
            The input data matrix, where each row represents a sample and each column represents a feature.
        y
            The parameter "y" is the target variable or the dependent variable. It represents the labels or
        classes that we want to predict or classify. In some machine learning algorithms, the target
        variable is not required, so it can be set to None.
        
        Returns
        -------
            The fit_transform method returns the result of calling the transform method on the input data
        X.
        
        '''
        self.fit(X, y)
        return self.transform(X, y)
    
class ArtistPopularityTransformer(BaseEstimator, TransformerMixin):
        
    def fit(self, X, y=None):
        '''The fit function calculates the average popularity of each artist in the dataset and stores it
        in a dictionary.
        
        Parameters
        ----------
        X
            The parameter X is a pandas DataFrame that contains the input features for training the model.
        Each row represents a sample, and each column represents a feature.
        y
            The parameter "y" represents the target variable or the dependent variable in the dataset. It
        is the variable that we are trying to predict or model. In this case, it seems to be the
        popularity of an artist.
        
        Returns
        -------
            the instance of the class itself (self).
        
        '''
        self.y = y
        X["popularity"] = self.y
        self.avg_popularity = X.groupby("artist_name")["popularity"].mean().to_dict()
        X.drop("popularity", axis=1, inplace=True)
        return self

    def transform(self, X, y=None):
        '''The function transforms a DataFrame by mapping the "artist_name" column to average popularity
        values and dropping the "track_id" and "track_name" columns.
        
        Parameters
        ----------
        X
            The parameter X is a pandas DataFrame that contains the input data. It has columns named
        "artist_name", "track_id", and "track_name".
        y
            The parameter "y" is the target variable or the labels. It is used for supervised learning
        tasks where we have both input features (X) and corresponding target values (y). In this
        specific code snippet, the "y" parameter is not used, so it is set to None.
        
        Returns
        -------
            the transformed dataset, X.
        
        '''
        X["artist_name"] = X["artist_name"].map(self.avg_popularity)
        X = X.drop(["track_id", "track_name"], axis=1)
        return X
    
    def fit_transform(self, X, y=None):
        '''The fit_transform function fits the data to a model and then transforms it.
        
        Parameters
        ----------
        X
            The input data matrix, where each row represents a sample and each column represents a feature.
        y
            The parameter "y" is the target variable or the labels associated with the input data "X". It
        is optional and can be None if the transformation does not require any target variable.
        
        Returns
        -------
            The fit_transform method returns the transformed data after fitting the model to the input
        data.
        
        '''
        self.fit(X, y)
        return self.transform(X, y)

