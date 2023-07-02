from sklearn.base import BaseEstimator, TransformerMixin
from featbyid import feat_dict

class FeatTransformer(BaseEstimator, TransformerMixin):
        
    def fit(self, X, y=None):
        self._feat = feat_dict(X)
        self._y = y
    
    def transform(self, X, y=None):
        X["popularity"] = self._y
        avg_popularity = X.groupby("artist_name")["popularity"].mean().to_dict()
        feats_averaged = {}

        for key, artists in self._feat.items():
            pop_list = [avg_popularity.get(artist, 0) for artist in artists]

            feats_averaged[key] = sum(pop_list) / len(pop_list) if pop_list else 0
        
        X["feats_avg_popularity"] = X["track_id"].map(feats_averaged)
        return X
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)
    
class ArtistPopularityTransformer(BaseEstimator, TransformerMixin):
        
    def fit(self, X, y=None):
        self.avg_popularity = X.groupby('artist_name')['popularity'].mean().to_dict()
        return self

    def transform(self, X, y=None):
        X['artist_name'] = X['artist_name'].map(self.avg_popularity)
        X = X.drop(['track_id', 'track_name'], axis=1)
        return X
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)

