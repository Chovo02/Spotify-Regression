from sklearn.base import BaseEstimator, TransformerMixin
from featbyid import feat_dict

class FeatTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self._y = y
        X["popularity"] = self._y
        self._avg_popularity = X.groupby("artist_name")["popularity"].mean().to_dict()
        X.drop("popularity", axis=1, inplace=True)
        return self
    
    def transform(self, X, y=None):
        X["popularity"] = self._y
        self._feat = feat_dict(X)
        market_list = {}
        feats_averaged = {}
        for key, items in self._feat.items():
            pop_list = [self._avg_popularity.get(artist, 0) for artist in items["feats"]]
            feats_averaged[key] = sum(pop_list) / len(pop_list) if pop_list else 0
            market_list[key] = items["markets"]
        X["feats_avg_popularity"] = X["track_id"].map(feats_averaged)
        X["markets"] = X["track_id"].map(market_list)
        X.drop("popularity", axis=1, inplace=True)
        return X
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)
    
class ArtistPopularityTransformer(BaseEstimator, TransformerMixin):
        
    def fit(self, X, y=None):
        self._y = y
        X["popularity"] = self._y
        self._avg_popularity = X.groupby("artist_name")["popularity"].mean().to_dict()
        X.drop("popularity", axis=1, inplace=True)
        return self

    def transform(self, X, y=None):
        X["artist_name"] = X["artist_name"].map(self._avg_popularity)
        X = X.drop(["track_id", "track_name"], axis=1)
        return X
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)

