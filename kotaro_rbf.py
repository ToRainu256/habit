
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.neighbors import NearestNeighbors
from rbf_gamma import CustomRBFClassifier
import nns
from sklearn.metrics import accuracy_score


class KotaroMethodRBF:
    def __init__(self, N=7,  gamma = 10, distance_type = 'max') -> None:
        self.N = N
        self.gamma = gamma
        self.distance_type = distance_type
        self.bandwidths = None
        self.rbf = None
        #if threshold == None:
        #    raise ValueError('You have to give thresold value')

    def get_params(self, deep=False):
        params = {'N': self.N, 'gamma': self.gamma,'distance_type':'max'}
        return params
    def set_params(self, **params):
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        self.N = params['N']
        self.gamma = params['gamma']
        self.distance_type = params['distance_type']
        return self
    
    def fit(self,  samples, y, **fit_params):
        _ = fit_params
        distances = nns.compute_distances(self.N, samples, distance_type=self.distance_type)
        self.rbf = CustomRBFClassifier(samples, y, 1.0/np.abs(self.gamma - distances))
        self.rbf.fit(samples, y, 1.0/np.abs(self.gamma - distances))

    def get_socre(self, X):
        socored = self.rbf.predict(X)
        return socored
    
    def score(self, X, y):
        predict = self.predict(X)
        score = accuracy_score(y, predict)
        return score
    
    def predict(self, X):
        socored = self.rbf.predict(X) 
        predict = np.where(socored < np.median(socored), -1 , 1 )
        return predict


