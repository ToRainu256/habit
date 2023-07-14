
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.neighbors import NearestNeighbors
import kde_kotaro as kdek
import nns

class KotaroMethod:
    def __init__(self, N=7, threshold = None, gamma = 10, distance_type = 'max') -> None:
        self.N = N
        self.threshold = threshold
        self.gamma = gamma
        self.distance_type = distance_type
        self.bandwidths = None
        self.kde = None
        #if threshold == None:
        #    raise ValueError('You have to give thresold value')

    def get_params(self, deep=False):
        params = {'N': self.N, 'threshold': self.threshold,'gamma': self.gamma,'distance_type':'max'}
        return params
    def set_params(self, **params):
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        self.N = params['N']
        self.threshold = params['threshold']
        self.gamma = params['gamma']
        self.distance_type = params['distance_type']
        return self
    
    def fit(self,  samples, y=None, **fit_params):
        _ = fit_params
        _ = y
        distances = nns.compute_distances(self.N, samples,distance_type=self.distance_type)
        self.kde = kdek.kernel_density_estimation(samples, np.abs(distances - self.gamma))

    def predict(self, X):
        density_estimation = self.kde(X)
        predict = np.where(density_estimation > self.threshold, -1 ,1 )
        return predict


