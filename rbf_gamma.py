import numpy as np
from scipy.interpolate import Rbf
from numba import jit, prange
import numpy as np

class CustomRBFClassifier:
    def __init__(self, x, y, epsilon):
        self.x = x
        self.y = y
        self.epsilon = epsilon
        self.lambda_ = self._compute_lambda()

    # ユークリッド距離計算関数
    #@jit(fastmath = True)
    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1-x2)**2))

    # RBF関数定義（ここではガウス関数を使用）
    #@jit(fastmath=True)
    def _rbf_gauss(self, x, c, epsilon):
        distance = self._euclidean_distance(x, c)
        return np.exp(-1 * distance**2 / (2 * epsilon**2))

    # パラメータの計算
    
    def _compute_lambda(self):
        n = len(self.x)
        A = np.zeros((n, n))
        for i in prange(n):
            for j in prange(n):
                A[i,j] = self._rbf_gauss(self.x[i], self.x[j], self.epsilon[j])
        return np.linalg.solve(A, self.y)

    # 補間関数の定義
    #@jit(fastmath=True)
    def interpolate(self, x0):
        return sum(self.lambda_[i] * self._rbf_gauss(x0, self.x[i], self.epsilon[i]) for i in range(len(self.x)))


    def fit(self, points, labels, epsilon):
        """
        points: shape=(n_points, n_dims)の配列。サンプル点の座標
        labels: shape=(n_points,)の配列。サンプル点のラベル（-1 または 1）
        gammas: shape=(n_points,)の配列。各サンプル点に対応するgamma
        """
        n_points = len(points)
        #assert len(labels) == n_points
        #assert len(gammas) == n_points
        #assert set(labels) == {-1, 1}, "Labels should be either -1 or 1."
        
        

 
    def predict(self, query_points):
        """
        query_points: shape=(n_queries, n_dims)の配列。問い合わせ点の座標
        """
       
        scores = np.array([self.interpolate(np.array([x]))for x in query_points])
        #print(scores)
        #scores = scores.reshape(query_points.shape)
        return scores

       

    # RBF関数定義（ここではガウス関数を使用）
    