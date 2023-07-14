from sklearn.svm import SVC
import numpy as np

def gaussian_kernel(x1: np.array, x2: np.array, gamma:float) -> float:
        """ ガウスカーネルを計算する。
            k(x1, x2) = exp(-gamma*|x - x2|^2)

        Args:
            x1 (np.array)   : 入力値1
            x2 (np.array)   : 入力値2
            param (np.array): ガウスカーネルのパラメータ

        Returns:
            float: ガウスカーネルの値
        """

        return np.exp(-1*gamma*np.linalg.norm(x1 - x2)**2 )
class Kotaro_method:
    def __init__(self,kernel='precomputed',C=1.0):
        self.C = C
        self.kernel = kernel
        self.x = np.linspace(-10, 10)

    
    
    def _kernel(self,X,Y):
        gamma = 1/ (1*X.var())
        if X is Y:
            kmatrix = np.zeros((len(X),len(X)))
            for i in range(len(X)):
                for j in range(i + i, len(X)):
                    d = gaussian_kernel(X[i],X[j],gamma)
                    kmatrix[i,j] = kmatrix[j,i] = d
        else:
            kmatrix = np.zeros((len(X),len(Y)))
            for i in range(len(X)):
                for j in range(len(Y)):
                    d = gaussian_kernel(X[i],Y[j],gamma)
                    kmatrix[i,j] = d
        kmatrix = -kmatrix
        return kmatrix
    def fit(self, X, y,):
        '''
        This method fits the classifier to the training data

        Parameters
        ----------

        X : 2-D numpy array
            training input data

        y : 1-D numpy array
            training label data same length as X

        '''
    
        model = SVC(kernel=self._kernel)
        maj_index = np.where(y == 0)
        #X = np.array([X[i] for i in maj_index]).reshape(-1,1)
        #y = np.array([y[i] for i in maj_index]).reshape(25,)
        
        print(X.shape,y.shape)
        #kernel_matrix = self.calc_kernel_matrix(X,gaussian_kernel,gamma)
        #print(kernel_matrix)
        #print(y)
        model.fit(X,y)
        self.M = model

    def decision_function(self, X):
        '''
        This method returns the value of the decision function for the given input X

        Parameters
        ----------
        X : 1-D numpy array

        Returns
        ----------
        val_dec_func : 1-D numpy array

        '''
        self.Z = self.M.decision_function(self._kernel())
        #val_dec_func = self.Z.T - 0.05 * (1/self.dens)**2  + 1.1 * np.exp(self.dens)
        return self.Z

    def predict(self, X):
        '''
        This method returns the predict label for the given input X.

        Parameters
        ----------
        X : 1-D numpy array

        Returns
        ----------
        pred : 1-D numpy array

        '''
        pred = np.sign(self.decision_function(X))
        return pred
