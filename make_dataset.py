from dataclasses import replace
import random
from sklearn.datasets import make_gaussian_quantiles
from collections import Counter
import numpy as np
import sys

rng = np.random.default_rng()

def create_donuts_dataset(
        n_samples=50,
        n_classes=2,
        n_erase=18,
        index=None,
        cov=2.0,
        random_state=0,
        train_data=True,
):
    ''' 
    create donuts dataset for experiment.

    Parameters
    ----------
    n_sanmples : int
        Number of generate data
    n_classes : int
        Number of classes
    index : list
        list of sample index  which will delete
    cov : int
        Covariance
    random_state : int
        Seed value for rng
    train_data : bool
        make train_data which is  imbalanced data with delete samples or not

    Returns
    ---------
    X : numpy.ndarray float64
        created data
    y : numpy.ndarray int64
        label of created data X
    '''
    X, y = make_gaussian_quantiles(
            n_samples=n_samples,
            n_features=1,
            n_classes=n_classes,
            random_state=0,
            shuffle=False,
            cov=cov,
    )
    one_index = np.where( y < 1 )
    if train_data :
        if index is None:
           np.random.seed(random_state)
           index = [np.random.randint(n_samples//2,n_samples-1) for p in range(0,n_erase)] 
        X = np.delete(X, index)
        y = np.delete(y, index)
    return X.reshape(-1,1), np.where(y==0, 1, 0) 


def create_separate_dataset(
    n_samples = 50,
    random_state= None,
    data_range = (0,5),
    region_bound = [1.4,2.8,3.9],
    decreased_sample = 0,
    hold_samples = False,
    ):
    if random_state != None: np.random.seed(random_state)
    
    rng = np.random.default_rng()
    begin,end = data_range
    if hold_samples:
        n_ma, n_mi = n_samples//2 + decreased_sample, n_samples//2 - decreased_sample
    else:
        n_ma, n_mi = n_samples//2, n_samples//2 - decreased_sample
    #rng = lambda s, e: (e - s) * np.random.rand() + s
    a = random.randint(1,n_ma)
    b = random.randint(1,n_mi)
    s = n_samples*
    n = [a,b, n_ma - a , n_mi - b]
    
    left_bound = 0
    right_bound = region_bound[0]
    
    
    X = rng.uniform(left_bound,right_bound,n[0])
    
    left_bound = right_bound
    right_bound = region_bound[1]
    
    X1 = np.append(X,rng.uniform(left_bound,right_bound,n[1]))
    
    left_bound = right_bound
    right_bound = region_bound[2]
    
    X2 = np.append(X1,rng.uniform(left_bound,right_bound,n[2]))
    
    left_bound = right_bound
    right_bound = 5.
    
    X3 = np.append(X2,rng.uniform(left_bound,right_bound,n[3]))
    y = [separate_data(i,region_bound) for i in X3]
    y = np.array(y)
    return X3.reshape(-1,1),y


def separate_data(x, bound_list):
    '''
    support function for create_separate_dataset.

    Parameters
    ----------
    x : float
        1d array of datapoints

    bound_list : list
        list of bound point
    Returns
    ----------
    label : int
        0 or 1
    '''
    a,b,c = bound_list
    if x < a or b < x < c : return 0
    else : return 1
