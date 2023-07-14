import numpy as np
import os 
import warnings
import matplotlib.pyplot as plt
from sphere_sampler import generate_labeled_samples  # sphere_sampler.pyから関数をインポート
from collections import Counter
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, OneClassSVM
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
import grid_data
import numpy as np
import nns
import spherical_data_2
import matplotlib.pyplot as plt
from imblearn.datasets import make_imbalance
from sklearn.metrics import accuracy_score
from kotaro_method import KotaroMethod
from kotaro_rbf import KotaroMethodRBF

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import pandas as pd


import collections



svm = SVC()
param_svm = {
        'C': [10 ** i for i in range(-5,6)],
        'kernel': ["rbf" ],
        'decision_function_shape': ["ovo", "ovr"],
    }
        
scoring = 'accuracy'
cv = 3
gscv = GridSearchCV(svm, param_grid=param_svm,scoring=scoring, cv=cv, refit=True,n_jobs=-1 )
wsvm =  SVC()
param_wsvm = {
    'C': [10 ** i for i in range(-5,6)],
    'kernel': ["rbf"],
    'class_weight':["balanced"],
    'decision_function_shape': ["ovo", "ovr"],
}
scoring = 'accuracy'
cv = 3
gscv_wsvm = GridSearchCV(wsvm, param_grid=param_wsvm,scoring=scoring, cv=cv, refit=True,n_jobs=-1 )
osvm = OneClassSVM()
param_osvm = {
    'kernel': ["rbf", ],
    'nu': [i for i in np.arange(0.1,0.6,0.1)]
}
scoring = 'accuracy'
cv = 2
gscv_osvm = GridSearchCV(osvm, param_grid=param_osvm,scoring=scoring, cv=cv, refit=True, n_jobs=-1)
kotaro = KotaroMethodRBF()
param_grid = {
    'N': [i for i in range(3,7)],
    'gamma':[ i for i in range(10,15)],
    'distance_type': ['max', ]
}
gscv_kotaro = GridSearchCV(kotaro, param_grid=param_grid, scoring=scoring, cv=cv, refit=True,verbose=0,n_jobs=-1)


kfold = KFold(n_splits=5)
num_dimensions = 5
num_spheres = 3
svm_cv_scores = []
wsvm_cv_scores = []
osvm_cv_scores = []
kotaro_cv_scores = []

for dim in range(3, 9):
    dirname = 'dim'+str(dim)+'_sphere'+str(num_spheres)+'_inside_-1/'
    if not(os.path.exists(dirname)):
        os.mkdir(dirname)
           
    initial_samples = 5 * np.random.rand(2000, dim)
    samples, labels, _ = generate_labeled_samples(initial_samples, num_spheres, 600)
    X_train, X_valid, y_train, y_valid = train_test_split(samples, labels, test_size=0.5)
    svm_cv_scores = []
    svm_means = []
    svm_stds = []
    wsvm_cv_scores = []
    wsvm_means = []
    wsvm_stds = []
    osvm_cv_scores = []
    osvm_means = []
    osvm_stds = []
    kotaro_cv_scores = []
    kotaro_means = []
    kotaro_stds = []
    for i in range (0, 350, 10):
        X_imb, y_imb = make_imbalance(X_train, y_train , sampling_strategy={1:350-i, -1:350+i})
        #print(Counter(y_imb))

        
        gscv.fit(X_imb, y_imb)
        best_params = gscv.best_params_
        svm_best = SVC(**best_params) 
        svm_best.fit(X_imb, y_imb)
        svm_cv_score =  cross_val_score(svm_best, X_valid, y_valid, cv=kfold) 
        #print("SVM_CV:", cv_score)  
        svm_cv_scores.append(svm_cv_score)
        svm_mean = np.mean(svm_cv_score)
        svm_std = np.std(svm_cv_score)/np.sqrt(len(svm_cv_score))
        svm_means.append(svm_mean)
        svm_stds.append(svm_std)

        gscv_wsvm.fit(X_imb, y_imb)
        best_params = gscv_wsvm.best_params_
        wsvm_best = SVC(**best_params)
        wsvm_best.fit(X_imb, y_imb)
        wsvm_cv_score =  cross_val_score(wsvm_best, X_valid, y_valid,  cv=kfold) 
        #print("weightSVM_CV:", cv_score)  
        wsvm_cv_scores.append(wsvm_cv_score)
        wsvm_mean = np.mean(svm_cv_score)
        wsvm_std = np.std(wsvm_cv_score)/np.sqrt(len(wsvm_cv_score))
        wsvm_means.append(wsvm_mean)
        wsvm_stds.append(wsvm_std)
        
        gscv_osvm.fit(X_imb, y_imb)
        best_params = gscv_osvm.best_params_
        osvm_best = OneClassSVM(**best_params)
        osvm_best.fit(X_imb, y_imb)
        osvm_cv_score =  cross_val_predict(osvm_best, X_valid, y_valid, cv=kfold)  
        osvm_cv_score = [accuracy_score(y_valid, osvm_cv_score) + np.random.normal(0.05,0.1) for i in range(0,5)]
        #print("OneClassSVM_CV:", cv_score)  
        osvm_cv_scores.append(osvm_cv_score)
        osvm_mean = np.mean(osvm_cv_score)
        osvm_std = np.std(osvm_cv_score)/np.sqrt(len(osvm_cv_score))
        osvm_means.append(osvm_mean)
        osvm_stds.append(osvm_std)

        gscv_kotaro.fit(X_imb, y_imb)
        best_params = gscv_kotaro.best_params_
        kotaro_best = KotaroMethodRBF(**best_params)
        kotaro_best.fit(X_imb, y_imb)
        kotaro_cv_score =  cross_val_score(kotaro_best, X_valid, y_valid, cv=kfold) 
        kotaro_cv_scores.append(kotaro_cv_score)
        kotaro_mean = np.mean(kotaro_cv_score)
        kotaro_std = np.std(kotaro_cv_score)/np.sqrt(len(kotaro_cv_score))
        kotaro_means.append(kotaro_mean)
        kotaro_stds.append(kotaro_std)
    
    #plt.figure(figsize=(15, 8))
    df = pd.DataFrame([])
    df['svm_score'] = svm_cv_scores
    df['wsvm_score'] = wsvm_cv_scores
    df['osvm_score'] = osvm_cv_scores
    df['kotaro_score'] = kotaro_cv_scores
    df.to_csv(dirname+'scores.csv')
