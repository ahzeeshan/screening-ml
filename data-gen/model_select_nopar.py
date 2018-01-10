# filter warnings messages from the notebook
import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
import numpy as np
from sklearn import linear_model
import scipy.io as spio
from sklearn.multioutput import MultiOutputRegressor
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
#from joblib import Parallel, delayed
import pickle
import time
start_time = time.time()

def get_best_model_for_data(X=None, y=None, models_and_parameters=None, NUM_TRIALS=1, filename='model_opt'):
    scaler = preprocessing.StandardScaler()
    non_nested_scores = np.zeros(NUM_TRIALS)
    nested_scores = np.zeros(NUM_TRIALS)
    model_avg_scores = dict()
    model_ntrial_avg_scores = dict()
    for name, (model, params) in models_and_parameters.items():
        model_avg_scores[name] = np.zeros(NUM_TRIALS)
    for i in range(NUM_TRIALS):
        print('Trial '+str(i)+'\n')
        inner_cv = KFold(n_splits=3, shuffle=True)
        outer_cv = KFold(n_splits=3, shuffle=True)
        for name, (model, params) in models_and_parameters.items():
            pipeline = Pipeline([('transformer', scaler), ('reg', model)])
            # Choose cross-validation techniques for the inner and outer loops,                                                                       
            # independently of the dataset. 
            # Non_nested parameter search and scoring
            clf = GridSearchCV(estimator=pipeline, param_grid=params, cv=inner_cv, n_jobs=-1)
            clf.fit(X, y)
#            print(clf.best_params_)
            non_nested_scores[i] = clf.best_score_
            # Nested CV with parameter optimization                                                                                 
            nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv,n_jobs=-1)
            model_avg_scores[name][i] = np.mean(nested_score)
            error_summary = 'Model: {name}\nR2 in the 3 outer folds: {scores}.\nAverage error: {avg}'
            print(error_summary.format(name=name, scores=nested_score,avg=model_avg_scores[name][i]))
            print()
    for name, (model, params) in models_and_parameters.items():
        model_ntrial_avg_scores[name] = np.mean(model_avg_scores[name])
    best_model_name, best_model_avg_score = max(model_ntrial_avg_scores.items(),
                                                key=(lambda name_averagescore: name_averagescore[1]))
    best_model, best_model_params = models_and_parameters[best_model_name]
    best_pipeline = Pipeline([('transformer', scaler), ('reg', best_model)])
    final_model = GridSearchCV(best_pipeline, param_grid = best_model_params, cv=inner_cv, n_jobs=-1)
    final_model.fit(X, y)
    print('Best model: \n\t{}'.format(best_model), end='\n\n')
    print('Estimation of its generalization error (negative mean squared error):\n\t{}'.format(
            best_model_avg_score), end='\n\n')
    print('Best parameter choice for this model: \n\t{params}'
          '\n(according to cross-validation `{cv}` on the whole dataset).'.format(
            params=final_model.best_params_, cv=inner_cv))
    
    with open(filename+'.pkl','wb') as f:
        pickle.dump([model_avg_scores, model_ntrial_avg_scores, best_model, best_model_params, final_model], f)

if __name__ == "__main__":
    # Load the dataset
    with open('lattice-type.txt') as f:
        lattice = f.read()
    lattice = "".join(lattice.split())

    print('Fitting models for lattice = '+lattice)

    data = spio.loadmat(lattice+'-data-posd-with-den.mat')
    X = data['xdata']
    y = data['ydata']
    
    these_models_and_parameters = {
        'lasso': (linear_model.Lasso(),
                  {'reg__alpha': [0.01, 0.1, 0.5, 1.,5.,10.]}),
        'elnet': (linear_model.ElasticNet(),
                  {'reg__alpha':[0.01, 0.1, 0.5, 1, 5., 10.], 'reg__l1_ratio':[0., 0.1, 0.5, 1.]}),
        'krg': (KernelRidge(),
                {'reg__kernel':['rbf','linear'], 'reg__alpha': [1e0, 0.1, 1e-2, 1e-3], 'reg__gamma': np.logspace(-2, 2, 5)}),
        'gpr': (GaussianProcessRegressor(kernel = kernels.RBF(), normalize_y=True),
                {'reg__kernel__length_scale':[0.01, 0.1, 1., 2., 10., 100.], 
                 'reg__kernel__length_scale_bounds': [(1e-2,1.), (1e-1,10.), (1.,10.), (1.,100.), (1e-2,1e2)]}),
        'gbr': (GradientBoostingRegressor(learning_rate=0.1, loss='ls', n_estimators=500),
                {'reg__max_depth': [2, 3, 4, 10, 20, 50],
                 'reg__min_samples_leaf': [2, 4], 
                 'reg__max_features':['auto', 'sqrt'],
                 'reg__subsample':[0.4, 1.0],
                 'reg__min_samples_split': [2, 4]}),
        'ada': (AdaBoostRegressor(base_estimator=DecisionTreeRegressor(), n_estimators=100, learning_rate=1.0),#max_depth alone doesn't work probably: issue on github
                {#'reg__base_estimator__max_depth': [2, 4, 10], 
                    'reg__base_estimator':[DecisionTreeRegressor(max_depth = 4, max_features='auto'),
                                           DecisionTreeRegressor(max_depth = None, max_features='auto'),
                                           DecisionTreeRegressor(max_depth = 4, max_features='sqrt'),
                                           DecisionTreeRegressor(max_depth = None, max_features='sqrt')]}),
        'svr': (SVR(),
                {'reg__C': [0.01, 0.05, 0.1, 1], 
                 'reg__kernel': ['linear', 'rbf']}),
        'rf': (RandomForestRegressor(n_estimators=50),
               {'reg__max_depth': [None, 5, 10, 50, 100],
                'reg__max_features':['auto', 'sqrt'],
                'reg__min_samples_split':[2, 4],
                'reg__min_samples_leaf':[2, 4]}),
        'brg': (linear_model.BayesianRidge(fit_intercept=True),
                {'reg__alpha_1': [1.e-6, 1.e-5]}),
        'lars': (linear_model.Lars(fit_intercept = True, normalize=False),
                 {'reg__n_nonzero_coefs': [5, 10, 100, np.inf]}),
        'ard': (linear_model.ARDRegression(normalize=False),
                {'reg__alpha_1':[1.e-6, 1.e-5]})}


    ncoeffs = y.shape[1]
    #Parallel(n_jobs=ncoeffs)(delayed(get_best_model_for_data)(X,y[:,coeff], models_and_parameters=these_models_and_parameters, NUM_TRIALS=2, filename=lattice+'_coeff'+str(coeff)) for coeff in range(y.shape[1]))
    for coeff in range(ncoeffs):
        get_best_model_for_data(X,y[:,coeff], models_and_parameters=these_models_and_parameters, NUM_TRIALS=50, filename=lattice+'_nparcoeff'+str(coeff))
    print("Time elapsed:  %s seconds" % (time.time() - start_time))
