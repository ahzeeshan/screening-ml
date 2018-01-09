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
import shelve


# Number of random trials

NUM_TRIALS = 100


# Load the dataset
lattice = 'cubic'
data = spio.loadmat(lattice+'-data-posd-with-den.mat')
X = data['xdata']
y = data['ydata']

models_and_parameters = {
    'lasso': (linear_model.Lasso(),
              {'reg__alpha': [0.01, 0.1, 0.5, 1.,5.,10.]}),
    'elnet': (linear_model.ElasticNet(),
              {'reg__alpha':[0.01, 0.1, 0.5, 1, 5., 10.], 'reg__l1_ratio':[0.,0.1,0.5,1.,2.1]}),
    'krg': (KernelRidge(),
            {'reg__kernel':['rbf','linear'], 'reg__alpha': [1e0, 0.1, 1e-2, 1e-3], 'reg__gamma': np.logspace(-2, 2, 5)}),
    'gpr': (GaussianProcessRegressor(kernel = kernels.RBF()),
            {'reg__kernel__length_scale':[0.01, 0.1, 1., 2., 10., 100.], 'reg__kernel__length_scale_bounds':[(1e-2,1.),(1e-1,1.),(1e-1,10.),(1.,10.),(1.,100.),(1e-2,1e2)]}),
    'gbr': (GradientBoostingRegressor(learning_rate=0.01, loss='ls', n_estimators=1000),
            {'reg__max_depth': [2, 3, 4, 10, 20, 50],'reg__min_samples_leaf': [2,3,4,10], 'reg__max_features':['auto', 'sqrt', 'log2'],'reg__subsample':[0.3, 0.4, 0.5],'reg__min_samples_split': [2, 3, 4]}),
    'ada': (AdaBoostRegressor(base_estimator=DecisionTreeRegressor(),n_estimators=500,learning_rate=0.01),#max_depth alone doesn't work probably
            {'reg__base_estimator__max_depth': [2,3,4,10], 'reg__base_estimator':[DecisionTreeRegressor(max_depth = 4, max_features='auto'), 
                                                                                     DecisionTreeRegressor(max_depth = None, max_features='auto'),
                                                                                     DecisionTreeRegressor(max_depth = 4, max_features='sqrt'),
                                                                                     DecisionTreeRegressor(max_depth = None, max_features='sqrt')]}),
    'svr': (SVR(),
            {'reg__C': [0.01, 0.05, 0.1, 1], 'reg__kernel': ['linear', 'rbf']}),
    'rf': (RandomForestRegressor(n_estimators=1000),
           {'reg__max_depth': [None, 5, 10, 50,100],'reg__max_features':['auto','sqrt','log2'],'reg__min_samples_split':[2,3,4],'reg__min_samples_leaf':[2,3,4]}),
    'brg': (linear_model.BayesianRidge(fit_intercept=True),
            {'reg__alpha_1': [1.e-6, 1.e-5]}),
    'lars': (linear_model.Lars(fit_intercept = True, normalize=False),
             {'reg__n_nonzero_coefs': [5, 10, 50, 500, np.inf]}),
    'ard': (linear_model.ARDRegression(),
            {'reg__alpha_1':[1.e-6, 1.e-5]})}

# Set up possible values of parameters to optimize over
scaler = preprocessing.StandardScaler()
#pipeline = Pipeline([('transformer', scalar), ('reg', MultiOutputRegressor(rg))])

# Arrays to store scores
non_nested_scores = np.zeros(NUM_TRIALS)
nested_scores = np.zeros(NUM_TRIALS)

model_avg_scores = [dict() for x in range(y.shape[1])]
model_ntrial_avg_scores = [dict() for x in range(y.shape[1])]
for coeff in range(y.shape[1]):
    for name, (model, params) in models_and_parameters.items():
        model_avg_scores[coeff][name] = np.zeros(NUM_TRIALS)

# Loop for each trial
for coeff in range(y.shape[1]):
    print('-----coeff = '+str(coeff)+'------\n')
    for i in range(NUM_TRIALS):
        print('Trial '+str(i)+'\n')
        inner_cv = KFold(n_splits=3, shuffle=True)
        outer_cv = KFold(n_splits=3, shuffle=True)
        for name, (model, params) in models_and_parameters.items():
            pipeline = Pipeline([('transformer', scaler), ('reg', model)])
            non_nested_scores = np.zeros(NUM_TRIALS)
            nested_scores = np.zeros(NUM_TRIALS)
            # Choose cross-validation techniques for the inner and outer loops,
            # independently of the dataset.
            # Non_nested parameter search and scoring
            #print(pipeline.get_params())
            clf = GridSearchCV(estimator=pipeline, param_grid=params, cv=inner_cv)

            clf.fit(X, y[:,coeff])
            print(clf.best_params_)
            non_nested_scores[i] = clf.best_score_
            # Nested CV with parameter optimization
            nested_score = cross_val_score(clf, X=X, y=y[:,coeff], cv=outer_cv)
            #nested_scores[name][i] = nested_score.mean()
            model_avg_scores[coeff][name][i] = np.mean(nested_score)
            error_summary = 'Model: {name}\nR2 in the 3 outer folds: {scores}.\nAverage error: {avg}'
            print(error_summary.format(name=name, scores=nested_score,avg=model_avg_scores[coeff][name][i]))
            print()
    #Generate average over NUM_TRIALS
    for name, (model, params) in models_and_parameters.items():
        model_ntrial_avg_scores[coeff][name] = np.mean(model_avg_scores[coeff][name])
    best_model_name, best_model_avg_score = max(model_ntrial_avg_scores[coeff].items(),
                                                key=(lambda name_averagescore: name_averagescore[1]))
    best_model, best_model_params = models_and_parameters[best_model_name]
    best_pipeline = Pipeline([('transformer', scaler), ('reg', best_model)])
    final_model = GridSearchCV(best_pipeline, param_grid = best_model_params, cv=inner_cv)
    final_model.fit(X, y[:,coeff])
    print('Best model: \n\t{}'.format(best_model), end='\n\n')
    print('Estimation of its generalization error (negative mean squared error):\n\t{}'.format(
            best_model_avg_score), end='\n\n')
    print('Best parameter choice for this model: \n\t{params}'
          '\n(according to cross-validation `{cv}` on the whole dataset).'.format(
            params=final_model.best_params_, cv=inner_cv))

filename='shelve.out'
my_shelf = shelve.open(filename,'n') # 'n' for new
for key in dir():
    try:
        my_shelf[key] = globals()[key]
    except TypeError:
        #
        # __builtins__, my_shelf, and imported modules can not be shelved.
        #
        print('ERROR shelving: {0}'.format(key))
my_shelf.close()
