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

# Number of random trials
NUM_TRIALS = 30

# Load the dataset
lattice = 'cubic'
data = spio.loadmat(lattice+'-data-posd.mat')
X = data['xdata']
y = data['ydata']

scaler = preprocessing.StandardScaler().fit(X)

models_and_parameters = {
    'lasso': (linear_model.Lasso(),
              {'reg__estimator__alpha': [0.01, 0.1, 0.5, 1.]}),
    'gbr': (GradientBoostingRegressor(learning_rate=0.01, min_samples_split=2, max_features='sqrt', loss='ls', subsample=0.4),
            {'reg__estimator__max_depth': [2,3,4],'reg__estimator__min_samples_leaf': [2,3,4], 'reg__estimator__learning_rate':[0.01, 0.1], 'reg__estimator__max_features':['auto', 'sqrt', 'log2']}),
    'elnet': (linear_model.ElasticNet(),
              {'reg__estimator__alpha':[0.01, 0.1, 0.5, 1], 'reg__estimator__l1_ratio':[0.,0.1,0.5,1.,2.]}),
    'ada': (AdaBoostRegressor(DecisionTreeRegressor(),n_estimators=500),
            {'reg__estimator__base_estimator__max_depth': [2,3,4], 'reg__estimator__learning_rate':[0.01, 0.1]}),
    'svr': (SVR(),
            {'reg__estimator__C': [0.01, 0.05, 0.1, 1], 'reg__estimator__kernel': ['linear', 'rbf']}),
    'rf': (RandomForestRegressor(),
           {'reg__estimator__max_depth': [5, 10, 50]}),
    'brg': (linear_model.BayesianRidge(fit_intercept=True),
            {'reg__estimator__alpha_1': [1.e-6, 1.e-5]}),
    'lars': (linear_model.Lars(fit_intercept = False, normalize=False),
             {'reg__estimator__n_nonzero_coefs': [10, 50, 500, np.inf]}),
    'ard': (linear_model.ARDRegression(),
            {'reg__estimator__alpha_1':[1.e-6, 1.e-5]})}

# Set up possible values of parameters to optimize over
scaler = preprocessing.StandardScaler()
#pipeline = Pipeline([('transformer', scalar), ('reg', MultiOutputRegressor(rg))])

# Arrays to store scores
non_nested_scores = np.zeros(NUM_TRIALS)
nested_scores = np.zeros(NUM_TRIALS)

# Loop for each trial
for i in range(NUM_TRIALS):
    print('Trial '+str(i)+'\n')
    inner_cv = KFold(n_splits=4, shuffle=True)
    outer_cv = KFold(n_splits=4, shuffle=True)
    average_scores_across_outer_folds_for_each_model = dict()
    for name, (model, params) in models_and_parameters.items():
        pipeline = Pipeline([('transformer', scaler), ('reg', MultiOutputRegressor(model))])
        non_nested_scores = np.zeros(NUM_TRIALS)
        nested_scores = np.zeros(NUM_TRIALS)
      # Choose cross-validation techniques for the inner and outer loops,
    # independently of the dataset.
    # E.g "LabelKFold", "LeaveOneOut", "LeaveOneLabelOut", etc.
      # Non_nested parameter search and scoring
        clf = GridSearchCV(estimator=pipeline, param_grid=params, cv=inner_cv)
        clf.fit(X, y)
        non_nested_scores[i] = clf.best_score_
    # Nested CV with parameter optimization
        nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv)
        nested_scores[i] = nested_score.mean()
        average_scores_across_outer_folds_for_each_model[name] = nested_scores[i]
        error_summary = 'Model: {name}\nR2 in the 3 outer folds: {scores}.\nAverage error: {avg}'
        print(error_summary.format(name=name, scores=nested_score,avg=nested_scores[i]))
        print()

'''
score_difference = non_nested_scores - nested_scores

print("Average difference of {0:6f} with std. dev. of {1:6f}."
      .format(score_difference.mean(), score_difference.std()))

# Plot scores on each trial for nested and non-nested CV
plt.figure()
plt.subplot(211)
non_nested_scores_line, = plt.plot(non_nested_scores, color='r')
nested_line, = plt.plot(nested_scores, color='b')
plt.ylabel("score", fontsize="14")
plt.legend([non_nested_scores_line, nested_line],
           ["Non-Nested CV", "Nested CV"],
           bbox_to_anchor=(0, .4, .5, 0))
plt.title("Non-Nested and Nested Cross Validation",
          x=.5, y=1.1, fontsize="15")

# Plot bar chart of the difference.
plt.subplot(212)
difference_plot = plt.bar(range(NUM_TRIALS), score_difference)
plt.xlabel("Individual Trial #")
plt.legend([difference_plot],
           ["Non-Nested CV - Nested CV Score"],
           bbox_to_anchor=(0, 1, .8, 0))
plt.ylabel("score difference", fontsize="14")

plt.show()

'''
