# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ## Wine quality

# The two datasets are related to red and white variants of the Portuguese "Vinho Verde" wine. 
#
# Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).
#
# These datasets can be viewed as classification or regression tasks. The classes are ordered and not balanced (e.g. there are many more normal wines than excellent or poor ones). 
# Outlier detection algorithms could be used to detect the few excellent or poor wines. Also, we are not sure if all input variables are relevant. So it could be interesting to test feature selection methods.

# ### Load the dataset

# +
# Load the libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot
from pathlib import Path
from collections import Counter

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# -

# Initial Ideas
#
# - Create one master dataset with dummy variable red or white
# - use white dataset for multi-classification problem as first pass with imbalanced classes
#
#

filepath_red = Path("../data/winequality-red.csv")
filepath_white = Path("../data/winequality-white.csv")
assert all([filepath_red.is_file(), filepath_white.is_file()])
dataset_red = pd.read_csv(filepath_red, sep=";").assign(red = 1)
dataset_white = pd.read_csv(filepath_white, sep=";").assign(white =1)

# #### Data peek

dataset_white.sample(5)

dataset_red.sample(5)

# ### Data Summary
#
# - one integer feature
# - no null values
# - same features in both datasets
# - features might require scaling
# - several outliers per input features , exception alcohol and density as expected
# - output class (quality) is imbalanced, CART or ensemble algorithms might perform better; also consider SMOTE method o address this. 
# - output is positively correlated with alcohol
# - density and residual sugar positively correlated
# - alcohol and density negatively correlated 
#

# ##### Data Shape

dataset_red.shape

dataset_white.shape

# #### Data types
#

dataset_red.info()

dataset_white.info()

# ##### Descriptive statistics

dataset_white.describe().T
# feature scaling should possibly considered; mean value is very different overall for features

dataset_red.describe().T

# output class balance
# output class imbalance
dataset_white['quality'].value_counts()

dataset_red['quality'].value_counts()

# ### Multiclass review with wine dataset only

# Data Visualizations

figsize = (14,8)
dataset_white.plot(kind='box', subplots=True, 
                   layout = (4,4), sharey=False, sharex=False, figsize=figsize);

dataset_white.hist(figsize=figsize);

# correlation
corr = dataset_white.iloc[:,:12].corr()
corr.style.background_gradient(cmap='coolwarm', axis=None).set_precision(2) 

# correlation per feature pairs
# top 10. 
melt_corr = dataset_white.corr().assign(cols = dataset_red.columns).melt(id_vars='cols', value_name = 'corr')
melt_corr.loc[melt_corr['corr'] != 1].assign(corr_abs = np.abs(melt_corr['corr'])).drop_duplicates('corr').nlargest(10, 'corr_abs').drop('corr_abs', axis =1)

# ### Join both datasets

# concatenate both dataset and create dummy variable
dataset  = pd.concat([dataset_red, dataset_white], sort=False).fillna(0)

# imbalanced class for predictor variable
dataset['quality'].value_counts()

# #### Data Transformation
#
# Creation of 2 prediction class high and low. 
#

cond_low = dataset['quality'].le(5)
cond_high = dataset['quality'].ge(6)

dataset.loc[cond_high, 'quality_trans'] = 'high'
dataset.loc[cond_low, 'quality_trans'] = 'low'

dataset.quality_trans.value_counts()

dataset = dataset.drop('quality', axis =1)
{col: i for i, col in enumerate(dataset.columns)}

# ## Evaluate Some Algorithms
# Now it is time to create some models of the data and estimate their accuracy on unseen data.
# Here is what we are going to cover in this step:
# 1. Separate out a validation dataset.
# 2. Setup the test harness to use 10-fold cross-validation.
# 3. Build 5 different models to predict species from flower measurements
# 4. Select the best model.

array = dataset.copy()
X = array.iloc[:, 0:13]
y= array.iloc[:,13]
hold_out_size = 0.2
seed = 49
X_train, X_hold_out, y_train, y_hold_out = train_test_split(X, y, test_size=hold_out_size, random_state=seed)

# +
# Spot-check algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))

# evaluate each model in turn
num_folds = 5
scoring = 'roc_auc' 
results = []
names = []

for name, model in models:
    kfold = RepeatedStratifiedKFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(estimator=model, X= X_train,y= y_train,cv = kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = f"{name} {cv_results.mean()} +/- ({cv_results.std()})"
    print(msg)
# -

# compare performance
fig1 = pyplot.figure()
fig1.suptitle('Algorithm comparison')
ax = fig1.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

# Problems with multicolinearity between variables for LDA. Let's scale the data and benchmark again with 5 algorithms below. 

# +
# Standardize the dataset
pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',
LogisticRegression(solver='lbfgs', multi_class='ovr'))])))
pipelines.append(('ScaledLDA', Pipeline([('Scaler', StandardScaler()),('LDA',
LinearDiscriminantAnalysis())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN',
KNeighborsClassifier())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART',
DecisionTreeClassifier())])))
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB',
GaussianNB())])))

results = []
names = []

for name, model in pipelines:
    kfold = RepeatedStratifiedKFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# -

# compare performance
# some impromevment on distance learning algorithms as KNN
fig1 = pyplot.figure()
fig1.suptitle('Algorithm comparison')
ax = fig1.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

# #### Ensemble Methods
#
# Four Ensemble methods are to be used here AdaBoost, Gradient Boosting from boosting methods and Random Forest and Extra Trees for bagging methods. 
# We will use the same test harness as before, 10-fold cross validation. No data standardization
# is used in this case because all four ensemble algorithms are based on decision trees that are
# less sensitive to data distributions.

# +
# ensembles
ensembles = []
ensembles.append(('AB', AdaBoostClassifier()))
ensembles.append(('GBM', GradientBoostingClassifier()))
ensembles.append(('RF', RandomForestClassifier(n_estimators=100)))
ensembles.append(('ET', ExtraTreesClassifier(n_estimators=100)))
results = []
names = []

for name, model in ensembles:
    kfold = RepeatedStratifiedKFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(estimator=model, X=X_train, y=y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = f" {name} :: {cv_results.mean()} ({cv_results.std()})"
    print(msg)
# -

# compare figures
fig = pyplot.figure()
ax = fig.add_subplot(111)
fig.suptitle('Ensemble Algorithm Comparison')
ax.set_xticklabels(names)
pyplot.boxplot(results);

# ### Algorithm Tunning
#
# Based on above results I've decided to tune the Extra Trees Classifier. n_estimators was the parameter used for tunning. No data transform is used. 

param_grid = {'n_estimators' : np.array([i for i in range(50,251,100)])}
model = ExtraTreesClassifier(random_state=seed)
kfold = RepeatedStratifiedKFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(X_train, y_train)
print(f"Best::{grid_result.best_score_, grid_result.best_params_}")

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, std, param in zip(means, stds, params):
    print(f"{mean}, ({std}), {param}")

# #### Finalize the model

# Now using an Extra Trees Classifier. 

# +
model = ExtraTreesClassifier(n_estimators=250)
model.fit(X_train, y_train)

predictions = model.predict(X_hold_out)

print(accuracy_score(y_hold_out, predictions))
print(confusion_matrix(y_hold_out, predictions))
print(classification_report(y_hold_out, predictions))
# -

# The model was able to catch 81% of all positive classifications (F1 score)
