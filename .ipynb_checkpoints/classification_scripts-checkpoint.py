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

# +
# load libraries
import pandas as pd
from typing import TypeVar, NamedTuple, Dict, List
from sklearn.model_selection import train_test_split, KFold, RepeatedStratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

PandasDataFrame = TypeVar('pd.core.frame.DataFrame')
PandasSeries = TypeVar('pd.core.frame.Series')

class HoldOutTrainFrames(NamedTuple):
    X_train: PandasDataFrame
    X_hold_out: PandasDataFrame
    y_train: PandasSeries
    y_hold_out: PandasSeries

def hold_out_set(dataframe: PandasDataFrame, y_loc: int = -1, seed:int = 42, size: float = 0.2 ) -> HoldOutTrainFrames:
    """ 
    Splits the Data
    
    """
    array = dataframe.copy()
    X = array.iloc[:, 0:y_loc]
    y = array.iloc[:, y_loc]
    frames = train_test_split(X, y, test_size=size, random_state=seed)
    return HoldOutTrainFrames(*frames)

def eval_class_no_scaling(X_train: PandasDataFrame, y_train: PandasSeries, seed:int, 
                          stratified:bool=False, multi_class:bool=False,
                          scoring:str='accuracy', num_folds:int=10)->PandasDataFrame:
    """
    Evaluates 5 vanilla classification algorithms for the given X_train, y_train pair using the cross_validation approach
    Prints DataFrame describe statistics
    Returns Dataframe with scoring results for each cv trial.
    NO scaling is done to the features
    """
    
    strat = KFold if not stratified else RepeatedStratifiedKFold
    multi = 'auto' if not multi_class else 'ovr' 
    
    models = []
    models.append(('LR', LogisticRegression(solver='lbfgs', multi_class=multi)))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    
    # evaluate each model in turn
    results = {}
    for name, model in models:
        kfold = strat(n_splits=num_folds, random_state=seed)
        cv_results = cross_val_score(estimator=model, X= X_train,y= y_train,cv = kfold, scoring=scoring)
        results[name] = cv_results
    
    results = pd.DataFrame(results)
    print(results.describe())
    results.plot(kind='box', figsize=(10,5))
    return results

def eval_class_scaling(X_train: PandasDataFrame, y_train: PandasSeries, seed:int, 
                          stratified:bool=False,multi_class:bool=False,
                          scoring:str='accuracy', num_folds:int=10, scaler:str='standard')-> PandasDataFrame:
    """
    Evaluates 5 vanilla classification algorithms for the given X_train, y_train pair using the cross_validation approach
    Prints DataFrame describe statistics
    Returns Dataframe with scoring results for each cv trial.
    Features are scaled using standard scaler
    """
    
    strat = KFold if not stratified else RepeatedStratifiedKFold
    multi = 'auto' if not multi_class else 'ovr' 
    # TODO other scaler options, baked in StandardScaler for now
    
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
    
    # evaluate each model in turn
    results = {}
    for name, model in pipelines:
        kfold = strat(n_splits=num_folds, random_state=seed)
        cv_results = cross_val_score(estimator=model, X= X_train,y= y_train,cv = kfold, scoring=scoring)
        results[name] = cv_results
    
    results = pd.DataFrame(results)
    print(results.describe())
    results.plot(kind='box', figsize=(10,5))
    return results

def eval_class_no_scaling_ensembles(X_train: PandasDataFrame, y_train: PandasSeries, seed:int, 
                          stratified:bool=False, multi_class:bool=False,n_estimators:int=100,
                          scoring:str='accuracy', num_folds:int=10)->PandasDataFrame:
    """
    Evaluates 5 ensembles classification algorithms for the given X_train, y_train pair using the cross_validation approach
    Prints DataFrame describe statistics
    Returns Dataframe with scoring results for each cv trial.
    No scaling to features
    
    """
    
    strat = KFold if not stratified else RepeatedStratifiedKFold
    multi = 'auto' if not multi_class else 'ovr' 
    
    ensembles = []
    ensembles.append(('AB', AdaBoostClassifier()))
    ensembles.append(('GBM', GradientBoostingClassifier()))
    ensembles.append(('RF', RandomForestClassifier(n_estimators=n_estimators)))
    ensembles.append(('ET', ExtraTreesClassifier(n_estimators=n_estimators)))
    
    # evaluate each model in turn
    results = {}
    for name, model in ensembles:
        kfold = KFold(n_splits=num_folds, random_state=seed)
        cv_results = cross_val_score(estimator=model, X= X_train,y= y_train,cv = kfold, scoring=scoring)
        results[name] = cv_results
    
    results = pd.DataFrame(results)
    print(results.describe())
    results.plot(kind='box', figsize=(10,5))
    return results

#test
