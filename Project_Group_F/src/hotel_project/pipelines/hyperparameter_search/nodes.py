"""
This is a boilerplate pipeline 'hyperparameter_search'
generated using Kedro 0.18.1
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import warnings
import mlflow

from typing import Any, Dict
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from hyperopt import fmin, tpe, hp

def hyperparameter_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    parameters: Dict[str, Any]):
    """Perform hyperparameter seacrh for a machine learning model.

    Args: 
        X_train[pd.DataFrame]: training data features.
        y_train[pd.Series]: training data labels.
        parameters[Dict[str, Any]]: dictionary containing the hyperparameter search configuration.
        best_cols[pickle]: selected best columns with feature selection

    Returns: 
        [pickle]: best model found during the hyperparameter search.

    Raises:
        ValueError: If an invalid combination of solver and penalty is encountered in Logistic Regression.
    """

    search_n_iter = parameters["search_n_iter"]

    # Create a base classifier and define parameter distribution
    model = RandomForestClassifier()
    if parameters["model"] == "random_forest":
        # Define the parameter search space for Random Forest
        param_dist = {
            'n_estimators': hp.quniform('n_estimators', 50, 500, 1),
            'max_depth': hp.choice('max_depth', [5,10,15,20,40,60,80,None]),
            'max_features': hp.choice('max_features', ['sqrt',10,15,20,25,30,None]),
            'max_samples': hp.choice('max_samples', [0.5,0.7,0.9,None])
        }
        
        # Define the objective functions for Bayesian Optimization
        def objective_rf(params):
            n_estimators = round(params['n_estimators'])
            max_depth = params['max_depth']
            max_features = params['max_features']
            max_samples = params['max_samples']
            
            # Define the Classifier with the specified hyperparameters
            rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, max_samples=max_samples,
                                        random_state=parameters["random_state"])

            # Perform cross-validation and return the average recall score
            cv_results = cross_validate(rf, np.array(X_train), np.array(y_train).ravel(), cv=5, scoring='recall',
                                        return_train_score=True,error_score="raise")
    
            # Calculate the penalty term based on the difference between training and validation scores
            penalty = abs(cv_results['train_score'].mean() - cv_results['test_score'].mean())

            # Combine the cross-validation score and the penalty term to form the objective
            objective_value = cv_results['test_score'].mean() - 2 * penalty  
            return -objective_value
        
        objective = objective_rf
    
    # Rest of the code for other models and their objective functions
    if parameters["model"] == "adaboost":
        model = AdaBoostClassifier()
        param_dist = {
            'estimator': hp.choice('estimator', [DecisionTreeClassifier(max_depth=3), LogisticRegression(max_iter=5000)]),
            'n_estimators': hp.quniform('n_estimators',100,500,1),
            'learning_rate': hp.uniform('learning_rate',0.01,1),
            'algorithm': hp.choice('algorithm', ['SAMME', 'SAMME.R'])
        }

        def objective_ab(params):
            estimator = params["estimator"]
            n_estimators = round(params["n_estimators"])
            learning_rate = params["learning_rate"]
            algorithm = params["algorithm"]

            ab = AdaBoostClassifier(estimator=estimator,n_estimators=n_estimators,learning_rate=learning_rate,algorithm=algorithm,
                                    random_state=parameters["random_state"])

            cv_results = cross_validate(ab, np.array(X_train), np.array(y_train).ravel(), cv=5, scoring='recall',
                                        return_train_score=True)
            
            penalty = abs(cv_results['train_score'].mean() - cv_results['test_score'].mean())

            objective_value = cv_results['test_score'].mean() - 2 * penalty  
            return -objective_value
        
        objective = objective_ab 
        
    if parameters["model"] == "decision_tree":
        model = DecisionTreeClassifier()
        param_dist = {
            'criterion': hp.choice('criterion', ['gini', 'entropy']),
            'max_depth': hp.choice('max_depth', [5,10,15,20,40,60,80,None]),
            'max_features': hp.choice('max_features', ['sqrt',10,15,20,25,30,None]),
            'ccp_alpha': hp.uniform('C',0,1)
        }

        def objective_dt(params):
            criterion = params["criterion"]
            max_depth = params["max_depth"]
            max_features = params["max_features"]
            ccp_alpha = params["ccp_alpha"]
            
            dt = DecisionTreeClassifier(criterion=criterion,max_depth=max_depth,max_features=max_features,ccp_alpha=ccp_alpha,
                                        random_state=parameters["random_state"])

            cv_results = cross_validate(dt, np.array(X_train), np.array(y_train).ravel(), cv=5, scoring='recall',
                                        return_train_score=True)

            penalty = abs(cv_results['train_score'].mean() - cv_results['test_score'].mean())

            objective_value = cv_results['test_score'].mean() - 2 * penalty  
            return -objective_value
        
        objective = objective_dt

    if parameters["model"] == "catboost":
        model = CatBoostClassifier()
        param_dist = {
            'n_estimators': hp.quniform('n_estimators',100,500,1),
            'learning_rate': hp.uniform('learning_rate',0.01,1),
            'max_depth': hp.quniform('max_depth',3,9,1),
            'subsample': hp.uniform('subsample',0.1,1)
        }

        def objective_cb(params):
            n_estimators = round(params["n_estimators"])
            max_depth = round(params["max_depth"])
            learning_rate = params["learning_rate"]
            subsample = params["subsample"]

            cb = CatBoostClassifier(n_estimators=n_estimators,learning_rate=learning_rate,max_depth=max_depth,subsample=subsample,
                                    random_state=parameters["random_state"])

            cv_results = cross_validate(cb, np.array(X_train), np.array(y_train).ravel(), cv=5, scoring='recall',
                                        return_train_score=True)
            
            penalty = abs(cv_results['train_score'].mean() - cv_results['test_score'].mean())

            objective_value = cv_results['test_score'].mean() - 2 * penalty  
            return -objective_value
        
        objective = objective_cb

    if parameters["model"] == "lgbm":
        model = LGBMClassifier()
        param_dist = {
            'n_estimators': hp.quniform('n_estimators',100,500,1),
            'learning_rate': hp.uniform('learning_rate',0.01,1),
            'max_depth': hp.quniform('max_depth',3,9,1),
            'subsample': hp.uniform('subsample',0.1,1),
            'min_child_weight': hp.uniform('min_child_weight',1,5)
        }
        def objective_lgbm(params):
            n_estimators = round(params["n_estimators"])
            learning_rate = params["learning_rate"]
            max_depth = round(params["max_depth"])
            subsample = params["subsample"]
            min_child_weight = params["min_child_weight"]

            lgb = LGBMClassifier(n_estimators=n_estimators,learning_rate=learning_rate,max_depth=max_depth,subsample=subsample,min_child_weight=min_child_weight,
                                 random_state=parameters["random_state"])

            cv_results = cross_validate(lgb, np.array(X_train), np.array(y_train).ravel(), cv=5, scoring='recall',
                                        return_train_score=True)
            
            penalty = abs(cv_results['train_score'].mean() - cv_results['test_score'].mean())

            objective_value = cv_results['test_score'].mean() - 2 * penalty  
            return -objective_value
        
        objective = objective_lgbm
        
    if parameters["model"] == "xgboost":
        model = xgb.XGBClassifier()
        param_dist = {
            'alpha': hp.uniform('alpha',0,5),
            'subsample': hp.uniform('subsample',0.1,1),
            'learning_rate': hp.uniform('learning_rate',0.01,1),
            'max_depth': hp.quniform('max_depth',3,9,1),
            'min_child_weight': hp.uniform('min_child_weight',1,5),
        }

        def objective_xgb(params):
            alpha = params["alpha"]
            min_child_weight = params["min_child_weight"]
            learning_rate = params["learning_rate"]
            max_depth = round(params["max_depth"])
            subsample = params["subsample"]

            xgbc = xgb.XGBClassifier(alpha=alpha,learning_rate=learning_rate,max_depth=max_depth,subsample=subsample,min_child_weight=min_child_weight,
                                     random_state=parameters["random_state"])

            cv_results = cross_validate(xgbc, np.array(X_train), np.array(y_train).ravel(), cv=5, scoring='recall',
                                        return_train_score=True)
            
            penalty = abs(cv_results['train_score'].mean() - cv_results['test_score'].mean())

            objective_value = cv_results['test_score'].mean() - 2*penalty  
            return -objective_value
        
        objective = objective_xgb
    
    if parameters["model"] == "logistic_regression":
        model = LogisticRegression()
        param_dist = {
            'penalty': hp.choice('penalty',['elasticnet','l2', 'l1', None]),
            'C': hp.uniform('C',0.001,10),
            'solver': hp.choice('solver', ['lbfgs','sag'])
        }

        def objective_lr(params):
            penalty = params['penalty']
            C = params['C']
            solver = params['solver']

            try:
                lr = LogisticRegression(penalty=penalty, C=C, solver=solver,
                                        random_state=parameters["random_state"])
                
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    cv_results = cross_validate(lr, np.array(X_train), np.array(y_train).ravel(), cv=5, scoring='recall',
                                                return_train_score=True)
            except ValueError:
                # Invalid solver and penalty combination
                return 1000.0

            penalty = abs(cv_results['train_score'].mean() - cv_results['test_score'].mean())
            objective_value = cv_results['test_score'].mean() - 2 * penalty
            return -objective_value
        
        objective = objective_lr
    
    if parameters["model"] == "kneighbors":
        model = KNeighborsClassifier()
        param_dist = {
            'n_neighbors': hp.quniform('n_neighbors',5,30,1),
            'weights': hp.choice('weights',['uniform','distance']),
            'metric': hp.choice('metric',['manhattan', 'minkowski'])
        }
        def objective_knn(params):
            n_neighbors = round(params["n_neighbors"])
            weights = params["weights"]
            metric = params["metric"]
                    
            knn = KNeighborsClassifier(n_neighbors=n_neighbors,weights=weights,metric=metric)

            cv_results = cross_validate(knn, np.array(X_train), np.array(y_train).ravel(), cv=5, scoring='recall',
                                    return_train_score=True)
            
            penalty = abs(cv_results['train_score'].mean() - cv_results['test_score'].mean())

            objective_value = cv_results['test_score'].mean() - 2 * penalty  
            return -objective_value
        
        objective = objective_knn

    if parameters["model"] == "extra_trees":
        model = ExtraTreesClassifier()
        param_dist = {
            'n_estimators': hp.quniform('n_estimators',50,500,1),
            'max_depth': hp.choice('max_depth', [2, 6, 10, 15, 20, 60, 80, None]),
            'max_features': hp.choice('max_features', ['sqrt', 'log2', None]),
            'bootstrap': hp.choice('bootstrap', [True, False]),
            'min_samples_split': hp.choice('min_samples_split', [5, 10, 25, 50, 100, 200])
        }

        def objective_et(params):
            n_estimators = round(params["n_estimators"])
            max_depth = params["max_depth"]
            max_features = params["max_features"]
            bootstrap = params["bootstrap"]
            min_samples_split = params["min_samples_split"]
                    
            et = ExtraTreesClassifier(n_estimators=n_estimators,max_depth=max_depth,max_features=max_features,bootstrap=bootstrap,min_samples_split=min_samples_split,
                                     random_state=parameters["random_state"])

            cv_results = cross_validate(et, np.array(X_train), np.array(y_train).ravel(), cv=5, scoring='recall',
                                    return_train_score=True)
            
            penalty = abs(cv_results['train_score'].mean() - cv_results['test_score'].mean())

            objective_value = cv_results['test_score'].mean() - 2*penalty  
            return -objective_value
        
        objective = objective_et
    
    # Execute the hyperparameter search using hyperopt's fmin function
    best_params = fmin(fn = objective,  # function to optimize
                       space = param_dist,  # Search space for hyperparameters
                       algo = tpe.suggest,  # optimization algorithm, hyperopt will select its parameters automatically
                       max_evals = search_n_iter,  # maximum number of iterations
                       return_argmin = False
                       )
    
    if parameters["model"] == "random_forest" or parameters["model"]=="adaboost" or parameters["model"] == "catboost" or parameters["model"] == "lgbm" or parameters["model"] == "extra_trees":
        best_params['n_estimators'] = round(best_params['n_estimators'])
    if parameters["model"] == "catboost" or parameters["model"] == "lgbm" or parameters["model"] == "xgboost":
        best_params["max_depth"] = round(best_params["max_depth"])
    if parameters["model"] == "kneighbors":
        best_params["n_neighbors"]=round(best_params["n_neighbors"])
    if parameters["model"] != "kneighbors":
        best_params['random_state'] = parameters["random_state"]
        
    # Set the parameters of the model to the best parameters found above
    model.set_params(**best_params)
    
    # MLFlow does not autolog the parameters of CatBoost
    if parameters["model"]=="catboost":
        mlflow.log_params(best_params)

    return model