"""
This is a boilerplate pipeline 'model_train'
generated using Kedro 0.18.1
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

import shap 
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_validate
import mlflow

def model_train(model, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame, parameters: Dict[str, Any], best_cols):
    """
    Trains a machine learning model using the specified input data and parameters.

    Arguments:
        - model: The machine learning model object to be trained.
        - X_train: The input features DataFrame for the training set.
        - X_test: The input features DataFrame for the test set.
        - y_train: The target variable DataFrame for the training set.
        - y_test: The target variable DataFrame for the test set.
        - parameters: A dictionary containing additional parameters for the training process.
        - best_cols: A list of column names representing the subset of features to be used for training.
        
        Outputs:
            - model: The trained machine learning model.
            - plt: The matplotlib pyplot object for the summary plot of SHAP values.
    """
    
    X_train_temp = X_train[best_cols].copy()
    X_test_temp = X_test[best_cols].copy()
    best_cols = list(X_train.columns)
    
    mlflow.set_tag("mlflow.runName", parameters["run_name"])
    mlflow.autolog(log_model_signatures=True, log_input_examples=True)

    model.fit(X_train_temp, y_train)
    
    # Create object that can calculate shap values
    explainer = shap.Explainer(model.predict,X_test)

    # Calculate shap values. This is what we will plot.
    warnings.filterwarnings("ignore")  # Ignore all warnings
    shap_values = explainer(X_test)
    
    # Create SHAP summary plot
    plt.figure()  # Create a new figure
    shap.summary_plot(shap_values, show=False)
    shap_summary_plot = plt.gcf()

    # Create SHAP bar plot
    # Calculate the feature importance values
    feature_importance_vals = np.abs(shap_values.values).mean(0)
    shap_importance = pd.DataFrame(list(zip(X_train.columns, feature_importance_vals)), columns=['col_name', 'feature_importance_vals'])

    # Select the top 10 features
    top_10_features = shap_importance.head(10).sort_values(by='feature_importance_vals', ascending=True)

    # Plot the feature importance
    plt.figure()
    plt.barh(top_10_features['col_name'], top_10_features['feature_importance_vals'])
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.title('Top 10 SHAP Feature Importance')
    sns.despine()
    plt.tight_layout()
    shap_bar_plot = plt.gcf()
    
    preds = model.predict(X_test_temp)
    pred_labels = np.rint(preds)
    recall = recall_score(y_test, pred_labels)
        
    log = logging.getLogger(__name__)
    log.info(f"#Best columns: {len(best_cols)}")
    log.info("Model Recall Score on test set: %0.2f%%", recall * 100)
    
    # Perform cross-validation and log the scores
    cv_results = cross_validate(model, X_train_temp, y_train, cv=5, scoring=["accuracy", "f1", "recall"], return_train_score=True)

    train_accuracy = cv_results["train_accuracy"]
    test_accuracy = cv_results["test_accuracy"]
    train_f1 = cv_results["train_f1"]
    test_f1 = cv_results["test_f1"]
    train_recall = cv_results["train_recall"]
    test_recall = cv_results["test_recall"]

    mlflow.log_metrics({
        "train_accuracy_mean": np.mean(train_accuracy),
        "train_accuracy_std": np.std(train_accuracy),
        "test_accuracy_mean": np.mean(test_accuracy),
        "test_accuracy_std": np.std(test_accuracy),
        "train_f1_mean": np.mean(train_f1),
        "train_f1_std": np.std(train_f1),
        "test_f1_mean": np.mean(test_f1),
        "test_f1_std": np.std(test_f1),
        "train_recall_mean": np.mean(train_recall),
        "train_recall_std": np.std(train_recall),
        "test_recall_mean": np.mean(test_recall),
        "test_recall_std": np.std(test_recall)
     })
    
    if parameters["model"]=="catboost":
        mlflow.catboost.log_model(model, artifact_path="test_model")
        
    return model,shap_summary_plot,shap_bar_plot