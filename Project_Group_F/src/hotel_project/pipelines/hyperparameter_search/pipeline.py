"""
This is a boilerplate pipeline 'hyperparameter_search'
generated using Kedro 0.18.1
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import hyperparameter_search

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([

        node(
                func = hyperparameter_search,
                inputs = ["X_train_data", "y_train_data", "parameters"],
                outputs = "tuned_model",
                name = "hyperparameter_search"
            ),

    ])
