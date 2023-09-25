"""
This is a boilerplate pipeline 'target_split'
generated using Kedro 0.18.1
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import split_target

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([

        node(
            func = split_target,
            inputs = ["train_with_best_columns", "test_with_best_columns", "parameters"],
            outputs = ["X_train_data", "y_train_data", "X_test_data", "y_test_data"],
            name = "split_target",
        )

    ])