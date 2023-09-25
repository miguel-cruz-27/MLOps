"""
This is a boilerplate pipeline 'feature_selection'
generated using Kedro 0.18.1
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import feature_selection


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([

        node(
            func = feature_selection,
            inputs = ["train_scaled", "train_encoded", "test_scaled", "test_encoded", "parameters"],
            outputs = ["best_columns", "train_with_best_columns", "test_with_best_columns"],
            name = "feature_selection",
        )

    ])
