"""
This is a boilerplate pipeline 'data_drift'
generated using Kedro 0.18.1
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import data_drift
from hotel_project.pipelines.model_predict.nodes import model_predict
from hotel_project.pipelines.target_split.nodes import split_target


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([

        node(
            func = split_target,
            inputs = ["train_with_best_columns", "test_with_best_columns", "parameters"],
            outputs = ["X_train_data", "y_train_data", "X_test_data", "y_test_data"],
            name = "split_target",
        ),

        node(
            func= model_predict,
            inputs=["test_model","X_test_data","y_test_data","test_encoded","parameters","best_columns"],
            outputs= "prediction",
            name="predict",
        ),

        node(
            func = data_drift,
            inputs = ["X_train_data", "X_test_data", "prediction", "y_test_data"],
            outputs = "drift_result",
            name = "drift_analysis",
        )

    ])
