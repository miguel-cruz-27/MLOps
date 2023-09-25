"""
This is a boilerplate pipeline 'data_split'
generated using Kedro 0.18.1
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import split_train_test

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([

        node(
            func = split_train_test,
            inputs = ["hotel_all_data", "parameters"],
            outputs = ["train_data", "test_data"],
            name = "split_train_test",
        )

    ])
