
"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import  model_predict

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func= model_predict,
                inputs=["test_model","X_test_data","y_test_data","test_encoded","parameters","best_columns"],
                outputs= "prediction",
                name="predict",
            ),
        ]
    )
