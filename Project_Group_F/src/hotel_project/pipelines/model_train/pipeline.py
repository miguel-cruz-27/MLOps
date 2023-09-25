"""
This is a boilerplate pipeline 'model_train'
generated using Kedro 0.18.1
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import model_train

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([

            node(
                func = model_train,
                inputs = ["tuned_model","X_train_data", "X_test_data", "y_train_data", "y_test_data", "parameters", "best_columns"],
                outputs = ["test_model", "output_plot","output_plot_2"],
                name = "train",
            ),

    ])
