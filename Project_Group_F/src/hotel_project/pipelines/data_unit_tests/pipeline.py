"""
This is a boilerplate pipeline 'data_unit_tests'
generated using Kedro 0.18.1
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import unit_test

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([

        node(
            func = unit_test,
            inputs = ["hotel_raw_data", "weather_data"],
            outputs = "",
            name = "unit_data_test",
        ),

    ])
