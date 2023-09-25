"""
This is a boilerplate pipeline 'data_merging'
generated using Kedro 0.18.1
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import separate_hotels_combine_holidays_weather

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([

        node(
                func = separate_hotels_combine_holidays_weather,
                inputs = ["hotel_raw_data", "weather_data"],
                outputs = "hotel_all_data",
                name = "separate_and_combine",
            ),

    ])
