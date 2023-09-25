"""
This is a boilerplate test file for pipeline 'data_merging'
generated using Kedro 0.18.1.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""

import pytest
import pandas as pd
import numpy as np

from kedro.framework.project import settings
from kedro.config import ConfigLoader
from kedro.framework.context import KedroContext
from kedro.framework.hooks import _create_hook_manager
import sys
import os

# path to hotel data and weather data
data_path = "data/01_raw/hotel_bookings.csv"
weather_path = "data/01_raw/Algarve 2015-07-01 to 2017-08-31.csv"


sys.path.append("src/hotel_project/pipelines")
from data_merging.nodes import separate_hotels_combine_holidays_weather

def test_separate_hotels_combine_holidays_weather_data_type():
    df = pd.read_csv(data_path, parse_dates = ["reservation_status_date"])
    weather = pd.read_csv(weather_path, parse_dates = ["datetime"])

    all_df = separate_hotels_combine_holidays_weather(df, weather)
    isinstance(all_df, pd.DataFrame)


def test_separate_hotels_combine_holidays_weather_all_columns():
    df = pd.read_csv(data_path, parse_dates = ["reservation_status_date"])
    weather = pd.read_csv(weather_path,  parse_dates = ["datetime"])

    all_df = separate_hotels_combine_holidays_weather(df, weather)
    # check if the DataFrame has 39 columns
    assert all_df.shape[1] == 39