
"""
This is a boilerplate test file for pipeline 'data_split'
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
import yaml

# path to hotel data and weather data
data_path = "data/02_intermediate/hotel_all_data.csv"

file_path = r'conf\base\parameters.yml'

def read_file_as_dict(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

parameters = read_file_as_dict(file_path)

sys.path.append("src/hotel_project/pipelines")
from data_split.nodes import split_train_test

def test_split_train_test_data_type():
    df = pd.read_csv(data_path)

    df_train, df_test = split_train_test(df, parameters)

    isinstance(df_train, pd.DataFrame)
    isinstance(df_test, pd.DataFrame)


def test_split_train_test_columns():
    df = pd.read_csv(data_path)

    df_train, df_test = split_train_test(df, parameters)

    assert df_train.shape[1] == 39
    assert df_test.shape[1] == 39