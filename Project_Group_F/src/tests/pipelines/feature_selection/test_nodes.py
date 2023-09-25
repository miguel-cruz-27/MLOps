"""
This is a boilerplate test file for pipeline 'feature_selection'
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

# path to data
train_scaled_path = "data/02_intermediate/train_scaled.csv"
train_encoded_path = "data/02_intermediate/train_encoded.csv"
test_scaled_path = "data/02_intermediate/test_scaled.csv"
test_encoded_path = "data/02_intermediate/test_encoded.csv"

parameters_path = r'conf\base\parameters.yml'

def read_file_as_dict(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

sys.path.append("src/hotel_project/pipelines")
from feature_selection.nodes import feature_selection


def test_feature_selection_data_type():
    df = pd.read_csv(train_scaled_path)
    df_encoded = pd.read_csv(train_encoded_path)
    df_test = pd.read_csv(test_scaled_path)
    df_test_encoded = pd.read_csv(test_encoded_path)
    parameters = read_file_as_dict(parameters_path)

    important_columns, train_important, test_important = feature_selection(df, df_encoded, df_test, df_test_encoded, parameters)

    isinstance(important_columns, list)
    isinstance(train_important, pd.DataFrame)
    isinstance(test_important, pd.DataFrame)


def test_feature_selection_columns():
    df = pd.read_csv(train_scaled_path)
    df_encoded = pd.read_csv(train_encoded_path)
    df_test = pd.read_csv(test_scaled_path)
    df_test_encoded = pd.read_csv(test_encoded_path)
    parameters = read_file_as_dict(parameters_path)

    important_columns, train_important, test_important = feature_selection(df, df_encoded, df_test, df_test_encoded, parameters)

    assert train_important.shape[1] == len(important_columns)+1
    assert test_important.shape[1] == len(important_columns)+1