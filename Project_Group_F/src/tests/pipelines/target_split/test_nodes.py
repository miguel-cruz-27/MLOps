"""
This is a boilerplate test file for pipeline 'target_split'
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
train_with_best_columns_path = "data/04_feature/train_with_best_columns.csv"
test_with_best_columns_path = "data/04_feature/test_with_best_columns.csv"

parameters_path = r'conf\base\parameters.yml'

def read_file_as_dict(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

sys.path.append("src/hotel_project/pipelines")
from target_split.nodes import split_target

def test_split_target_data_type():

    train_with_best_columns = pd.read_csv(train_with_best_columns_path)
    test_with_best_columns = pd.read_csv(test_with_best_columns_path)
    parameters = read_file_as_dict(parameters_path)

    X_train_data, y_train_data, X_test_data, y_test_data = split_target(train_with_best_columns, test_with_best_columns, parameters)

    isinstance(X_train_data, pd.DataFrame)
    isinstance(X_test_data, pd.DataFrame)
    isinstance(y_train_data, pd.Series)
    isinstance(y_test_data, pd.Series)


def test_split_target_size():

    train_with_best_columns = pd.read_csv(train_with_best_columns_path)
    test_with_best_columns = pd.read_csv(test_with_best_columns_path)
    parameters = read_file_as_dict(parameters_path)

    X_train_data, y_train_data, X_test_data, y_test_data = split_target(train_with_best_columns, test_with_best_columns, parameters)

    assert X_train_data.shape[1] == X_test_data.shape[1]
    assert X_train_data.shape[0] == y_train_data.shape[0]
    assert X_test_data.shape[0] == y_test_data.shape[0]