"""
This is a boilerplate test file for pipeline 'model_train'
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
import matplotlib as mpl
import sys
import os
import pickle
import yaml

# Open the pickle file in read mode
with open("data/04_feature/best_columns.pkl", 'rb') as file:
    # Load the contents of the pickle file
    best_columns = pickle.load(file)
    
with open("data/06_models/tuned_model.pkl", 'rb') as file:
    tuned_model = pickle.load(file)
    
# Verify the data type and assign it to a variable
if isinstance(best_columns, list):
    best_columns_list = best_columns
else:
    # Handle the case if the pickle file does not contain a list
    print("Error: The pickle file does not contain a list.")

# path to data
X_train_data_path = "data/05_model_input/X_train.csv"
X_test_data_path = "data/05_model_input/X_test.csv"
y_train_data_path = "data/05_model_input/y_train.csv"
y_test_data_path = "data/05_model_input/y_test.csv"

parameters_path = r'conf\base\parameters.yml'

def read_file_as_dict(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

sys.path.append("src/hotel_project/pipelines")
from model_train.nodes import model_train

def test_model_train():
    x_train = pd.read_csv(X_train_data_path)
    x_test = pd.read_csv(X_test_data_path)
    y_train = pd.read_csv(y_train_data_path)
    y_test = pd.read_csv(y_test_data_path)
    parameters = read_file_as_dict(parameters_path)

    test_model, output_plot, output_plot_2 = model_train(tuned_model, x_train, x_test, y_train, y_test, parameters, best_columns_list)

    assert isinstance(output_plot, mpl.figure.Figure)
    assert isinstance(output_plot_2, mpl.figure.Figure)