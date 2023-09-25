"""
This is a boilerplate test file for pipeline 'data_preprocessing'
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
import pickle
import yaml

# path to data
train_data_path = "data/02_intermediate/train_data.csv"
train_deleted_duplicates_path = "data/02_intermediate/train_deleted_duplicates.csv"
test_data_path = "data/02_intermediate/test_data.csv"
train_fixed_data_types_path = "data/02_intermediate/train_fixed_data_types.csv"
test_fixed_data_types_path = "data/02_intermediate/test_fixed_data_types.csv"
train_without_incoherences_path = "data/02_intermediate/train_without_incoherences.csv"
train_new_features_path = "data/02_intermediate/train_new_features.csv"
test_new_features_path = "data/02_intermediate/test_new_features.csv"
train_cat_joined_path = "data/02_intermediate/train_cat_joined.csv"
test_cat_joined_path = "data/02_intermediate/test_cat_joined.csv"
train_without_missing_path = "data/02_intermediate/train_without_missing.csv"
test_without_missing_path = "data/02_intermediate/test_without_missing.csv"
train_without_outliers_path = "data/02_intermediate/train_without_outliers.csv"
train_scaled_path = "data/02_intermediate/train_scaled.csv"
test_scaled_path = "data/02_intermediate/test_scaled.csv"
train_encoded_path = "data/02_intermediate/train_encoded.csv"
test_encoded_path = "data/02_intermediate/test_encoded.csv"

metric_features_path = "data/04_feature/metric_features.pkl"
non_metric_features_path = "data/04_feature/non_metric_features.pkl"
encoded_non_metric_features_path = "data/04_feature/encoded_non_metric_features.pkl"

parameters_path = r'conf\base\parameters.yml'

def read_pickle_file_as_list(file_path):
    with open(file_path, 'rb') as file:
        data_list = pickle.load(file)
    return data_list

def read_file_as_dict(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

sys.path.append("src/hotel_project/pipelines")
from data_preprocessing.nodes import delete_duplicates, fix_data_types, remove_incoherences, treat_missing_values, treat_outliers, scale_data, create_features, encoding_features, joining_categories

def test_delete_duplicates():
    df = pd.read_csv(train_data_path)
    df_deleted_duplicates = delete_duplicates(df)

    assert df_deleted_duplicates.duplicated().sum()==0
    
def test_fix_data_types_na():
    df_train = pd.read_csv(train_deleted_duplicates_path)
    df_test = pd.read_csv(test_data_path)
    
    train_fixed_data_types, test_fixed_data_types = fix_data_types(df_train,df_test)
    
    assert train_fixed_data_types["company"].isna().sum()==0
    assert test_fixed_data_types["company"].isna().sum()==0
    assert train_fixed_data_types["agent"].isna().sum()==0
    assert test_fixed_data_types["agent"].isna().sum()==0
    
def test_fix_data_types_dt_int():
    df_train = pd.read_csv(train_deleted_duplicates_path)
    df_test = pd.read_csv(test_data_path)
    
    train_fixed_data_types, test_fixed_data_types = fix_data_types(df_train,df_test)
    
    assert pd.api.types.is_datetime64_any_dtype(train_fixed_data_types['reservation_status_date'])
    assert pd.api.types.is_datetime64_any_dtype(test_fixed_data_types['reservation_status_date'])
    assert pd.api.types.is_integer_dtype(train_fixed_data_types['children'])
    assert pd.api.types.is_integer_dtype(test_fixed_data_types['children'])
    
def test_remove_incoherences():
    df = pd.read_csv(train_fixed_data_types_path)
    
    df_removed_incoherences = remove_incoherences(df)
    
    assert not ((df_removed_incoherences['adults'] == 0) & (df_removed_incoherences['children'] == 0) & (df_removed_incoherences['babies'] == 0)).any() 
    assert not ((df_removed_incoherences['stays_in_weekend_nights'] == 0) & (df_removed_incoherences['stays_in_week_nights'] == 0)).any()
    
def test_create_features():
    df_train = pd.read_csv(train_without_incoherences_path, parse_dates=["reservation_status_date","arrival_date"])   
    df_test = pd.read_csv(test_fixed_data_types_path,parse_dates=["reservation_status_date","arrival_date"]) 
    
    train_new_features, test_new_features, raw_describe, new_features_describe, metric_features, non_metric_features = create_features(df_train,df_test)
    
    assert train_new_features.shape[1]==45
    assert test_new_features.shape[1]==45

def test_joining_categories():
    df_train = pd.read_csv(train_new_features_path)   
    df_test = pd.read_csv(test_new_features_path) 
    
    train_cat_joined, test_cat_joined = joining_categories(df_train, df_test)
  
    assert "others" in train_cat_joined["company"].unique()
    assert "others" in train_cat_joined["agent"].unique()
    assert "others" in train_cat_joined["country"].unique()
    
def test_treat_missing_values():
    df_train = pd.read_csv(train_cat_joined_path)   
    df_test = pd.read_csv(test_cat_joined_path) 
    metric_features = read_pickle_file_as_list(metric_features_path)
    non_metric_features = read_pickle_file_as_list(non_metric_features_path)
    
    train_without_missing, test_without_missing = treat_missing_values(df_train,df_test,metric_features,non_metric_features)
    
    assert train_without_missing.isna().sum().sum()==0
    assert test_without_missing.isna().sum().sum()==0
    
def test_treat_outliers():
    df_train = pd.read_csv(train_without_missing_path)
    parameters = read_file_as_dict(parameters_path)
    metric_features = read_pickle_file_as_list(metric_features_path)
    
    train_without_outliers = treat_outliers(df_train,parameters["outlier_removal"],metric_features)
    
    isinstance(train_without_outliers, pd.DataFrame)
    
def test_scale_data():
    df_train = pd.read_csv(train_without_outliers_path)
    df_test = pd.read_csv(test_without_missing_path)
    parameters = read_file_as_dict(parameters_path)
    metric_features = read_pickle_file_as_list(metric_features_path)
     
    train_scaled, test_scaled, not_scaled_describe, scaled_describe = scale_data(df_train,df_test, parameters["scale_data"],metric_features)         
    isinstance(train_scaled, pd.DataFrame)   
    isinstance(test_scaled, pd.DataFrame) 
    
def test_encoding_features():
    df_train = pd.read_csv(train_scaled_path)
    df_test = pd.read_csv(test_scaled_path)
    non_metric_features = read_pickle_file_as_list(non_metric_features_path)
     
    train_encoded, test_encoded, encoded_non_metric_features = encoding_features(df_train,df_test,non_metric_features)         
    isinstance(train_encoded, pd.DataFrame)   
    isinstance(test_encoded, pd.DataFrame) 