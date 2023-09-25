"""
This is a boilerplate pipeline 'target_split'
generated using Kedro 0.18.1
"""

from typing import Any, Dict, Tuple

import pandas as pd


def split_target(
    train_data: pd.DataFrame, test_data: pd.DataFrame, parameters: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Splits data into features and target sets.

    Args:
        train:data: Data containing features and target from the training dataset.
        test_data: Data containing features and target from the testing dataset.
        parameters: Parameters defined in parameters.yml.
    Returns:
        Split data into target and features.
    """

    assert [col for col in train_data.columns if train_data[col].isnull().any()] == []

    y_train = train_data[parameters["target_column"]]
    X_train = train_data.drop(columns = parameters["target_column"], axis = 1)

    assert [col for col in test_data.columns if test_data[col].isnull().any()] == []

    y_test = test_data[parameters["target_column"]]
    X_test = test_data.drop(columns = parameters["target_column"], axis = 1)

    return X_train, y_train, X_test, y_test