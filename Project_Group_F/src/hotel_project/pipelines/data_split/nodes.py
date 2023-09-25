"""
This is a boilerplate pipeline 'data_split'
generated using Kedro 0.18.1
"""

from typing import Any, Dict, Tuple

import pandas as pd

from sklearn.model_selection import train_test_split


def split_train_test(
    data: pd.DataFrame, parameters: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits data into features and target training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters.yml.
    Returns:
        Split data.
    """

    y = data[parameters["target_column"]]
    X = data.drop(columns = parameters["target_column"], axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = parameters["test_fraction"], random_state = parameters["random_state"])
    train_data = pd.concat([X_train, y_train], axis = 1)
    test_data = pd.concat([X_test, y_test], axis = 1)

    return train_data, test_data
