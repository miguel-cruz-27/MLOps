"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

import sklearn


def model_predict(model, X_test: pd.DataFrame, y_test: pd.DataFrame, test_data: pd.DataFrame, parameters: Dict[str, Any],best_cols) -> pd.DataFrame:
    
    X_test_temp = X_test[best_cols].copy()
    test_predict = test_data[best_cols].copy()

    # just to control if the model is working
    preds = model.predict(X_test_temp)
    pred_labels = np.rint(preds)
    recall = sklearn.metrics.recall_score(y_test, pred_labels)
    log = logging.getLogger(__name__)
    log.info("Model recall on test set: %0.2f%%", recall * 100)

    preds = model.predict(test_predict)
    test_predict["prediction"] = preds

    # calculate the predicted probabilites column
    probabilities = model.predict_proba(test_predict)
    first_values = probabilities[:, 0]
    test_predict["predicted_proba"] = first_values

    return test_predict