"""Project pipelines."""

from typing import Dict
from kedro.pipeline import Pipeline

from hotel_project.pipelines import (
    data_merging as merging,
    data_preprocessing as preprocessing,
    data_split as split_data,
    target_split as split_target,
    hyperparameter_search,
    model_train as train,
    feature_selection as best_features,
    model_predict as predict,
#    data_drift as drift_test,
)

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """

    merging_stage = merging.create_pipeline()
    split_data_stage = split_data.create_pipeline()
#    drift_test_stage = drift_test.create_pipeline()
    preprocessing_stage = preprocessing.create_pipeline()
    split_target_stage = split_target.create_pipeline()
    hyperparameter_search_stage = hyperparameter_search.create_pipeline()
    train_stage = train.create_pipeline()
    feature_selection_stage = best_features.create_pipeline()
    predict_stage = predict.create_pipeline()
    

    return {
        "merging": merging_stage,
        "data_split": split_data_stage,
#        "drift_test" : drift_test_stage,
        "preprocessing": preprocessing_stage,
        "target_split": split_target_stage,
        "hyperparameter_search": hyperparameter_search_stage,
        "train": train_stage,
        "feature_selection": feature_selection_stage,
        "predict": predict_stage,
        "__default__": merging_stage + split_data_stage + preprocessing_stage + feature_selection_stage + split_target_stage + hyperparameter_search_stage + train_stage + predict_stage
    }
