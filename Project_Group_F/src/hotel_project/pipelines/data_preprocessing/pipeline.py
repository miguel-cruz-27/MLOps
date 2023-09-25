"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.18.1
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import delete_duplicates, fix_data_types, remove_incoherences, treat_missing_values, treat_outliers, scale_data, create_features, encoding_features, joining_categories

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([

        node(
            func = delete_duplicates,
            inputs = "train_data",
            outputs = "train_deleted_duplicates",
            name = "duplicates",
        ),

        node(
            func = fix_data_types,
            inputs = ["train_deleted_duplicates", "test_data"],
            outputs = ["train_fixed_data_types", "test_fixed_data_types"],
            name = "fix_data_types",
        ),

        node(
            func = remove_incoherences,
            inputs = "train_fixed_data_types",
            outputs = "train_without_incoherences",
            name = "incoherences",
        ),

        node(
            func = create_features, 
            inputs = ["train_without_incoherences", "test_fixed_data_types"],
            outputs = ["train_new_features", "test_new_features", "raw_describe", "new_features_describe", "metric_features", "non_metric_features"],
            name = "new_features",
        ),

        node(
            func = joining_categories,
            inputs = ["train_new_features", "test_new_features"],
            outputs = ["train_cat_joined", "test_cat_joined"],
            name = "cat_joined",
        ),

        node(
            func = treat_missing_values,
            inputs = ["train_cat_joined", "test_cat_joined", "metric_features", "non_metric_features"],
            outputs = ["train_without_missing", "test_without_missing"],
            name = "missing_values"
        ),

        node(
            func = treat_outliers,
            inputs = ["train_without_missing", "params:outlier_removal", "metric_features"],
            outputs = "train_without_outliers",
            name = "outliers",
        ),

        node(
            func = scale_data,
            inputs = ["train_without_outliers", "test_without_missing", "params:scale_data", "metric_features"],
            outputs = ["train_scaled", "test_scaled", "not_scaled_describe", "scaled_describe"],
            name = "scaling"
        ),

        node(
            func = encoding_features,
            inputs = ["train_scaled", "test_scaled", "non_metric_features"],
            outputs = ["train_encoded", "test_encoded", "encoded_non_metric_features"],
            name = "encoding",
        ),
        
    ])
