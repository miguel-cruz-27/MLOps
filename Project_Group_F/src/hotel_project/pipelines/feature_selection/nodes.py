"""
This is a boilerplate pipeline 'feature_selection'
generated using Kedro 0.18.1
"""
import logging
from typing import Any, Dict

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, SequentialFeatureSelector, f_classif, RFE
from sklearn.metrics import recall_score
from sklearn.linear_model import LassoCV, LogisticRegression, Ridge, ElasticNet
from scipy.stats import kendalltau, chi2_contingency
from sklearn.model_selection import train_test_split
from boruta import BorutaPy


def feature_selection(data: pd.DataFrame, data_encoded: pd.DataFrame, test: pd.DataFrame, test_encoded: pd.DataFrame,
                      parameters: Dict[str, Any]):
    """Perform feature selection on the data based on the specified method.

    Args: 
        data[pd.DataFrame]: original training data.
        data_encoded[pd.DataFrame]: encoded training data.
        test[pd.DataFrame]: original test data.
        test_encoded[pd.DataFrame]: encoded test data.
        parameters[Dict[str, Any]]: dictionary containing the parameters for feature selection.

    Returns: 
        tuple: A tuple containing the following elements:
            - train_important (pd.DataFrame): The training data with selected important columns.
            - test_important (pd.DataFrame): The test data with selected important columns.
            - important_columns (List[str]): A list of column names that are selected as important.
    """

    # splitting the data with the non metric features not encoded, because we need them to be in that way
    y = data[parameters["target_column"]]
    X = data.drop(columns = parameters["target_column"], axis = 1)
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify = y, test_size = parameters["test_fraction"], random_state = parameters["random_state"])

    # splitting the data with the non metric features encoded, because we also need them for some methods
    y_encoded = data_encoded[parameters["target_column"]]
    X_encoded = data_encoded.drop(columns = parameters["target_column"], axis = 1)
    X_train_encoded, X_val_encoded, y_train_encoded, y_val_encoded = train_test_split(X_encoded, y_encoded, stratify = y_encoded, test_size = parameters["test_fraction"], random_state = parameters["random_state"])

    # defining again metric and non-metric features without features that could be causing leakage of information about the target
    non_metric_features = ['market_segment', 'deposit_type', 'meal', 'reserved_room_type', 'distribution_channel',
                           'agent', 'weekend_or_weekday', 'is_repeated_guest', 'company', 'customer_type']
    
    metric_features = ['lead_time', 'arrival_date_year', 'arrival_date_week_number', 'arrival_date_day_of_month', 'stays_in_weekend_nights', 'stays_in_week_nights',
                       'adults', 'children', 'babies', 'previous_cancellations', 'previous_bookings_not_canceled', 'booking_changes', 'days_in_waiting_list',
                       'adr', 'required_car_parking_spaces', 'total_of_special_requests', 'all_children', 'adr_pp', 'total_nights',
                       'arrival_date_month', 'number_holidays', 'temp', 'precip', 'wind_speed', 'cloud_cover', 'visibility', 'arrival_date_month_sin', 'arrival_date_month_cos']
    
    non_metric_features_encoded = ['ohc_distribution_channel_Direct', 'ohc_distribution_channel_TA/TO', 'ohc_distribution_channel_Undefined', 
                                   'ohc_deposit_type_Non Refund', 'ohc_deposit_type_Refundable', 'ohc_reserved_room_type_B', 'ohc_reserved_room_type_C', 'ohc_reserved_room_type_D', 'ohc_reserved_room_type_E', 'ohc_reserved_room_type_F', 'ohc_reserved_room_type_G', 'ohc_reserved_room_type_H',
                                   'ohc_reserved_room_type_L', 'ohc_customer_type_Group', 'ohc_customer_type_Transient', 'ohc_customer_type_Transient-Party', 'ohc_meal_FB', 'ohc_meal_HB', 'ohc_meal_SC', 'ohc_meal_Undefined', 
                                   'ohc_weekend_or_weekday_Just Weekday', 'ohc_weekend_or_weekday_Just Weekend', 'ohc_market_segment_Corporate', 'ohc_market_segment_Direct', 'ohc_market_segment_Groups', 
                                   'ohc_market_segment_Offline TA/TO', 'ohc_market_segment_Online TA', 'ohc_company_223.0', 'ohc_company_281.0', 'ohc_company_94.0', 'ohc_company_not applicable', 'ohc_company_others', 
                                   'ohc_is_repeated_guest_1', 'ohc_agent_241.0', 'ohc_agent_250.0', 'ohc_agent_40.0', 'ohc_agent_not applicable', 'ohc_agent_others']
    
    # using this outlier removal method, the rows corresponding to this category are removed
    if parameters["outlier_removal"]=="iqr":
        non_metric_features_encoded.remove('ohc_distribution_channel_Undefined')
    
    # dropping the variables with data leakage/ dates
    X_train_non_metric = X_train[non_metric_features]
    X_train_encoded = X_train_encoded[metric_features + non_metric_features_encoded]
    X_val_encoded = X_val_encoded[metric_features + non_metric_features_encoded]

    important_columns = []

    if parameters["with_feature_selection"] == False:

        important_columns = list(X_train_encoded.columns)
        train_important = data_encoded[important_columns + [parameters["target_column"]]]
        test_important = test_encoded[important_columns + [parameters["target_column"]]]

        return important_columns, train_important, test_important

    if parameters["feature_selection"] == "anova":

        # creating a SelectKBest instance to select features with best ANOVA F-Values
        anova_selector = SelectKBest(score_func = f_classif, k = 'all')
        # applying the selector to the features and target
        anova_selector.fit(X_train_encoded[metric_features], y_train_encoded)

        anova = pd.DataFrame(data = (anova_selector.scores_, anova_selector.pvalues_), columns = X_train_encoded[metric_features].columns, index = ['ANOVA F-statistic', 'ANOVA p-value'])

        important_columns = list(anova.columns[anova.loc['ANOVA p-value'] < 0.05])
        train_important = data_encoded[important_columns + [parameters["target_column"]]]
        test_important = test_encoded[important_columns + [parameters["target_column"]]]


    elif parameters["feature_selection"] == "kendall":

        kendall_tau_list = []
        kendall_p_list = []
        for feature in X_train_encoded[metric_features]:
            tau, p_value = kendalltau(X_train_encoded[metric_features][feature], y_train_encoded)
            kendall_tau_list.append(tau)
            kendall_p_list.append(p_value)

        kendall = pd.DataFrame(data = (kendall_tau_list, kendall_p_list), columns = X_train_encoded[metric_features].columns, index = ['Kendall coefficient', 'Kendall p-value'])

        important_columns = list(kendall.columns[kendall.loc['Kendall p-value'] < 0.05])
        train_important = data_encoded[important_columns + [parameters["target_column"]]]
        test_important = test_encoded[important_columns + [parameters["target_column"]]]


    elif parameters["feature_selection"] == "chi_square":
        features_to_use=[]

        for variable in X_train[non_metric_features]:
            # firstly we compute a simple cross tabulation
            dfObserved = pd.crosstab(y_train, X_train[variable])
            
            # then we perform the chi-square test of independence of the variables in the contingency table
            # we get the test statistic, the p-value of the test, the degrees of fredoom and the expected frequencies
            chi2, p, dof, expected = chi2_contingency(dfObserved.values)
            
            # then we transform the expected frequencies, based on the marginal sums of the table, into a pandas dataframe
            dfExpected = pd.DataFrame(expected, columns = dfObserved.columns, index = dfObserved.index)
            
            # and we make the decision of either the variable is helpful or not
            if p < 0.05:
                features_to_use.append(variable)
            
            if "market_segment" in features_to_use:
                important_columns.append('ohc_market_segment_Corporate')
                important_columns.append('ohc_market_segment_Direct')
                important_columns.append('ohc_market_segment_Groups')
                important_columns.append('ohc_market_segment_Offline TA/TO')
                important_columns.append('ohc_market_segment_Online TA')
            if "deposit_type" in features_to_use:
                important_columns.append('ohc_deposit_type_Non Refund')
                important_columns.append('ohc_deposit_type_Refundable')
            if "meal" in features_to_use:
                important_columns.append('ohc_meal_FB')
                important_columns.append('ohc_meal_HB')
                important_columns.append('ohc_meal_SC')
                important_columns.append('ohc_meal_Undefined')
            if "reserved_room_type" in features_to_use:
                important_columns.append('ohc_reserved_room_type_B')
                important_columns.append('ohc_reserved_room_type_C')
                important_columns.append('ohc_reserved_room_type_D')
                important_columns.append('ohc_reserved_room_type_E')
                important_columns.append('ohc_reserved_room_type_F')
                important_columns.append('ohc_reserved_room_type_G')
                important_columns.append('ohc_reserved_room_type_H')
                important_columns.append('ohc_reserved_room_type_L')
            if "distribution_channel" in features_to_use:
                important_columns.append('ohc_distribution_channel_Direct')
                important_columns.append('ohc_distribution_channel_TA/TO')
                important_columns.append('ohc_distribution_channel_Undefined')
            if "agent" in features_to_use:
                important_columns.append('ohc_agent_241.0')
                important_columns.append('ohc_agent_250.0')
                important_columns.append('ohc_agent_40.0')
                important_columns.append('ohc_agent_not applicable')
                important_columns.append('ohc_agent_others')
            if "weekend_or_weekday" in features_to_use:
                important_columns.append('ohc_weekend_or_weekday_Just Weekday')
                important_columns.append('ohc_weekend_or_weekday_Just Weekend')
            if "is_repeated_guest" in features_to_use:
                important_columns.append('ohc_is_repeated_guest_1')
            if "company" in features_to_use:
                important_columns.append('ohc_company_223.0')
                important_columns.append('ohc_company_281.0')
                important_columns.append('ohc_company_94.0')
                important_columns.append('ohc_company_not applicable')
                important_columns.append('ohc_company_others')
            if "customer_type" in features_to_use:
                important_columns.append('ohc_customer_type_Group')
                important_columns.append('ohc_customer_type_Transient')
                important_columns.append('ohc_customer_type_Transient-Party')

        train_important = data[important_columns + [parameters["target_column"]]]
        test_important = test[important_columns + [parameters["target_column"]]]


    elif parameters["feature_selection"] == "rfe":

        # number of features (we will try from using 1 feature to using all the features)
        nof_list = list(range(1, 28))
        high_score = 0

        # variable to store the optimum features
        nof = 0           
        recall_list = []

        model = LogisticRegression(max_iter=500, random_state = parameters["random_state"])

        for n in nof_list:

            # creating a RFE instance in which the estimator is our model passed to the function
            rfe = RFE(model, n_features_to_select = n).fit(X_train_encoded[metric_features], y_train_encoded)

            # first we select the most important features and we got the most important features to keep in the model
            feat_x_train_rfe = rfe.transform(X_train_encoded[metric_features])
            feat_x_val_rfe = rfe.transform(X_val_encoded[metric_features])

            # then, knowing the features to keep, we apply the model with the desired variables
            model.fit(feat_x_train_rfe, y_train_encoded)
            
            # getting the values that the model predicted for the validation dataset
            feat_y_val_pred = model.predict(feat_x_val_rfe)
            
            # calculating the f1 score for the model and storing it in the list
            value = recall_score(y_val_encoded, feat_y_val_pred)
            recall_list.append(value)
            
            # and we want to keep track of what is the model with the highest score, because that is the perfect number of features
            if(value > high_score):
                high_score = value
                nof = nof_list[n-1]

        # creating a RFE instance with number of features, since we aleady know that is the ideal number of features (previous func)
        rfe = RFE(estimator = model, n_features_to_select = nof).fit(X = X_train_encoded[metric_features], y = y_train_encoded)

        # then, we transform the data using RFE by applying the method fit_transform()
        feat_x_rfe = rfe.transform(X_train_encoded[metric_features])

        # we can see the variables that were selected by RFE as the most "important" ones by calling the attribute support_ 
        # and the feature ranking by calling the attribute ranking_
        selected_features = pd.DataFrame(np.concatenate((rfe.ranking_.reshape(-1, 1), rfe.support_.reshape(-1, 1)), axis = 1), 
                                         columns = ['Ranking', 'Selected'], index = X_train_encoded[metric_features].columns)
        
        important_columns = list(selected_features[selected_features['Ranking'] == 1].index)
        train_important = data_encoded[important_columns + [parameters["target_column"]]]
        test_important = test_encoded[important_columns + [parameters["target_column"]]]

    
    elif parameters["feature_selection"] == "lasso":

        # creating a lasso regression instance
        lasso_reg = LassoCV(random_state = parameters["random_state"])

        # fitting the data to reg
        lasso_reg.fit(X_train_encoded[metric_features], y_train_encoded)

        # assigning the coefficients to the features in a pandas series
        lasso_coef = pd.Series(lasso_reg.coef_, index = X_train_encoded[metric_features].columns)

        important_columns = lasso_coef[lasso_coef > 0.01].index.tolist()
        train_important = data_encoded[important_columns + [parameters["target_column"]]]
        test_important = test_encoded[important_columns + [parameters["target_column"]]]


    elif parameters["feature_selection"] == "ridge":

        # creating a ridge regression instance
        ridge_reg = Ridge(random_state = parameters["random_state"])

        # fitting the data to reg
        ridge_reg.fit(X_train_encoded[metric_features], y_train_encoded)

        # assigning the coefficients to the features in a pandas series
        ridge_coef = pd.Series(ridge_reg.coef_, index = X_train_encoded[metric_features].columns)

        important_columns = ridge_coef[ridge_coef > 0.01].index.tolist()
        train_important = data_encoded[important_columns + [parameters["target_column"]]]
        test_important = test_encoded[important_columns + [parameters["target_column"]]]


    elif parameters["feature_selection"] == "elastic_net":
        # creating an elastic net instance
        e_net = ElasticNet(alpha = 0.1, l1_ratio = 0.2, random_state = parameters["random_state"])
        # fitting the data to the net
        e_net.fit(X_train_encoded[metric_features], y_train_encoded)
        # assigning the coefficients to the features in a pandas series
        enet_coef = pd.Series(e_net.coef_, index = X_train_encoded[metric_features].columns)
        important_columns = enet_coef[enet_coef > 0.001].index.tolist()
        train_important = data_encoded[important_columns + [parameters["target_column"]]]
        test_important = test_encoded[important_columns + [parameters["target_column"]]]


    elif parameters["feature_selection"] == "boruta_random_forest":

        # BorutaPy accepts numpy arrays only, hence the .values attribute
        x_boruta = X_train_encoded.values
        y_boruta = y_train_encoded.values
        y_boruta = y_boruta.ravel()

        # Define Random Forest classifier
        rf = RandomForestClassifier(n_jobs = -1, max_depth = 5, random_state = parameters["random_state"])

        # Define Boruta feature selection method
        feat_selector = BorutaPy(rf, n_estimators = 'auto', verbose = 2, random_state = parameters["random_state"])

        # Find all relevant features 
        feat_selector.fit(x_boruta, y_boruta)

        selected_feat = []
        for i in feat_selector.support_:
            if i==True:
                selected_feat.append('yes')
            else:
                selected_feat.append('no')

        feat_ranking = pd.DataFrame({'Features': X_train_encoded.columns, 'Selected': selected_feat, 'Ranking': feat_selector.ranking_})
        # Sort the DataFrame by ranking
        feat_ranking = feat_ranking.sort_values(by = 'Ranking')

        X_train_base = X_train_encoded.iloc[:, feat_selector.support_ == True]
        # important_columns = list(feat_ranking.loc[feat_ranking['Selected'] == 'yes', 'Features'])

        rf = RandomForestClassifier(random_state = parameters["random_state"], n_jobs = -1)

        # Create the Sequential Feature Selector
        sfs = SequentialFeatureSelector(estimator = rf, cv = 5, n_features_to_select = 'auto', n_jobs = -1)

        # Apply the feature selector on the training data
        selected_features = sfs.fit_transform(np.array(X_train_base), np.array(y_train_encoded).ravel())

        # Get the indices of the selected features
        selected_feature_indices = sfs.get_support(indices=True)

        # Subset the training data with the selected features
        X_train_base = X_train_base.iloc[:, selected_feature_indices]

        important_columns = list(X_train_base.columns)
        train_important = data_encoded[important_columns + [parameters["target_column"]]]
        test_important = test_encoded[important_columns + [parameters["target_column"]]]

    else:
        feature_selection_method = parameters["feature_selection"]
        raise ValueError(f"Invalid feature selection method: {feature_selection_method}")


    log = logging.getLogger(__name__)
    log.info(f"Number of best columns is: {len(important_columns)}")

    return important_columns, train_important, test_important