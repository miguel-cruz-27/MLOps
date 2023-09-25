"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.18.1
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, RobustScaler


def delete_duplicates(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Delete duplicates.
    Args:
        data: Data containing features and target.
    Returns:
        data: Data without duplicates.
    """

    df_without_duplicates = data.copy()

    df_without_duplicates.drop_duplicates(inplace = True)

    return df_without_duplicates


def fix_data_types(
    data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> pd.DataFrame:
    """Fix the incorrect data types.
    Args:
        data: Data containing features and target.
    Returns:
        data: Data with the features with the correct data types.
    """

    df_fixed = data.copy()
    test_fixed = test_data.copy()

    df_fixed['reservation_status_date'] = pd.to_datetime(df_fixed['reservation_status_date'])
    df_fixed['children'] = df_fixed['children'].astype('Int64')
    test_fixed['reservation_status_date'] = pd.to_datetime(test_fixed['reservation_status_date'])
    test_fixed['children'] = test_fixed['children'].astype('Int64')

    df_fixed['company'].fillna(value = 'not applicable', inplace = True)
    df_fixed['agent'].fillna(value = 'not applicable', inplace = True)
    test_fixed['company'].fillna(value = 'not applicable', inplace = True)
    test_fixed['agent'].fillna(value = 'not applicable', inplace = True)

    return df_fixed, test_fixed


def remove_incoherences(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Remove the data incoherences.
    Args:
        data: Data containing features and target.
    Returns:
        data: Data without the incoherences.
    """

    df_incoherences = data.copy()

    df_incoherences.drop(df_incoherences[(df_incoherences['adults'] == 0) & (df_incoherences['children'] == 0) & (df_incoherences['babies'] == 0)].index, inplace = True)
    df_incoherences.drop(df_incoherences[(df_incoherences['stays_in_weekend_nights'] == 0) & (df_incoherences['stays_in_week_nights'] == 0)].index, inplace = True)

    df_incoherences['is_repeated_guest'] = np.where((df_incoherences['previous_cancellations'] > 0) & (df_incoherences['is_repeated_guest'] == 0), 1, 0)
    df_incoherences['is_repeated_guest'] = np.where((df_incoherences['previous_bookings_not_canceled'] > 0) & (df_incoherences['is_repeated_guest'] == 0), 1, 0)

    return df_incoherences


def create_features(
    data: pd.DataFrame,
    test_data: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict, Dict]:
    """Create features.
    Args:
        data: Data containing features and target.
    Returns:
        data: Data with extra features.   
    """

    df_with_features = data.copy()
    test_with_features = test_data.copy()

    describe_to_dict = df_with_features.describe().to_dict()

    # creating a dictionary mapping ISO codes to continents
    continent_dict = {
        'DZA': 'Africa','AGO': 'Africa','BEN': 'Africa','BWA': 'Africa','BFA': 'Africa','BDI': 'Africa','CPV': 'Africa','CMR': 'Africa','CAF': 'Africa','TCD': 'Africa','COM': 'Africa','COG': 'Africa','COD': 'Africa','CIV': 'Africa','DJI': 'Africa','EGY': 'Africa','GNQ': 'Africa','ERI': 'Africa','ETH': 'Africa','GAB': 'Africa','GMB': 'Africa','GHA': 'Africa','GIN': 'Africa','GNB': 'Africa','KEN': 'Africa','LSO': 'Africa','LBR': 'Africa','LBY': 'Africa','MDG': 'Africa','MWI': 'Africa','MLI': 'Africa','MRT': 'Africa','MUS': 'Africa','MAR': 'Africa','MOZ': 'Africa','NAM': 'Africa','NER': 'Africa','NGA': 'Africa','RWA': 'Africa','STP': 'Africa','SEN': 'Africa','SYC': 'Africa','SLE': 'Africa','SOM': 'Africa','ZAF': 'Africa','SSD': 'Africa','SDN': 'Africa','SWZ': 'Africa','TZA': 'Africa','TGO': 'Africa','TMP': 'Africa','TUN': 'Africa','UGA': 'Africa','ZMB': 'Africa','ZWE': 'Africa',
        'AFG': 'Asia','ARM': 'Asia','AZE': 'Asia','BHR': 'Asia','BGD': 'Asia','BTN': 'Asia','BRN': 'Asia','KHM': 'Asia','CHN': 'Asia','CN': 'Asia','CCK': 'Asia','CYP': 'Asia','PSE': 'Asia','GEO': 'Asia','HKG': 'Asia','IND': 'Asia','IDN': 'Asia','IRN': 'Asia','IRQ': 'Asia','ISR': 'Asia','JPN': 'Asia','JOR': 'Asia','KAZ': 'Asia','KWT': 'Asia','KGZ': 'Asia','LAO': 'Asia','LBN': 'Asia','MAC': 'Asia','MYS': 'Asia','MDV': 'Asia','MNG': 'Asia','MMR': 'Asia','NPL': 'Asia','PRK': 'Asia','OMN': 'Asia','PAK': 'Asia','PHL': 'Asia','QAT': 'Asia','SAU': 'Asia','SGP': 'Asia','KOR': 'Asia','LKA': 'Asia','SYR': 'Asia','TWN': 'Asia','TJK': 'Asia','THA': 'Asia','TLS': 'Asia','TUR': 'Asia','TKM': 'Asia','ARE': 'Asia','UZB': 'Asia','VNM': 'Asia','YEM': 'Asia',
        'ALB': 'Europe','AIA': 'Europe','AND': 'Europe','ATF': 'Europe','AUT': 'Europe',    'BLR': 'Europe','BEL': 'Europe','BIH': 'Europe','BGR': 'Europe','HRV': 'Europe','CYM': 'Europe','CYP': 'Europe','CZE': 'Europe','DNK': 'Europe','EST': 'Europe','FIN': 'Europe','FRA': 'Europe','FRO': 'Europe','DEU': 'Europe','GRC': 'Europe','GIB': 'Europe','HUN': 'Europe','ISL': 'Europe','IRL': 'Europe','ITA': 'Europe','JEY': 'Europe','KAZ': 'Europe','KOS': 'Europe','IOT': 'Europe','LVA': 'Europe','LIE': 'Europe','LTU': 'Europe','LUX': 'Europe','MKD': 'Europe','MLT': 'Europe','MDA': 'Europe','MCO': 'Europe','MNE': 'Europe','NLD': 'Europe','NCL': 'Europe','NOR': 'Europe','PCN': 'Europe','PYF': 'Europe','POL': 'Europe','PRT': 'Europe','ROU': 'Europe','RUS': 'Europe','SMR': 'Europe','SJM': 'Europe','SRB': 'Europe','SVK': 'Europe','SVN': 'Europe','ESP': 'Europe','SWE': 'Europe','CHE': 'Europe','UKR': 'Europe','GBR': 'Europe','VAT': 'Europe','IMN': 'Europe','GGY': 'Europe','VGB': 'Europe','GLP': 'Europe','MYT': 'Europe',
        'CAN': 'NorthAmerica','MEX': 'NorthAmerica','SPM': 'NorthAmerica','VIR': 'NorthAmerica','USA': 'NorthAmerica','UMI': 'NorthAmerica','ARG': 'SouthAmerica','ABW': 'SouthAmerica','BOL': 'SouthAmerica','BRA': 'SouthAmerica','CHL': 'SouthAmerica','COL': 'SouthAmerica','ECU': 'SouthAmerica','FLK': 'SouthAmerica','GUY': 'SouthAmerica','GUF': 'SouthAmerica','PRY': 'SouthAmerica','PER': 'SouthAmerica','SUR': 'SouthAmerica','URY': 'SouthAmerica','VEN': 'SouthAmerica',
        'AUS': 'Oceania','ASM': 'Oceania','COK': 'Oceania','FJI': 'Oceania','KIR': 'Oceania','MHL': 'Oceania','FSM': 'Oceania','NRU': 'Oceania','NZL': 'Oceania','PLW': 'Oceania','PNG': 'Oceania','WSM': 'Oceania','SLB': 'Oceania','TON': 'Oceania','TUV': 'Oceania','VUT': 'Oceania','WLF': 'Oceania',
        'ATA': 'Antarctica','BVT': 'Antarctica','HMD': 'Antarctica',
        'BLZ': 'CentralAmerica','BMU': 'CentralAmerica','CRI': 'CentralAmerica','SLV': 'CentralAmerica','GTM': 'CentralAmerica','HND': 'CentralAmerica','NIC': 'CentralAmerica','PAN': 'CentralAmerica','ATG': 'CentralAmerica','BHS': 'CentralAmerica','BRB': 'CentralAmerica','CUB': 'CentralAmerica','DMA': 'CentralAmerica','DOM': 'CentralAmerica','GRD': 'CentralAmerica','HTI': 'CentralAmerica','JAM': 'CentralAmerica','KNA': 'CentralAmerica','LCA': 'CentralAmerica','PRI': 'CentralAmerica','VCT': 'CentralAmerica','TTO': 'CentralAmerica'
    }

    # creating feature 'continent'
    df_with_features['continent'] = df_with_features['country'].map(continent_dict)
    test_with_features['continent'] = test_with_features['country'].map(continent_dict)

    # creating feature 'weekend_or_weekday'
    df_with_features['weekend_or_weekday'] = 0
    for i in range(0, len(df_with_features)):
        if df_with_features['stays_in_week_nights'].iloc[i] == 0 and df_with_features['stays_in_weekend_nights'].iloc[i] > 0:
            df_with_features['weekend_or_weekday'].iloc[i] = 'Just Weekend'
        if df_with_features['stays_in_week_nights'].iloc[i] > 0 and df_with_features['stays_in_weekend_nights'].iloc[i] == 0:
            df_with_features['weekend_or_weekday'].iloc[i] = 'Just Weekday'
        if df_with_features['stays_in_week_nights'].iloc[i] > 0 and df_with_features['stays_in_weekend_nights'].iloc[i] > 0:
            df_with_features['weekend_or_weekday'].iloc[i] = 'Both Weekday and Weekend'
        if df_with_features['stays_in_week_nights'].iloc[i] == 0 and df_with_features['stays_in_weekend_nights'].iloc[i] == 0:
            df_with_features['weekend_or_weekday'].iloc[i] = 'Undefined'
    test_with_features['weekend_or_weekday'] = 0
    for i in range(0, len(test_with_features)):
        if test_with_features['stays_in_week_nights'].iloc[i] == 0 and test_with_features['stays_in_weekend_nights'].iloc[i] > 0:
            test_with_features['weekend_or_weekday'].iloc[i] = 'Just Weekend'
        if test_with_features['stays_in_week_nights'].iloc[i] > 0 and test_with_features['stays_in_weekend_nights'].iloc[i] == 0:
            test_with_features['weekend_or_weekday'].iloc[i] = 'Just Weekday'
        if test_with_features['stays_in_week_nights'].iloc[i] > 0 and test_with_features['stays_in_weekend_nights'].iloc[i] > 0:
            test_with_features['weekend_or_weekday'].iloc[i] = 'Both Weekday and Weekend'
        if test_with_features['stays_in_week_nights'].iloc[i] == 0 and test_with_features['stays_in_weekend_nights'].iloc[i] == 0:
            test_with_features['weekend_or_weekday'].iloc[i] = 'Undefined'

    # creating feature 'all_children'
    df_with_features['all_children'] = df_with_features['children'] + df_with_features['babies']
    test_with_features['all_children'] = test_with_features['children'] + test_with_features['babies']

    # creating feature 'adr_pp'
    df_with_features['adr_pp'] = df_with_features['adr'] / (df_with_features['adults'] + df_with_features['children'])
    test_with_features['adr_pp'] = test_with_features['adr'] / (test_with_features['adults'] + test_with_features['children'])
    
    # creating feature 'total_nights
    df_with_features['total_nights'] = df_with_features['stays_in_week_nights'] + df_with_features['stays_in_weekend_nights']
    test_with_features['total_nights'] = test_with_features['stays_in_week_nights'] + test_with_features['stays_in_weekend_nights']

    ###### Converting the months' names to the corresponding numbers
    # defining a dictionary with the corresponding numbers of months
    months_dic = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}

    # converting 'arrival_date_month' names to numbers
    df_with_features['arrival_date_month'].replace(months_dic, inplace = True)
    test_with_features['arrival_date_month'].replace(months_dic, inplace = True)

    # creating feature 'days_between_last_status'
    df_with_features['days_between_last_status'] = (df_with_features['reservation_status_date'] - df_with_features['arrival_date']).dt.days
    test_with_features['days_between_last_status'] = (test_with_features['reservation_status_date'] - test_with_features['arrival_date']).dt.days

    describe_to_dict_verified = df_with_features.describe().to_dict()

    # defining metric and non-metric features
    non_metric_features = ['country', 'market_segment', 'deposit_type', 'reservation_status', 'meal', 'reserved_room_type', 'distribution_channel', 'assigned_room_type',
                           'continent', 'agent', 'weekend_or_weekday', 'is_repeated_guest', 'company', 'customer_type']
    
    metric_features = ['lead_time', 'arrival_date_year', 'arrival_date_week_number', 'arrival_date_day_of_month', 'stays_in_weekend_nights', 'stays_in_week_nights',
                       'adults', 'children', 'babies', 'previous_cancellations', 'previous_bookings_not_canceled', 'booking_changes', 'days_in_waiting_list',
                       'adr', 'required_car_parking_spaces', 'total_of_special_requests', 'all_children', 'adr_pp', 'total_nights', 'days_between_last_status',
                       'arrival_date_month', 'number_holidays', 'temp', 'precip', 'wind_speed', 'cloud_cover', 'visibility']

    return df_with_features, test_with_features, describe_to_dict, describe_to_dict_verified, metric_features, non_metric_features


def joining_categories(
    data: pd.DataFrame,
    test_data: pd.DataFrame
) -> pd.DataFrame:
    """Merge the categories in the specified columns that occur less frequently than the top 5 categories into a new category called 'others'.
    Args:
        data: Data containing features and target.
    Returns:
        data: Data with the categories merged.
    """

    df_cat_merged = data.copy()
    test_cat_merged = test_data.copy()

    to_merge = ['agent', 'company', 'country']

    # iterating through each column specified in to_merge
    for i in range(0, len(to_merge)):

        # count the occurrences of each unique value in the column
        counts = df_cat_merged[to_merge[i]].value_counts()

        # get the categories to merge (all of them except the top 5)
        categories_to_merge = counts[5:].index
        
        # converting the columns to type object
        df_cat_merged[to_merge[i]] = df_cat_merged[to_merge[i]].astype(str)
        test_cat_merged[to_merge[i]] = test_cat_merged[to_merge[i]].astype(str)

        # replace the values in the column that belong to the categories to merge with 'others'
        df_cat_merged.loc[df_cat_merged[to_merge[i]].isin(categories_to_merge), to_merge[i]] = 'others'
        test_cat_merged.loc[test_cat_merged[to_merge[i]].isin(categories_to_merge), to_merge[i]] = 'others'
        
    return df_cat_merged, test_cat_merged


def treat_missing_values(
    data: pd.DataFrame, test: pd.DataFrame, mf: list = None, nmf: list = None
) -> pd.DataFrame:
    """Treat the missing values.
    Args:
        data: Data containing the input data with missing values to be filled.
        test: 
        mf: List of strings containing the names of the features to be filled with the median (metric features).
        nmf: List of strings or array containing the names of the features to be filled with the mode (non-metric features).
        
    Returns:
        data: Data without the missing values.
    """

    df_missing = data.copy()
    test_missing = test.copy()

    # replacing other types of missing values not recognized by pandas with NaN and checking again the number of missing values
    missing_values = ['n/a', 'na', '--', '', 'unknown', 'Unknown', 'NULL']
    df_missing.replace(missing_values, np.nan, inplace = True)
    test_missing.replace(missing_values, np.nan, inplace = True)

    # calculating the medians and the modes of the variables, and then concatenating these two measures into one imputer object
    if mf == None:
        modes = df_missing[nmf].mode().loc[0]
        input_values = modes
    
    if nmf == None:
        medians = df_missing[mf].median()
        input_values = medians

    if (mf != None) & (nmf != None):
        modes = df_missing[nmf].mode().loc[0]
        medians = df_missing[mf].median()
        input_values = pd.concat([medians, modes])
    
    # filling the missing values
    df_missing.fillna(input_values, inplace = True)
    test_missing.fillna(input_values, inplace = True)

    return df_missing, test_missing


def treat_outliers(
    data: pd.DataFrame, outlier_method: str, mf: list
) -> pd.DataFrame:
    """Treat outliers.
    Args:
        data: Data containing features and target.
        outlier_method: Name of the method to use to deal with the outliers.
        mf: List with the names of the metric features.
    Returns:
        data: Data without outliers.
    """

    df_outlier = data.copy()

    if outlier_method == "none":

        # dont do nothing to the outliers
        return df_outlier

    elif outlier_method == "manual":
        
        filter = (
            (df_outlier['lead_time'] <= 365)
            &
            (df_outlier['stays_in_weekend_nights'] <= 6)
            &
            (df_outlier['stays_in_week_nights'] <= 15)
            &
            (df_outlier['previous_cancellations'] <= 10)
            &
            (df_outlier['previous_bookings_not_canceled'] <= 36)
            &
            (df_outlier['days_in_waiting_list'] <= 90)
            & 
            (df_outlier['precip'] <= 50)
        )

    elif outlier_method == "iqr":
        
        # calculating the quartiles and the correspondent inter quartil range
        q25 = df_outlier[mf].quantile(.25)
        q75 = df_outlier[mf].quantile(.75)
        iqr = (q75 - q25)
        
        # defining the limits to search for outliers
        upper_lim = q75 + 1.5 * iqr
        lower_lim = q25 - 1.5 * iqr
        
        # applying the calculated measures to filter the observations in our dataset and transforming the result into a pandas series
        filter = []
        for metric in mf:
            llim = lower_lim[metric]
            ulim = upper_lim[metric]
            filter.append(df_outlier[metric].between(llim, ulim, inclusive = 'both'))
        filter = pd.Series(np.all(filter,0), index = df_outlier.index)

    else:
        raise ValueError(f"Invalid outlier method name: {outlier_method}")
    # applying the filter to our dataframe
    df_outlier = df_outlier[filter]
    return df_outlier


def scale_data(
    data: pd.DataFrame, test: pd.DataFrame, scaler_method: str, mf: list
) -> Tuple[pd.DataFrame, Dict, Dict]:
    """Scale the metric features.
    Args:
        data: Data containing features and target.
        scaler_method: Name of the scaler to use to normalize the metric features.
        mf: List with the names of the metric features to scale.
    Returns:
        data: Data scaled.
    """

    df_scaled = data.copy()
    test_scaled = test.copy()

    describe_to_dict = df_scaled.describe().to_dict()

    if scaler_method == "none":
        return df_scaled, test_scaled, describe_to_dict, describe_to_dict

    elif scaler_method == "standard":
        scaler = StandardScaler()
    elif scaler_method == "min_max":
        scaler = MinMaxScaler()
    elif scaler_method == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Invalid scaler name: {scaler_method}")

    # scaling the data
    scaled_feat = scaler.fit_transform(df_scaled.loc[:, mf])
    scaled_feat_test = scaler.transform(test_scaled.loc[:, mf])

    # and then we assign the scaled data to the dataframe
    df_scaled.loc[:, mf] = scaled_feat
    test_scaled.loc[:, mf] = scaled_feat_test
    
    describe_to_dict_verified = df_scaled.describe().to_dict()

    return df_scaled, test_scaled, describe_to_dict, describe_to_dict_verified


def encoding_features(
    data: pd.DataFrame, test: pd.DataFrame, nmf: list
) -> Tuple[pd.DataFrame, list]:
    """Perform one-hot encoding to the non metric features specified as input. Encode a month feature using a cyclical encoding technique.
    Args:
        data: Data containing features and target.
        nmf: list of strings containing the names of the columns in data to be encoded.
    Returns:
        data: Data encoded.
    """
    
    df_train = data.copy()
    df_test = test.copy()

    # concatenating the two datasets to encode
    df = pd.concat([df_train, df_test])

    # defining an instance of OneHotEncoder
    encoder = OneHotEncoder(sparse_output = False, drop = 'first')
    
    # use it to encode our variables
    ohc_feat = encoder.fit_transform(df.loc[:, nmf])
    
    # after this we get the new names of the features, and transform this new information into a pandas dataframe
    ohc_feat_names = encoder.get_feature_names_out()

    # adding prefix 'ohc_' in the feature names, to make it easier to later identify
    ohc_feat_names = ['ohc_' + name for name in ohc_feat_names]
    ohc_df = pd.DataFrame(columns = ohc_feat_names, data = ohc_feat, index = df.index)
    
    # reassigning df to contain the ohc variables and non_metric_features to contain the new names of the encoded variables
    df_ohc = pd.concat([df.drop(columns = nmf), ohc_df], axis = 1)
    
    # separating again the datasets into training and test
    df_ohc_train = df_ohc.loc[list(df_train.index), :]
    df_ohc_test = df_ohc.loc[list(df_test.index), :]

    # creating the sine and cosine functions
    df_ohc_train['arrival_date_month_sin'] = np.sin((df_ohc_train['arrival_date_month'] - 1) * (2. * np.pi / 12))
    df_ohc_train['arrival_date_month_cos'] = np.cos((df_ohc_train['arrival_date_month'] - 1) * (2. * np.pi / 12))
    df_ohc_test['arrival_date_month_sin'] = np.sin((df_ohc_test['arrival_date_month'] - 1) * (2. * np.pi / 12))
    df_ohc_test['arrival_date_month_cos'] = np.cos((df_ohc_test['arrival_date_month'] - 1) * (2. * np.pi / 12))
    
    return df_ohc_train, df_ohc_test, ohc_feat_names
