"""
This is a boilerplate pipeline 'data_merging'
generated using Kedro 0.18.1
"""

import pandas as pd
import numpy as np

def separate_hotels_combine_holidays_weather(
    data: pd.DataFrame, weather: pd.DataFrame
) -> pd.DataFrame:
    """Delete duplicates.
    Args:
        data: Data containing features and target.
    Returns:
        data[pd.DataFrame]: Data without duplicates.
    """

    df_separated = data.copy()

    df_separated = df_separated[df_separated['hotel'] == 'Resort Hotel']
    df_separated.drop(columns = 'hotel', inplace = True)

    ################################################################
    # HOLIDAYS #####################################################
    ################################################################

    # Defining the list of holidays in lisbon
    # Source: https://www.timeanddate.com/holidays/portugal/2015
    holidays = [pd.to_datetime('2015-01-01'), pd.to_datetime('2016-01-01'), pd.to_datetime('2017-01-01'),
                pd.to_datetime('2015-02-17'), pd.to_datetime('2016-02-09'), pd.to_datetime('2017-02-28'),
                pd.to_datetime('2015-04-03'), pd.to_datetime('2016-03-25'), pd.to_datetime('2017-04-14'),
                pd.to_datetime('2015-04-05'), pd.to_datetime('2016-03-27'), pd.to_datetime('2017-04-16'),
                pd.to_datetime('2015-04-25'), pd.to_datetime('2016-04-25'), pd.to_datetime('2017-04-25'),
                pd.to_datetime('2015-05-01'), pd.to_datetime('2016-05-01'), pd.to_datetime('2017-05-01'),
                pd.to_datetime('2016-05-26'), pd.to_datetime('2017-06-15'),
                pd.to_datetime('2015-06-10'), pd.to_datetime('2016-06-10'), pd.to_datetime('2017-06-10'),
                pd.to_datetime('2015-06-13'), pd.to_datetime('2016-06-13'), pd.to_datetime('2017-06-13'),
                pd.to_datetime('2015-08-15'), pd.to_datetime('2016-08-15'), pd.to_datetime('2017-08-15'),
                pd.to_datetime('2016-10-05'), pd.to_datetime('2017-10-05'),
                pd.to_datetime('2016-11-01'), pd.to_datetime('2017-11-01'),
                pd.to_datetime('2016-12-01'), pd.to_datetime('2016-12-01'),
                pd.to_datetime('2015-12-08'), pd.to_datetime('2016-12-08'), pd.to_datetime('2017-12-08'),
                pd.to_datetime('2015-12-25'), pd.to_datetime('2016-12-25'), pd.to_datetime('2017-12-25')]
    
    # Extracting the year, month, and day columns from 'df_separated'
    year = df_separated['arrival_date_year']
    month = df_separated['arrival_date_month']
    day = df_separated['arrival_date_day_of_month']

    # Creating new datetime column ('arrival_date') in df_separated
    df_separated['arrival_date'] = pd.to_datetime(year.astype(str) + '-' + month + '-' + day.astype(str))

    # Creating feature 'departure_date'
    df_separated['departure_date'] = df_separated['arrival_date'] + pd.to_timedelta(df_separated['stays_in_week_nights'] + 
                                                                                    df_separated['stays_in_weekend_nights'], unit = 'd')
    
    # creating the holidays 
    def count_holidays(start_date, end_date):
        """
        inputs:
        - start_date: datetime object representing the start date of a reservation period.
        - end_date: datetime object representing the end date of a reservation period.
        
        function to count the number of holidays that fall within the reservation period (inclusive of start and end dates), returning an integer
        """

        num_holidays = sum(1 for holiday in holidays if start_date <= holiday <= end_date)
        
        return num_holidays
    
    # calculate the number of holidays that the reservation had
    df_separated['number_holidays'] = df_separated.apply(lambda row: count_holidays(row['arrival_date'], row['departure_date']), axis = 1)

    ################################################################
    # WEATHER ######################################################
    ################################################################

    weather.drop(['name', 'tempmax', 'tempmin', 'feelslikemax', 'feelslikemin', 'feelslike', 'severerisk','sunrise',
                  'sunset', 'moonphase', 'conditions', 'description', 'icon', 'stations'], axis = 1, inplace = True)
    
    weather.drop(['dew','humidity','precipprob','precipcover','preciptype','snow','snowdepth','windgust',
                 'winddir','sealevelpressure', 'solarradiation', 'solarenergy', 'uvindex'], axis = 1, inplace = True)

    def average_weather(start_date, end_date, df_weather):
        """
        inputs:
        - start_date: datetime object representing the start date of a reservation period.
        - end_date: datetime object representing the end date of a reservation period.
        - df_weather: pandas DataFrame containing the weather indicators for each day.
        
        function to average the values of the weather indicators that fall within the reservation period (inclusive of start and 
        end dates)
        """
        period = df_weather['datetime'].between(start_date, end_date)
        average_temp = np.mean(df_weather.loc[period, 'temp'].values)
        average_precip = np.mean(df_weather.loc[period, 'precip'].values)
        average_wind_speed = np.mean(df_weather.loc[period, 'windspeed'].values)
        average_cloud_cover = np.mean(df_weather.loc[period, 'cloudcover'].values)
        average_visibility = np.mean(df_weather.loc[period, 'visibility'].values)
        
        return average_temp, average_precip, average_wind_speed, average_cloud_cover, average_visibility
    
    # calculate the average of each weather indicator corresponding to the reservation period
    results = df_separated.apply(lambda row: average_weather(row['arrival_date'], row['departure_date'], weather), axis = 1)
    df_separated['temp'] = list(map(lambda t: t[0], results))
    df_separated['precip'] = list(map(lambda t: t[1], results))
    df_separated['wind_speed'] = list(map(lambda t: t[2], results))
    df_separated['cloud_cover'] = list(map(lambda t: t[3], results))
    df_separated['visibility'] = list(map(lambda t: t[4], results))

    return df_separated