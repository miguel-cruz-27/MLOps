"""
This is a boilerplate pipeline 'data_unit_tests'
generated using Kedro 0.18.1
"""
import logging

import pandas as pd
import great_expectations as ge

def unit_test(
    data: pd.DataFrame, weather: pd.DataFrame
): 

    pd_df_ge = ge.from_pandas(data)

    assert pd_df_ge.expect_table_column_count_to_equal(32).success == True

    assert pd_df_ge.expect_column_to_exist("hotel").success == True
    assert pd_df_ge.expect_column_to_exist("is_canceled").success == True
    assert pd_df_ge.expect_column_to_exist("lead_time").success == True
    assert pd_df_ge.expect_column_to_exist("arrival_date_year").success == True
    assert pd_df_ge.expect_column_to_exist("arrival_date_month").success == True
    assert pd_df_ge.expect_column_to_exist("arrival_date_week_number").success == True
    assert pd_df_ge.expect_column_to_exist("arrival_date_day_of_month").success == True
    assert pd_df_ge.expect_column_to_exist("stays_in_weekend_nights").success == True
    assert pd_df_ge.expect_column_to_exist("stays_in_week_nights").success == True
    assert pd_df_ge.expect_column_to_exist("adults").success == True
    assert pd_df_ge.expect_column_to_exist("children").success == True
    assert pd_df_ge.expect_column_to_exist("babies").success == True
    assert pd_df_ge.expect_column_to_exist("meal").success == True
    assert pd_df_ge.expect_column_to_exist("country").success == True
    assert pd_df_ge.expect_column_to_exist("market_segment").success == True
    assert pd_df_ge.expect_column_to_exist("distribution_channel").success == True
    assert pd_df_ge.expect_column_to_exist("is_repeated_guest").success == True
    assert pd_df_ge.expect_column_to_exist("previous_cancellations").success == True
    assert pd_df_ge.expect_column_to_exist("previous_bookings_not_canceled").success == True
    assert pd_df_ge.expect_column_to_exist("reserved_room_type").success == True
    assert pd_df_ge.expect_column_to_exist("assigned_room_type").success == True
    assert pd_df_ge.expect_column_to_exist("booking_changes").success == True
    assert pd_df_ge.expect_column_to_exist("deposit_type").success == True
    assert pd_df_ge.expect_column_to_exist("agent").success == True
    assert pd_df_ge.expect_column_to_exist("company").success == True
    assert pd_df_ge.expect_column_to_exist("days_in_waiting_list").success == True
    assert pd_df_ge.expect_column_to_exist("customer_type").success == True
    assert pd_df_ge.expect_column_to_exist("adr").success == True
    assert pd_df_ge.expect_column_to_exist("required_car_parking_spaces").success == True
    assert pd_df_ge.expect_column_to_exist("total_of_special_requests").success == True
    assert pd_df_ge.expect_column_to_exist("reservation_status").success == True
    assert pd_df_ge.expect_column_to_exist("reservation_status_date").success == True

    assert pd_df_ge.expect_column_values_to_be_of_type("hotel", "str").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("is_canceled", "int").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("lead_time", "int").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("arrival_date_year", "int").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("arrival_date_month", "str").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("arrival_date_week_number", "int").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("arrival_date_day_of_month", "int").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("stays_in_weekend_nights", "int").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("stays_in_week_nights", "int").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("adults", "int").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("children", "int").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("babies", "int").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("meal", "str").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("country", "str").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("market_segment", "str").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("distribution_channel", "str").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("is_repeated_guest", "int").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("previous_cancellations", "int").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("previous_bookings_not_canceled", "int").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("reserved_room_type", "str").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("assigned_room_type", "str").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("booking_changes", "int").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("deposit_type", "str").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("agent", "int").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("company", "int").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("days_in_waiting_list", "int").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("customer_type", "str").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("adr", "float").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("required_car_parking_spaces", "int").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("total_of_special_requests", "int").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("reservation_status", "str").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("reservation_status_date", "datetime64[ns]").success == True


    # regarding the column with the name of the hotel to which the reservation refers to, it is important that this column does not have any missing values and only has one of the two options for values: "Resort Hotel" or "City Hotel"
    assert pd_df_ge.expect_column_values_to_not_be_null("hotel").success == True
    assert pd_df_ge.expect_column_values_to_be_in_set("hotel", ["Resort Hotel", "City Hotel"]).success == True

    assert pd_df_ge.expect_column_values_to_be_in_set("is_canceled", [0, 1]).success == True
    assert pd_df_ge.expect_column_values_to_be_in_set("arrival_date_year", [2015, 2016, 2017]).success == True
    assert pd_df_ge.expect_column_values_to_be_in_set("arrival_date_month", list(range(1, 13))).success == True
    assert pd_df_ge.expect_column_values_to_be_in_set("arrival_date_week_number", list(range(1, 54))).success == True
    assert pd_df_ge.expect_column_values_to_be_in_set("arrival_date_day_of_month", list(range(1, 32))).success == True
    assert pd_df_ge.expect_column_values_to_be_in_set("is_repeated_guest", [0, 1]).success == True
    assert pd_df_ge.expect_column_values_to_be_in_set("meal", ["SC", "FB", "Undefined", "HB", "BB"]).success == True
    assert pd_df_ge.expect_column_values_to_be_in_set("market_segment", ["Online TA", "Offline TA/TO", "Direct", "Groups", "Corporate", "Complementary"]).success == True
    assert pd_df_ge.expect_column_values_to_be_in_set("distribution_channel", ["TA/TO", "Direct", "Corporate", "Undefined"]).success == True
    assert pd_df_ge.expect_column_values_to_be_in_set("reserved_room_type", ["A", "D", "E", "G", "F", "C", "H", "L", "B", "P"]).success == True
    assert pd_df_ge.expect_column_values_to_be_in_set("assigned_room_type", ["A", "D", "E", "G", "F", "C", "H", "L", "B", "P", "I"]).success == True
    assert pd_df_ge.expect_column_values_to_be_in_set("deposit_type", ["No Deposit", "Non Refund", "Refundable"]).success == True
    assert pd_df_ge.expect_column_values_to_be_in_set("customer_type", ["Transient", "Transient-Party", "Contract", "Group"]).success == True
    assert pd_df_ge.expect_column_values_to_be_in_set("reservation_status", ["Check-Out", "Canceled", "No-Show"]).success == True

    assert pd_df_ge.expect_column_values_to_be_between("lead_time", 0, 9999).success == True
    assert pd_df_ge.expect_column_values_to_be_between("stays_in_weekend_nights", 0, 9999).success == True
    assert pd_df_ge.expect_column_values_to_be_between("stays_in_week_nights", 0, 9999).success == True
    assert pd_df_ge.expect_column_values_to_be_between("adults", 0, 9999).success == True
    assert pd_df_ge.expect_column_values_to_be_between("children", 0, 9999).success == True
    assert pd_df_ge.expect_column_values_to_be_between("babies", 0, 9999).success == True
    assert pd_df_ge.expect_column_values_to_be_between("previous_cancellations", 0, 9999).success == True
    assert pd_df_ge.expect_column_values_to_be_between("previous_bookings_not_canceled", 0, 9999).success == True
    assert pd_df_ge.expect_column_values_to_be_between("booking_changes", 0, 9999).success == True
    assert pd_df_ge.expect_column_values_to_be_between("days_in_waiting_list", 0, 9999).success == True
    assert pd_df_ge.expect_column_values_to_be_between("adr", -9999, 9999).success == True
    assert pd_df_ge.expect_column_values_to_be_between("required_car_parking_spaces", 0, 9999).success == True
    assert pd_df_ge.expect_column_values_to_be_between("total_of_special_requests", 0, 9999).success == True



    pd_weather_ge = ge.from_pandas(weather)

    assert pd_weather_ge.expect_table_column_count_to_equal(33).success == True

    assert pd_weather_ge.expect_column_to_exist("name").success == True
    assert pd_weather_ge.expect_column_to_exist("datetime").success == True
    assert pd_weather_ge.expect_column_to_exist("tempmax").success == True
    assert pd_weather_ge.expect_column_to_exist("tempmin").success == True
    assert pd_weather_ge.expect_column_to_exist("temp").success == True
    assert pd_weather_ge.expect_column_to_exist("feelslikemax").success == True
    assert pd_weather_ge.expect_column_to_exist("feelslikemin").success == True
    assert pd_weather_ge.expect_column_to_exist("feelslike").success == True
    assert pd_weather_ge.expect_column_to_exist("dew").success == True
    assert pd_weather_ge.expect_column_to_exist("humidity").success == True
    assert pd_weather_ge.expect_column_to_exist("precip").success == True
    assert pd_weather_ge.expect_column_to_exist("precipprob").success == True
    assert pd_weather_ge.expect_column_to_exist("precipcover").success == True
    assert pd_weather_ge.expect_column_to_exist("preciptype").success == True
    assert pd_weather_ge.expect_column_to_exist("snow").success == True
    assert pd_weather_ge.expect_column_to_exist("snowdepth").success == True
    assert pd_weather_ge.expect_column_to_exist("windgust").success == True
    assert pd_weather_ge.expect_column_to_exist("windspeed").success == True
    assert pd_weather_ge.expect_column_to_exist("winddir").success == True
    assert pd_weather_ge.expect_column_to_exist("sealevelpressure").success == True
    assert pd_weather_ge.expect_column_to_exist("cloudcover").success == True
    assert pd_weather_ge.expect_column_to_exist("visibility").success == True
    assert pd_weather_ge.expect_column_to_exist("solarradiation").success == True
    assert pd_weather_ge.expect_column_to_exist("solarenergy").success == True
    assert pd_weather_ge.expect_column_to_exist("uvindex").success == True
    assert pd_weather_ge.expect_column_to_exist("severerisk").success == True
    assert pd_weather_ge.expect_column_to_exist("sunrise").success == True
    assert pd_weather_ge.expect_column_to_exist("sunset").success == True
    assert pd_weather_ge.expect_column_to_exist("moonphase").success == True
    assert pd_weather_ge.expect_column_to_exist("conditions").success == True
    assert pd_weather_ge.expect_column_to_exist("description").success == True
    assert pd_weather_ge.expect_column_to_exist("icon").success == True
    assert pd_weather_ge.expect_column_to_exist("stations").success == True

    assert pd_weather_ge.expect_column_values_to_be_of_type("datetime", "datetime64[ns]").success == True
    assert pd_weather_ge.expect_column_values_to_be_of_type("temp", "float").success == True
    assert pd_weather_ge.expect_column_values_to_be_of_type("precip", "float").success == True
    assert pd_weather_ge.expect_column_values_to_be_of_type("windspeed", "float").success == True
    assert pd_weather_ge.expect_column_values_to_be_of_type("cloudcover", "float").success == True
    assert pd_weather_ge.expect_column_values_to_be_of_type("visibility", "float").success == True

    assert pd_weather_ge.expect_column_values_to_be_between("temp", -9999, 9999).success == True
    assert pd_weather_ge.expect_column_values_to_be_between("precip", 0, 9999).success == True
    assert pd_weather_ge.expect_column_values_to_be_between("windspeed", 0, 9999).success == True
    assert pd_weather_ge.expect_column_values_to_be_between("cloudcover", 0, 9999).success == True
    assert pd_weather_ge.expect_column_values_to_be_between("visibility", 0, 9999).success == True


    log = logging.getLogger(__name__)
    log.info("Data passed on the unit data tests")

    return 0