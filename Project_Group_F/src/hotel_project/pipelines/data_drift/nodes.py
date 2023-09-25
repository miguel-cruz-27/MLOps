"""
This is a boilerplate pipeline 'data_drift'
generated using Kedro 0.18.1
"""

import pandas as pd
import nannyml as nml
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from IPython.display import display

def data_drift(reference_df, analysis_df, predictions_df, y_test_data):
    column_names = ['lead_time', 'arrival_date_year', 'stays_in_week_nights', 'adults', 'children', 'previous_cancellations', 
                    'previous_bookings_not_canceled', 'booking_changes', 'adr', 'required_car_parking_spaces', 
                    'total_of_special_requests', 'cloud_cover','visibility', 'arrival_date_month_cos', 'ohc_distribution_channel_Direct',
                    'ohc_deposit_type_Non Refund', 'ohc_customer_type_Transient', 'ohc_customer_type_Transient-Party', 'ohc_market_segment_Corporate',
                    'ohc_market_segment_Online TA', 'ohc_agent_241.0', 'ohc_agent_others']
      
    #define the threshold for the test as parameters in the parameters catalog
    constant_threshold = nml.thresholds.StandardDeviationThreshold(std_lower_multiplier=2, std_upper_multiplier=2, offset_from=np.mean)
    constant_threshold.thresholds(reference_df)
    
    #Univariate data drift 
    def univariate_data_drift(reference_df, analysis_df):
        # Initialize the drift calculator
        calc = nml.UnivariateDriftCalculator(
            column_names=column_names,
            continuous_methods=['jensen_shannon'],
            categorical_methods=['chi2'],
            thresholds={"jensen_shannon":constant_threshold})

        # Fit the calculator with the reference DataFrame
        calc.fit(reference_df)

        # Calculate the data drift using the analysis DataFrame
        results = calc.calculate(analysis_df)

        # Plot the drift for continuous columns using Jensen-Shannon method
        figure = results.filter(column_names=results.continuous_column_names, methods=['jensen_shannon']).plot(kind='drift')
        figure.show()
    
        # Plot the distribution of continuous columns using Jensen-Shannon method
        figure = results.filter(column_names=results.continuous_column_names, methods=['jensen_shannon']).plot(kind='distribution')
        figure.show()

        # We do not have categorical columns at this moment because those columns are already with the one hot encoding.
        # Hoever if we used this pipeline before the one hot encoding we can use the code below.

        # Plot the drift for categorical columns using Chi-Square method
        #figure = results.filter(column_names=results.categorical_column_names, methods=['chi2']).plot(kind='drift')
        #figure.show()

        # Plot the distribution of categorical columns using Chi-Square method
        #figure = results.filter(column_names=results.categorical_column_names, methods=['chi2']).plot(kind='distribution')
        #figure.show()
        
        # Return the drift results
        return results
    
    #Multivariate data drift
    def multivariate_data_drift(reference_df, analysis_df):
        calc = nml.DataReconstructionDriftCalculator(
            column_names=column_names)
        
        calc.fit(reference_df)
        results2 = calc.calculate(analysis_df)

        display(results2.filter(period='analysis').to_df())

        display(results2.filter(period='reference').to_df())

        figure = results2.plot()
        figure.show()
        
        # Return the drift results
        return results2
    
    # Estimate Performance
    def estimate_performance(analysis_df, predictions_df, y_test_data):

        # Join columns to the reference_df
        columns_to_append = ['predicted_proba', 'prediction']
        first_df = analysis_df.merge(predictions_df[columns_to_append], left_index=True, right_index=True)
        last_df = first_df.merge(y_test_data['is_canceled'], left_index=True, right_index=True)

        
        estimator = nml.CBPE(
             y_pred_proba= 'predicted_proba',
             y_pred= 'prediction',
             y_true= 'is_canceled',
             metrics=['recall',"accuracy"],
             problem_type='classification_binary',
             thresholds={"recall":constant_threshold})
        
        estimator = estimator.fit(last_df)
        estimated_performance = estimator.estimate(last_df)

        fig = estimated_performance.filter(metrics=['recall', "accuracy"], period='analysis').plot()
        fig.show()

        return estimated_performance

    # Call the univariate and multivariate data drift functions
    univariate_results = univariate_data_drift(reference_df, analysis_df)
    multivariate_results = multivariate_data_drift(reference_df, analysis_df)
    estimate_results = estimate_performance(analysis_df, predictions_df, y_test_data)
    
    # Return the drift results
    return univariate_results, multivariate_results, estimate_results
