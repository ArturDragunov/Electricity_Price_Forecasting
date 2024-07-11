import pandas as pd
import plotly.graph_objs as go
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import seaborn as sns
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import re
import shap
from datetime import datetime, timedelta
from tqdm import tqdm
import lightgbm as lgb
import itertools
from collections import defaultdict


#%% Function to fit the model and predict
def fit_and_predict(df, params, train_end, predict_end, train_window_days=365):
    # Define the training and prediction periods
    train_start = train_end - timedelta(days=train_window_days)
    
    # Create masks for training and prediction sets
    mask_train = (df.index >= train_start) & (df.index <= train_end)
    mask_predict = (df.index > train_end) & (df.index < predict_end)
    
    # Prepare training and prediction data
    train_X, train_y = df[mask_train].drop(columns=['price']), df[mask_train]['price']
    predict_X = df[mask_predict].drop(columns=['price'])
    
    # Train the model
    train_data = lgb.Dataset(train_X, label=train_y)
    model = lgb.train(params, train_data)
    
    # Make predictions
    predictions = model.predict(predict_X)
    
    # Save predictions in a DataFrame
    result_df = pd.DataFrame({
        'predicted': predictions
    }, index=predict_X.index)
    
    return result_df, model

#%% Visualize features importances
def plot_importances(split_importances, gain_importances, top_n=25):
    split_imp = defaultdict(int)
    gain_imp = defaultdict(int)
    
    for importances in split_importances:
        for name, value in importances:
            split_imp[name] += value
    
    for importances in gain_importances:
        for name, value in importances:
            gain_imp[name] += value
    
    split_imp = list(split_imp.items())
    gain_imp = list(gain_imp.items())
    
    split_imp = list(reversed(sorted(split_imp, key=lambda x: x[1])))
    gain_imp = list(reversed(sorted(gain_imp, key=lambda x: x[1])))
    
    # Select top N features from each importance metric
    top_split_features = [feature for feature, _ in split_imp[:top_n]]
    top_gain_features = [feature for feature, _ in gain_imp[:top_n]]
    
    # Plotting the importances
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    x = top_n
    
    ax1.barh([a for a, b in split_imp[:x]], [b for a, b in split_imp][:x])
    ax1.set_xlabel('Importance')
    ax1.set_ylabel('Feature')
    ax1.set_title('Feature Split Importances from LightGBM Model')
    ax1.invert_yaxis()  # Invert y-axis to have the most important at the top
    
    ax2.barh([a for a, b in gain_imp][:x], [b for a, b in gain_imp][:x])
    ax2.set_xlabel('Importance')
    ax2.set_ylabel('Feature')
    ax2.set_title('Feature Gain Importances from LightGBM Model')
    ax2.invert_yaxis()  # Invert y-axis to have the most important at the top
    
    plt.tight_layout()
    plt.show()
    
    return top_split_features, top_gain_features

#%% Hyperparameter Grid Search
# Function to perform manual grid search
def manual_grid_search(df, param_grid, default_params, start_date, end_date):
    results = []
    param_combinations = list(itertools.product(*param_grid.values()))
    param_names = list(param_grid.keys())
    
    for param_comb in param_combinations:
        params = default_params.copy()
        params.update(dict(zip(param_names, param_comb)))
        
        print(f"Training with parameters: {params}")
        
        rmse, _, _, _, _ = backtest_model(df, params, start_date, end_date)
        
        results.append({
            'params': params,
            'rmse': rmse
        })
    
    return results


#%% Shapley values
# Function to plot SHAP importances
def plot_shap_importances(model, X):
    """
    SHAP measures the impact of variables taking into account the interaction with other variables.

    Shapley values calculate the importance of a feature by comparing what a model predicts with and without the feature. 
    However, since the order in which a model sees features can affect its predictions, this is done in every possible order,
    so that the features are fairly compared.
    This plot shows the 20 most important features. For each feature a distribution is plotted on how the train samples
    influence the model outcome. The more red the dots, the higher the feature value, the more blue the lower the feature value.

    X axis of the plot corresponds to the values of the target, and color corresponds to the values of the feature
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X)

#%% Backtest
# Function to backtest the model using sliding window approach
def backtest_model(df, params, predict_start_, predict_end_, train_window_days=365, predict_window_days=61): # 61 as it's in May and June
    results = []
    models = []
    gain_importances = []
    split_importances = []

    predict_start = datetime.strptime(predict_start_, "%Y-%m-%d")
    predict_end = datetime.strptime(predict_end_, "%Y-%m-%d")
    current_date = predict_start

    total_iterations = (predict_end - predict_start).days // predict_window_days
    progress_bar = tqdm(total=total_iterations)

    while current_date < predict_end:
        predictset_start = current_date
        predictset_end = current_date + timedelta(days=predict_window_days)
        if predictset_end > predict_end:
            predictset_end = predict_end

        trainset_start = predictset_start - timedelta(days=train_window_days)
        trainset_end = predictset_start
        mask_predict = (df.index >= pd.to_datetime(predictset_start)) & (df.index < pd.to_datetime(predictset_end))
        mask_train = (df.index >= pd.to_datetime(trainset_start)) & (df.index < pd.to_datetime(trainset_end))


        if mask_predict.sum() == 0:
            current_date += timedelta(days=predict_window_days)
            continue

        train_X, train_y = df[mask_train].drop(columns=['price']), df[mask_train]['price']
        test_X, test_y = df[mask_predict].drop(columns=['price']), df[mask_predict]['price']
        train_data = lgb.Dataset(train_X, label=train_y)
        model = lgb.train(params, train_data)
        y_pred = model.predict(test_X)

        models.append(model)

        result_ = pd.DataFrame({
            'actual': test_y,
            'predicted': y_pred
        }, index=test_y.index)

        results.append(result_)

        importance_split = model.feature_importance(importance_type='split')
        importance_gain = model.feature_importance(importance_type='gain')
        feature_names = model.feature_name()
        importance_split_dict = dict(zip(feature_names, importance_split))
        importance_gain_dict = dict(zip(feature_names, importance_gain))
        sorted_importance_split = sorted(importance_split_dict.items(), key=lambda x: x[1], reverse=True)
        sorted_importance_gain = sorted(importance_gain_dict.items(), key=lambda x: x[1], reverse=True)
        gain_importances.append(sorted_importance_gain)
        split_importances.append(sorted_importance_split)

        current_date += timedelta(days=predict_window_days)
        progress_bar.update(1)

    progress_bar.close()

    all_results = pd.concat(results)
    rmse = np.sqrt(mean_squared_error(all_results['actual'], all_results['predicted']))
    return rmse, all_results, gain_importances, split_importances, models

#%% Seasonality variables
# Define RBF function
def radial_basis_function(x, m, alpha=1000): # alpha is set
    return np.exp(-0.5 * (x - m)**2 / alpha)

def fourier_transform(data, col, t, sp, k=1):
    data[f'{col}_cos_{sp}_{k}'] = np.cos(2 * np.pi * k * t / sp)
    data[f'{col}_sin_{sp}_{k}'] = np.sin(2 * np.pi * k * t / sp)
    cols = [f'{col}_cos_{sp}_{k}', f'{col}_sin_{sp}_{k}']
    return data, cols

def create_fourier_terms(data):
    time_idx = data['timestamp_rt']
    data['day_of_year'] = time_idx.dt.dayofyear
    data['week_of_year'] = time_idx.dt.isocalendar().week
    data['day_of_week'] = time_idx.dt.dayofweek
    cols = []
    
    # Intraday Fourier terms
    t_hour = time_idx.dt.hour
    for k in range(1, 3):
        data, cols_ = fourier_transform(data, col='hour', t=t_hour, sp=24, k=k)
        cols.extend(cols_)
    
    # Weekly Fourier terms
    data, cols_ = fourier_transform(data, col='week', t=data.day_of_week, sp=7, k=1)
    cols.extend(cols_)
    
    # Yearly Fourier terms
    data, cols_ = fourier_transform(data, col='year', t=data.day_of_year, sp=365.25, k=1)
    cols.extend(cols_)
    data.drop(columns=['day_of_year','day_of_week','week_of_year'],inplace=True)
    return data, cols

def create_RBF_features(data, alpha, col):
    length = data[col].nunique()
    for h in range(length):
        data[f'rbf_{h}'] = radial_basis_function(data[col].astype(int), m=h, alpha=alpha)
    return data

#%% daily and weekly profiles
def prepare_data_for_profile(df):
    data = df.copy()
    data = data[data['timestamp_rt'] < pd.Timestamp('2021-01-01')]
    data['day_of_week'] = data['timestamp_rt'].dt.strftime('%A')  # Day of the week as a string
    data['hour'] = data['timestamp_rt'].dt.hour  # Hour as an integer
    return data

def plot_weekly_profile(df, node_to_analyze):
    data = prepare_data_for_profile(df)
    
    # Group the data by day of week and hour, calculate median price
    result = data.groupby(['day_of_week', 'hour'])['price'].median().reset_index()
    
    # Define custom order for days of the week
    custom_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    result['day_of_week'] = pd.Categorical(result['day_of_week'], categories=custom_order, ordered=True)
    result = result.sort_values(['day_of_week', 'hour'])
    
    # Create the subplots (3 per row)
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 12), sharex=True)
    fig.suptitle(f'Weekly Profile of Median Hourly Prices for {node_to_analyze}', fontsize=16)
    
    # Plot for each day
    for i, day in enumerate(custom_order):
        row, col = divmod(i, 3)
        ax = axes[row, col]
        sns.lineplot(x='hour', y='price', data=result[result['day_of_week'] == day], ax=ax)
        ax.set_title(day)
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Median Price')
        ax.set_xticks(range(0, 24))
        ax.grid(True)
    
    # Hide the unused subplot (bottom-right)
    if len(custom_order) % 3:
        fig.delaxes(axes[-1, -1])
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

#%% predict autoarima
def predict_and_adjust(series, train_end_date, forecast_start_date, forecast_end_date):
    # Convert dates to datetime.date
    train_start_date = pd.to_datetime('2019-01-15').date()
    train_end_date = pd.to_datetime(train_end_date).date()
    forecast_start_date = pd.to_datetime(forecast_start_date).date()
    forecast_end_date = pd.to_datetime(forecast_end_date).date()
    
    # Train the model
    train_data = series[train_start_date:train_end_date]  # Use all data up to the train_end_date
    print(f"Training data for {series.name}:")
    print(train_data.tail())

    # Fit auto ARIMA
    model = auto_arima(train_data, seasonal=True, m=7, suppress_warnings=True, stepwise=True)  # Weekly seasonality
    
    # Get the best order
    order = model.order
    seasonal_order = model.seasonal_order
    
    # Refit ARIMA with the best order
    final_model = ARIMA(train_data, order=order, seasonal_order=seasonal_order)
    fitted_model = final_model.fit()
    
    # Forecast
    forecast_steps = len(pd.date_range(start=forecast_start_date, end=forecast_end_date, freq='D'))
    forecast = fitted_model.forecast(steps=forecast_steps)
    
    # Create a new index for the forecast to start from forecast_start_date
    new_forecast_index = pd.date_range(start=forecast_start_date, periods=forecast_steps, freq='D')
    forecast.index = new_forecast_index

    april_start = pd.to_datetime('2022-04-01').date()
    april_end = pd.to_datetime('2022-04-30').date()
    # Calculate the adjustment
    recent_median = series[april_start:april_end].median()
    forecast_median = forecast.median()
    adjustment = recent_median - forecast_median
    # Apply the adjustment
    adjusted_forecast = forecast + adjustment
    adjusted_forecast.index = new_forecast_index

    print(f"Adjusted forecast for {series.name}:")
    print(adjusted_forecast)
    
    return adjusted_forecast

#%% Function to get the previous year's data for a specific range
def get_previous_year_data(df, year):
    start_date = f'{year-1}-05-01 00:00:00'
    end_date = f'{year-1}-06-30 23:00:00'
    return df.loc[(slice(pd.to_datetime(start_date), pd.to_datetime(end_date)), slice(None)), :].copy()

#%% Function to prepare new data for imputation
def prepare_new_data(df, variables_to_impute):
    # Get the last year's data for May and June
    new_data = get_previous_year_data(df, 2022)
    
    # Extract the timestamp_rt level and shift it by one year
    new_timestamp_rt = new_data.index.get_level_values('timestamp_rt') + pd.DateOffset(years=1)
    
    # Create a new MultiIndex with the shifted timestamp_rt and the original node
    new_index = pd.MultiIndex.from_arrays([new_timestamp_rt, new_data.index.get_level_values('node')],
                                          names=['timestamp_rt', 'node'])
    new_data.index = new_index
    
    # Set all columns not in variables_to_impute to NaN
    for col in new_data.columns:
        if col not in variables_to_impute:
            new_data[col] = np.nan
    
    return new_data

#%% Function to fill WIND values from the previous day for specific dates
def fill_wind_from_previous_day(df, target_date, source_date):
    # Ensure 'datetime' is in datetime format
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Extract the dates
    target_mask = df['datetime'].dt.date == pd.to_datetime(target_date).date()
    source_mask = df['datetime'].dt.date == pd.to_datetime(source_date).date()
    
    # Get the source and target data
    source_wind = df.loc[source_mask, ['datetime', 'WIND']].copy()
    source_wind['datetime'] = source_wind['datetime'] + pd.DateOffset(days=1)
    
    # Map the WIND values from source to target
    df.loc[target_mask, 'WIND'] = df.loc[target_mask, 'WIND'].fillna(
        df.loc[target_mask, 'datetime'].map(source_wind.set_index('datetime')['WIND'])
    )

#%% check na
def check_nan_values(x, threshold=1):
    """
    Returns a Pandas Series with column names and their corresponding count of missing values.
    """
    cols_with_nans = x.columns[x.isna().any().values]
    nan_counts = x[cols_with_nans].isna().sum()
    nan_counts = nan_counts[nan_counts > threshold]  # Filter by threshold
    nan_counts = nan_counts.sort_values(ascending=False)  # Sort in descending order
    
    return nan_counts

#%% Print rows where missing values
def missing_rows_nan(Xt,specific_column = False):
    """
    Filters out only those rows where are nan values. If specific column is not specified, then checks for all columns. Otherwise, for only one.
    """
    if specific_column:
        df = Xt[Xt[specific_column].isnull()]
    else:
        df = Xt[Xt.isnull().any(axis=1)]
    return df

#%% Visualization of results

def create_plot(df_pred_res, targets = ['rt_price', 'da_price', 'rt_price_hat'],width=2000,height=800):
    """
    Creates a plotly plot out of time series
    Requirement: timestamp_rt should be of datatime index.
    """
    fig = go.Figure()

    node_list = df_pred_res['node'].unique()

    buttons = []

    # Create traces for each node and targets
    for i, node in enumerate(node_list):
        visible = [False] * len(node_list)
        visible[i] = True

        # Create a button object for the node we are on
        button = dict(
            label=node,
            method="update",
            args=[{"visible": visible}]
        )

        # Add the button to our list of buttons
        buttons.append(button)

    # Create traces for each target
    for target in targets:
        for node in node_list:
            fig.add_trace(
                go.Scatter(
                    x=df_pred_res['timestamp_rt'][df_pred_res['node'] == node],
                    y=df_pred_res[target][df_pred_res['node'] == node],
                    name=f"{target} - {node}",
                    visible=False
                )
            )

    # Initialize the chart with the first node
    for target in targets:
        fig.update_traces(visible=True, selector=dict(name=f"{target} - {node_list[0]}"))

    fig.update_layout(
        updatemenus=[
            dict(
                active=0,
                type="dropdown",
                buttons=buttons,
                x=0,
                y=1.1,
                xanchor='left',
                yanchor='bottom'
            )
        ],
        # legend=dict(x=0, y=1.1, xanchor='left', yanchor='bottom')  # Add this line to move the legend
    )

    fig.update_layout(
        autosize=False,
        width=width,
        height=height
    )

    fig.show()


#%% Drop columns
def drop_columns_by_pattern(df, pattern_to_drop, pattern_to_keep_from_drop=""):
    """
    pattern_to_drop is a string specifying the part of the name of the columns which you want to exclude from the dataset.
    pattern_to_keep_from_drop specifies those column names which could be matched by pattern_to_drop but which you still want to keep.
    Example:
        ut2.drop_columns_by_pattern(data,'TGC_ratio','lf_meteologica_PJM_RECO.trz_TGC_ratio')
    """
    # Find columns based on the pattern
    columns_to_drop = [col for col in df.columns if re.search(pattern_to_drop, col) and not re.search(pattern_to_keep_from_drop, col)]
    
    # Drop the columns
    df = df.drop(columns=columns_to_drop)
    
    return df

#%% Keep columns by pattern
def keep_columns_by_patterns(df, patterns_to_keep):
    """
    patterns_to_keep is a list of regex patterns which you want to keep in your dataset. Drop everything else.
    """
    # Find columns based on the patterns
    columns_to_drop = [col for col in df.columns if not any(re.search(pattern, col) for pattern in patterns_to_keep)]

    # Drop the columns
    df = df.drop(columns=columns_to_drop)

    return df