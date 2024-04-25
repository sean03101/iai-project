import numpy as np
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler

## Read data --------------------------------------------------------------------------------------------

def load_and_combine_excel_sheets(file_paths):
    '''
    Load and concat CSV files and return a dataframe
    '''
    all_dataframes = []

    for file_path in file_paths:

        df1 = pd.read_excel(file_path, sheet_name=0)
        all_dataframes.append(df1)

        df2 = pd.read_excel(file_path, sheet_name=1)
        all_dataframes.append(df2)
        
    combined_dataframe = pd.concat(all_dataframes, ignore_index=True)
    return combined_dataframe

## target ei ���� --------------------------------------------------------------------------------------------

def calculate_ei(df):
    """
    Remove values that are outliers, and then calculate 'ei' values in the DataFrame based on specific conditions.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.Series: Series containing the 'ei' values.
    """

    df.loc[df['tg02'] <= 10, 'tg02'] = np.nan
    df.loc[df['tg03'] <= 1000, 'tg03'] = np.nan
    df.loc[df['tg04'] <= 1, 'tg04'] = np.nan

    df = df.dropna(axis=0)
    df.reset_index(drop=True, inplace=True)

    df['ei'] = df['tg04'] / (df['tg02'] * df['tg03'] * 0.0003).round(4).astype('float64')

    return df

## UCL, LCL ���� --------------------------------------------------------------------------------------------

def calculate_cumulative_stats(df, target_column, cumulate_jr, upper_value, lower_value):
    """
    Calculate cumulative statistics including mean and control limits (UCL and LCL) for a target column 
    in a DataFrame based on a rolling window of 'cumulate_jr' journal entries. The function filters entries 
    within the specified control limits and recalculates cumulative statistics for the filtered data.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - target_column (str): Name of the target column for which cumulative statistics are calculated.
    - cumulate_jr (int): The number of journal entries to include in the rolling window for cumulative calculation.
    - upper_value (float): Multiplier for calculating the Upper Control Limit (UCL).
    - lower_value (float): Multiplier for calculating the Lower Control Limit (LCL).

    Returns:
    - UCL & LCL updated per jr
    """

    jr_cumulative_mean = []
    jr_cumulative_var = []

    for i in range(len(df['jr'].unique())):

        if i <= 1:
            jr_range = [df['jr'].unique()[0]]
        elif i < cumulate_jr and i > 1:
            jr_range = df['jr'].unique()[0:i]     
        else:
            jr_range = df['jr'].unique()[i-cumulate_jr:i]
        
        filtered_df_tmp = df[df['jr'].isin(jr_range)]

        jr_cumulative_mean.append(filtered_df_tmp[target_column].mean())
        jr_cumulative_var.append(filtered_df_tmp[target_column].var())

    ucl_tmp = np.array(jr_cumulative_mean) + np.array([upper_value]*len(jr_cumulative_mean))
    lcl_tmp = np.array(jr_cumulative_var) - np.array([lower_value]*len(jr_cumulative_mean))

    ucl_tmp[0] = np.nan
    lcl_tmp[0] = np.nan

    df_tmp = pd.DataFrame({
    'jr': df['jr'].unique(),
    'UCL': ucl_tmp,
    'LCL': lcl_tmp
    })

    df = df.merge(df_tmp, on='jr', how='left')

    filter_df = df[(df[target_column] >= df['LCL']) & (df[target_column] <= df['UCL'])]

    filtered_jr_cumulative_mean = []
    filtered_jr_cumulative_var = []

    for i in range(len(df['jr'].unique())):

        if i <= 1:
            jr_range = [df['jr'].unique()[0]] 
        elif i < cumulate_jr and i > 1:
            jr_range = df['jr'].unique()[0:i]  
        else:
            jr_range = df['jr'].unique()[i-cumulate_jr:i]

        filtered_df_tmp = filter_df[filter_df['jr'].isin(jr_range)]

        filtered_jr_cumulative_mean.append(filtered_df_tmp[target_column].mean())
        filtered_jr_cumulative_var.append(filtered_df_tmp[target_column].var())

    ucl = np.array(filtered_jr_cumulative_mean) + np.array([upper_value]*len(filtered_jr_cumulative_mean))
    lcl = np.array(filtered_jr_cumulative_mean) - np.array([lower_value]*len(filtered_jr_cumulative_mean))
    
    df_tmp = pd.DataFrame({
        'jr': df['jr'].unique(),
        'UCL': ucl,
        'LCL': lcl
    })

    df = df.drop(['UCL','LCL'], axis=1)
    df = df.merge(df_tmp, on='jr', how='left')
    df = df.reset_index(drop=True)

    return df

## jr_progress --------------------------------------------------------------------------------------------

def jr_progress(df):
    """
    Calculate 'jr_progress' for each row in the DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - Create 'jr_progress' column to indicate how much progress the instance has made in the process
    """
    df['date'] = pd.to_datetime(df['date'])
    category_start_end_dates = df.groupby('jr')['date'].agg(['min', 'max'])

    def calculate_progress(row):
        start_date = category_start_end_dates.loc[row['jr'], 'min']
        end_date = category_start_end_dates.loc[row['jr'], 'max']
        current_date = row['date']

        if start_date != end_date:
            elapsed_time = (current_date - start_date).total_seconds() / 60.0
            progress = elapsed_time 
            return progress
        else:
            return 1.0

    df['jr_progress'] = df.apply(calculate_progress, axis=1)

    df = df.drop(['stop'], axis=1)
    df = df.reset_index(drop=True)

    return df

## classification ei label -------------------------------------------------------------------------------------------- 
def classify_abnormal_values(df, target_column):
    """
    Classify values in a DataFrame based on their relation to UCL and LCL.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - target_column (str): Name of the target column to classify.

    Adds a new column to the DataFrame, 'is_abnormal', where:
    - 0 indicates the value is within the normal range (between LCL and UCL),
    - 1 indicates the value is below the LCL (abnormally low),
    - 2 indicates the value is above the UCL (abnormally high).

    Also returns the value counts of the 'is_abnormal' column.
    """
    df['is_abnormal'] = np.where(df[target_column] > df['UCL'], 2,
                                 np.where(df[target_column] < df['LCL'], 1, 0))
    
    return df['is_abnormal'].value_counts()


## for regression
def build_dataset_regression(df, window_size, time_gap):
    """
    Build time series dataset for regression.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - window_size (int): Size of the sliding window.
    - time_gap (int): Time gap for delayed prediction.

    Returns:
    - X (list): List of input sequences.
    - y (list): List of corresponding target values.
    - y_date (list): List of additional information for the target values.
    """
    def is_continuous(series):
        return all((series.shift(-1) - series).dropna() == pd.Timedelta(minutes=1))

    X = []
    y = []
    y_date = []

    for i in range(len(df) - window_size + 1 - time_gap):
        subset = df.iloc[i:i+window_size + time_gap]
        if is_continuous(subset['date']):
            X.append(subset.iloc[:window_size])
            y.append(subset.iloc[-1]['ei']) 
            y_date.append(subset.iloc[-1][['is_abnormal','ei','date','UCL','LCL']])

    return X, y, y_date
## for classification
def build_dataset_classifiaciton(df, window_size, time_gap):
    """
    Build time series dataset for classification.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - window_size (int): Size of the sliding window.
    - time_gap (int): Time gap for delayed prediction.

    Returns:
    - X (list): List of input sequences.
    - y (list): List of corresponding target values.
    - y_date (list): List of additional information for the target values.
    """
    def is_continuous(series):
        return all((series.shift(-1) - series).dropna() == pd.Timedelta(minutes=1))

    X = []
    y = []
    y_date = []

    for i in range(len(df) - window_size + 1 - time_gap):
        subset = df.iloc[i:i+window_size + time_gap]
        if is_continuous(subset['date']):
            X.append(subset.iloc[:window_size])
            y.append(subset.iloc[-1]['is_abnormal']) 
            y_date.append(subset.iloc[-1][['is_abnormal','ei','date','UCL','LCL']])

    return X, y, y_date

## jr_window_patch -------------------------------------------------------------------------------------------- 

def jr_window_patch(X):
    """
    Create dummy variable columns for a categorical column in a DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - column_name (str): Name of the categorical column.

    Returns:
    - Create a 'jr_window_patch' column to indicate whether the instance belongs to the same process as the target value or a different process.
    """
    for df_tmp in X:
        category_to_index = {category: idx for idx, category in enumerate(df_tmp['jr'].unique())}
        for category in df_tmp['jr'].unique():
            df_tmp[f'jr_window_patch'] = df_tmp['jr'].apply(lambda x: category_to_index[x]).astype('float64')

    return X

## Data Split --------------------------------------------------------------------------------------------

def split_data(X, y, y_date, test_ratio=0.8):
    """
    Split data into training and testing sets based on discontinuous points in time series data
    and a specified test set ratio.

    Parameters:
    - X (list of pd.DataFrame): List of DataFrames containing features.
    - y (list): Corresponding list of target variables.
    - y_date (list): List of date information corresponding to y.
    - test_ratio (float): Ratio for splitting the data into training and testing sets.

    Returns:
    - Tuple of lists: (X_train, y_train, X_test, y_test, y_test_date)
    """
    def find_discontinuous_points(series):
        discontinuous_points = []
        for i in range(1, len(series)):
            if series.iloc[i] - series.iloc[i-1] != pd.Timedelta(minutes=1):
                discontinuous_points.append(i)
        return discontinuous_points

    break_points = find_discontinuous_points(pd.concat([x['date'] for x in X]))

    break_point = None
    best_ratio = float('inf')
    total_length = len(X)
    test_ratio = 0.8
    for point in break_points:
        ratio = point / total_length
        if abs(ratio - test_ratio) < best_ratio:
            best_ratio = abs(ratio - test_ratio)
            test_best_point = point

    if test_ratio is not None:        
        X_train = X[:test_best_point]
        y_train = y[:test_best_point]
        X_test = X[test_best_point:]
        y_test = y[test_best_point:]
        y_test_date = y_date[test_best_point:]

    X_train = [x.drop(columns=['jr','date','UCL','LCL']) for x in X_train]
    X_test = [x.drop(columns=['jr','date','UCL','LCL']) for x in X_test]

    return X_train, y_train, X_test, y_test, y_test_date

## Data Scaling --------------------------------------------------------------------------------------------

def scale_data(X_train, X_test):
    """
    Scale the training and testing data using MinMaxScaler.

    Parameters:
    - X_train (list of pd.DataFrame): List of DataFrames containing training data.
    - X_test (list of pd.DataFrame): List of DataFrames containing testing data.

    Returns:
    - X_train_scaled (list of np.ndarray): List of scaled training data arrays.
    - X_test_scaled (list of np.ndarray): List of scaled testing data arrays.
    - scaler (MinMaxScaler): The MinMaxScaler instance used for scaling.
    """
    X_train_array = [x.values for x in X_train]
    X_test_array = [x.values for x in X_test]

    scaler = MinMaxScaler().fit(np.concatenate(X_train_array, axis=0))

    X_train_scaled = [scaler.transform(x) for x in X_train_array]
    X_test_scaled = [scaler.transform(x) for x in X_test_array]

    return X_train_scaled, X_test_scaled, scaler

## Time series visualization --------------------------------------------------------------------------------------------

def plot_time_series_regression(y_test_date, all_predictions, start_index, end_index, save_path=None):
    
    """
    This function plots the time series regression between specified indices, including actual values,
    predictions, and the upper control limit (UCL) and lower control limit (LCL) based on the input data.
    It highlights the area between UCL and LCL to visually represent the expected range of values.
    
    Parameters:
    - y_test_date: A list of dictionaries, where each dictionary represents a data point in the time series,
      containing 'date', 'ei' (actual value), 'UCL', and 'LCL' keys.
    - all_predictions: A numpy array or list containing prediction values for the entire dataset.
    - start_index: The starting index from which to plot the data.
    - end_index: The ending index until which to plot the data.
    - save_path: Optional; if provided, the path where the plot image will be saved.

    Outputs:
    - A plot is displayed showing the actual values ('ei') and predictions with the UCL and LCL lines for the specified
      date range. If a save path is provided, the plot is also saved as a PNG file.

    The plot visually compares the actual values against the predictions and shows how they fit within the
    control limits, providing insights into the model's performance and the data's variability over time.
    """
    
    
    UCL_list = np.concatenate([np.array([series['UCL']]) for series in y_test_date if 'UCL' in series])
    LCL_list = np.concatenate([np.array([series['LCL']]) for series in y_test_date if 'LCL' in series])
    date_list = np.concatenate([np.array([series['date']]) for series in y_test_date if 'date' in series])
    ei_list = np.concatenate([np.array([series['ei']]) for series in y_test_date if 'date' in series])

    dates = date_list[start_index:end_index]
    ei_list_sliced = ei_list[start_index:end_index]
    all_predictions_list = all_predictions[start_index:end_index]
    UCL_list_tmp = UCL_list[start_index:end_index]
    LCL_list_tmp = LCL_list[start_index:end_index]

    plt.figure(figsize=(45, 10))
    plt.plot(dates, ei_list_sliced, label="Actual Values", color='blue')
    plt.plot(dates, all_predictions_list, label="Predictions", color='red', linestyle='--')
    plt.plot(dates, UCL_list_tmp, color='black', label=f'UCL (Based on yesterday)', linewidth=2)
    plt.plot(dates, LCL_list_tmp, color='black', label=f'LCL (Based on yesterday)', linewidth=2)
    plt.fill_between(dates, LCL_list_tmp, UCL_list_tmp, color='grey', alpha=0.2)
    plt.title(f"time series prediction_{start_index}~{end_index}")
    plt.savefig(f'{save_path}/time series prediction_{start_index}~{end_index}.png',dpi=300)
    plt.legend(fontsize=20)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %I%p'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())

    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tight_layout()
    plt.show()
    plt.clf()



def plot_time_series_classification(y_test_date, all_predictions, start_index, end_index, save_path=None):
    """
    This function plots the classification results for a time series between specified indices. It shows actual values,
    classifications as predictions (under LCL, over UCL), and the upper and lower control limits (UCL and LCL) based on
    the input data. The function highlights instances classified as under LCL or over UCL with different markers.

    Parameters:
    - y_test_date: A list of dictionaries, with each dictionary representing a data point in the time series,
      containing 'date', 'ei' (actual value), 'UCL', and 'LCL' keys.
    - all_predictions: A numpy array or list containing classification predictions for the entire dataset, where
      0 indicates normal, 1 indicates under LCL, and 2 indicates over UCL.
    - start_index: The starting index from which to plot the data.
    - end_index: The ending index until which to plot the data.
    - save_path: Optional; if provided, the path where the plot image will be saved.

    Outputs:
    - A plot is displayed showing the actual values with classifications for predictions under LCL and over UCL,
      along with UCL and LCL lines for the specified date range. The plot includes markers for classified points,
      facilitating visual assessment of the classification performance over time. If a save path is provided, the plot
      is also saved as a PNG file.

    This visualization is particularly useful for assessing the performance of a classification model in time series
    data, indicating how well the model can identify values falling outside of control limits.
    """
    UCL_list = np.concatenate([np.array([series['UCL']]) for series in y_test_date if 'UCL' in series])
    LCL_list = np.concatenate([np.array([series['LCL']]) for series in y_test_date if 'LCL' in series])
    date_list = np.concatenate([np.array([series['date']]) for series in y_test_date if 'date' in series])
    ei_list = np.concatenate([np.array([series['ei']]) for series in y_test_date if 'date' in series])

    normal_predictions = np.array(all_predictions) != 0
    indices = np.arange(len(all_predictions))[normal_predictions]
    normal_predictions = np.array(all_predictions)[normal_predictions]

    predictions_array = np.array(all_predictions)
    under_LCL_indices = np.where(predictions_array == 1)[0]
    under_LCL = predictions_array[under_LCL_indices]

    over_UCL_indices = np.where(predictions_array == 2)[0]
    over_UCL = predictions_array[over_UCL_indices]

    dates = date_list[start_index:end_index]
    ei_list_sliced = ei_list[start_index:end_index]
    UCL_list_sliced = UCL_list[start_index:end_index]
    LCL_list_sliced = LCL_list[start_index:end_index]

    under_LCL_indices_sliced = [i for i in under_LCL_indices if start_index <= i < end_index]
    over_UCL_indices_sliced = [i for i in over_UCL_indices if start_index <= i < end_index]
    under_LCL_sliced = under_LCL[np.isin(under_LCL_indices, under_LCL_indices_sliced)]
    over_UCL_sliced = over_UCL[np.isin(over_UCL_indices, over_UCL_indices_sliced)]
    under_LCL_dates = [dates[i - start_index] for i in under_LCL_indices_sliced]
    over_UCL_dates = [dates[i - start_index] for i in over_UCL_indices_sliced]

    plt.figure(figsize=(45, 10))
    plt.plot(dates, ei_list_sliced, label="Actual Values", color='blue')
    plt.scatter(under_LCL_dates, under_LCL_sliced, label="Predictions (under LCL)", color='red')
    plt.scatter(over_UCL_dates, over_UCL_sliced, label="Predictions (over UCL)", color='orange')
    plt.plot(dates, UCL_list_sliced, color='black', linestyle='--', label='UCL (Based on yesterday)', linewidth=2)
    plt.plot(dates, LCL_list_sliced, color='black', linestyle='--', label='LCL (Based on yesterday)', linewidth=2)
    plt.fill_between(dates, LCL_list_sliced, UCL_list_sliced, color='grey', alpha=0.2)
    plt.title(f"time series prediction_{start_index}~{end_index}")
    plt.savefig(f'{save_path}/time series prediction_{start_index}~{end_index}.png',dpi=300)
    plt.legend(fontsize=20)

    plt.xlim(dates[0], dates[-1])
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %I%p'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tight_layout() 
    plt.show()
    plt.clf()