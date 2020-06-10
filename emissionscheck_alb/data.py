import pandas as pd


def keep_pass_fail(data):
    '''
    Removes rows that do not have value `"P"` or `"F"` in target variable `OVERALL_RESULT`
    
    Parameters
    ----------
    df : pandas.DataFrame
        Data-frame containing the column `OVERALL_RESULT`
    Returns
    -------
    pandas.DataFrame
        DataFrame of rows with desired values `'P'` or `'F'` in target variable 
    '''
    # Mask to select only rows with desired target values
    overall_result_mask = (data['OVERALL_RESULT'] == 'P') | (data['OVERALL_RESULT'] == 'F')
    
    return data[overall_result_mask]

def get_car_age(data):
    '''
    Calculates age of car based on model year and year of emissions test.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the columns `TEST_EDATE` and `'MODEL_YEAR'`
        `'MODEL_YEAR'` should be `int` type

    Returns
    -------
    pandas.DataFrame
        DataFrame of rows with new column `CAR_AGE`


    Notes
    -----
    Adding +1 to car age to avoid negative values (many cars are released a year prior to year in car model)

    '''
    # make copy of df
    df_copy = data.copy()

    # convert df['TEST_EDATE'] to datetime

    df_copy.loc[:, 'CAR_AGE'] = pd.to_datetime(df_copy.loc[: ,'TEST_EDATE']).dt.year - df_copy.loc[:, 'MODEL_YEAR']

    # add +1 to df['CAR_AGE']
    df_copy.loc[:, 'CAR_AGE'] += 1

    return df_copy

def encode_target(target_series):
    '''
    Encode OVERALL_RESULT `"P"` or `"F"` in target variable `OVERALL_RESULT`

    Parameters
    ----------
    df : pandas.DataFrame
        Data-frame containing the column `OVERALL_RESULT`
    Returns
    -------
    pandas.DataFrame
        DataFrame with encoded `F` and `P` target variable 
    '''
    # make copy of df

    # Mask to select only rows with desired target values
    target_series = target_series.replace({'P': 0, 'F': 1})

    return target_series