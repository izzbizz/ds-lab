def keep_pass_fail(df):
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
    overall_result_mask = (df['OVERALL_RESULT'] == 'P') | (df['OVERALL_RESULT'] == 'F')
    
    return df[overall_result_mask]

def get_car_age(df):
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
    # convert df['TEST_EDATE'] to datetime
    df['CAR_AGE'] = pd.to_datetime(df['TEST_EDATE']).dt.year - df['MODEL_YEAR']
    
    # add +1 to df['CAR_AGE']
    df['CAR_AGE'] += 1
    
    return df