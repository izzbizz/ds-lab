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


def split_data(features_matrix, target_series, train_size=None, random_state=None):
    """
    Split dataset into training and test sets
    
    Parameters
    ----------
    features_matrix : pandas.DataFrame
        Features/predictors of the dataset
    
    target_series : pandas Series
        Observation labels to be predicted

    train_size : float
        The proportion of the dataset to include in the train split. Should be between 0.0 and 1.0
        
    random_state : int
        Pass an int for reproducible output across multiple function calls.
    
    Returns
    -------
    tuple of numpy.ndarray,
        features_train, features_test, target_train, target_test
        
    """
    from sklearn.model_selection import train_test_split

    features_train, features_test, target_train, target_test = train_test_split(
        features_matrix,
        target_series,
        train_size=train_size,
        random_state=random_state
    )

    return features_train, features_test, target_train, target_test

def encode_split_data(cat_features, cont_features, data, target_series, train_size, random_state=None):
    """
    Encodes categorical features, concatenates encoded features with continuous features and splits data
    
    Parameters
    ----------
    cat_features : list of str
        List of column names in `data` containing categorical features
    
    cont_features : list of str
        List of column names in `data` containing continuous features
    
    data : pandas.DataFrame

                
    target_series : pandas Series
        Observation labels to be predicted

    train_size : float
        The proportion of the dataset to include in the train split. Should be between 0.0 and 1.0
        
    random_state : int
        Pass an int for reproducible output across multiple function calls.
    
    Returns
    -------
    tuple of numpy.ndarray,
        features_train, features_test, target_train, target_test
        
    """
    df_working_copy = data.copy()
    
    # convert categorical columns into `category` dtype
    for column_name in df_working_copy[cat_features].columns:
        df_working_copy[column_name] = df_working_copy[column_name].astype("category")

    features_dummy = pd.get_dummies(df_working_copy[cat_features], drop_first=True)

    features_encoded = pd.concat([df_working_copy[cont_features], features_dummy], axis=1)

    from sklearn.model_selection import train_test_split

    features_train, features_test, target_train, target_test = train_test_split(
        features_encoded,
        target_series,
        train_size=train_size,
        random_state=random_state
    )

    return features_train, features_test, target_train, target_test

