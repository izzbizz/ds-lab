import pandas as pd
from sklearn.model_selection import train_test_split


def keep_pass_fail(data):
    '''
    Removes rows that don't have pass/fail target variable in `OVERALL_RESULT`

    Parameters
    ----------
    df : pandas.DataFrame containing column `OVERALL_RESULT`

    Returns
    -------
    pandas.DataFrame
    '''
    data = data[data["OVERALL_RESULT"].isin(['P', 'F'])]

    return data


def get_car_age(data):
    '''
    Calculates age of car based on model year and year of emissions test.

    Parameters
    ----------
    data : pandas.DataFrame
        Contains columns `TEST_EDATE` (datetime) and `'MODEL_YEAR'` (int)

    Returns
    -------
    pandas.DataFrame
        DataFrame with new column `CAR_AGE`

    '''
    # make copy of df to prevent SettingWithCopyWarning
    df_copy = data.copy()

    df_copy['CAR_AGE'] = pd.to_datetime(df_copy['TEST_EDATE']).dt.year - df_copy['MODEL_YEAR']

    # add +1 to avoid negative age values
    df_copy['CAR_AGE'] += 1

    return df_copy


def encode_target(target_series):
    '''
    Encode OVERALL_RESULT `"P"` or `"F"` in target variable `OVERALL_RESULT`

    Parameters
    ----------
    target_series : pandas.Series

    '''

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

    train_size : float (between 0.0 - 1.0)
        The proportion of dataset to include in the train split

    random_state : int
        Pass an int for reproducible output across multiple function calls.

    Returns
    -------
    tuple of numpy.ndarray,
        features_train, features_test, target_train, target_test

    """

    features_train, features_test, target_train, target_test = train_test_split(
        features_matrix,
        target_series,
        train_size=train_size,
        random_state=random_state
    )

    return features_train, features_test, target_train, target_test

def encode_split_data(cat_features, data, target_series, train_size, random_state=None):
    """
    Encodes categorical features, concatenates encoded features with continuous features and splits data

    Parameters
    ----------
    cat_features : list of str
        List of column names in `data` containing categorical features

    data : pandas.DataFrame

    target_series : pandas Series
        Observation labels to be predicted

    train_size : float (between 0.0 - 1.0)
        The proportion of dataset to include in the train split

    random_state : int
        Pass an int for reproducible output across multiple function calls.

    Returns
    -------
    tuple of numpy.ndarray,
        features_train, features_test, target_train, target_test

    """
    df_working_copy = data.copy()

    # convert categorical columns into `category` dtype
    for column_name in cat_features:
        df_working_copy[column_name] = df_working_copy[column_name].astype("category")

    features_dummy = pd.get_dummies(df_working_copy[cat_features], drop_first=True)

    cont_features = [col for col in data.columns if not col in cat_features]

    features_encoded = pd.concat([df_working_copy[cont_features], features_dummy], axis=1)

    from sklearn.model_selection import train_test_split

    features_train, features_test, target_train, target_test = train_test_split(
        features_encoded,
        target_series,
        train_size=train_size,
        random_state=random_state
    )

    return features_train, features_test, target_train, target_test
