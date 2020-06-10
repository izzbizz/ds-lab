import pandas as pd
import numpy as np
import logging


def split_data(features_matrix, target_series, train_size, random_state=None):
    """Permute and split DataFrame index into train and test.
    Parameters
    ----------

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
    Create features matrix with one-hot encoding/ concatenating of categorical features
    Split data
    """
    df_working_copy = data.copy()

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


def run_minimal_model(features_train, features_test, target_train, target_test, model, random_state=None, class_weight=None, max_fpr=None):
    """
    """
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import roc_auc_score, plot_confusion_matrix, classification_report

    if model == "logreg":

        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(class_weight=class_weight, random_state=random_state)

        clf_pipeline = make_pipeline(MinMaxScaler(), lr)

    elif model == "svc":


        from sklearn.svm import SVC
        svc = SVC(class_weight=class_weight)

        clf_pipeline = make_pipeline(MinMaxScaler(), svc)


    # fit model
    fitted_model = clf_pipeline.fit(features_train, target_train)

    # predict labels
    target_pred = clf_pipeline.predict(features_test)

    # predict predicted probabilities
    target_pred_proba = clf_pipeline.predict_proba(features_test)

    # classification report
    print(classification_report(target_test, target_pred))

    # confusion matrix
    plot_confusion_matrix(clf_pipeline, features_test, target_test)

    # calculate ROC
    roc_auc = roc_auc_score(target_test, target_pred_proba[:, 1], max_fpr=max_fpr)

    print(f'ROC AUC Score is {roc_auc}')

    return fitted_model
