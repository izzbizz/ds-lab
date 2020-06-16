import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, plot_confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def run_baseline_model(features_train, features_test, target_train, target_test, model, random_state=None, class_weight=None, max_fpr=None, max_iter=None):
    """
    Run logistic regression or SVC with cross validation

    Parameters
    ----------
    features_train: numpy.ndarray

    features_test: numpy.ndarray

    target_train: numpy.ndarray

    target_test: numpy.ndarray

    model: str, "logreg" or "svc"
        Run logistic regression or support vector classifier

    random_state : int
        Pass an int for reproducible output across multiple function calls.

    class_weight : None or "balanced"
        Adjust weights inversely proportional to class frequencies

    max_fpr: float > 0 and <= 1
        Standardized partial AUC over the range [0, max_fpr] is given

    Notes
    -----
    Shows confusion matrix, classification report, cross val score (ROC AUC)

    """

    if model == "logreg":
        lr = LogisticRegression(class_weight=class_weight, random_state=random_state,max_iter=max_iter)

        clf_pipeline = make_pipeline(MinMaxScaler(), lr)

    elif model == "svc":
        svc = SVC(class_weight=class_weight)

        clf_pipeline = make_pipeline(MinMaxScaler(), svc)

    # fit and predict labels and probabilities
    fitted_model = clf_pipeline.fit(features_train, target_train)

    target_pred = clf_pipeline.predict(features_test)

    target_pred_proba = clf_pipeline.predict_proba(features_test)

    # compute metrics
    print(classification_report(target_test, target_pred))

    plot_confusion_matrix(clf_pipeline, features_test, target_test)

    roc_auc = roc_auc_score(target_test, target_pred_proba[:, 1], max_fpr=max_fpr)
    print(f'ROC AUC Score is {roc_auc}')

    # cross validation
    cv_score = cross_val_score(clf_pipeline, features_train, target_train, scoring="roc_auc")
    mean_cv_score = round(np.mean(cv_score), 2)

    print(f'Cross-validation scores: {cv_score}')
    print(f'Mean cross-validation score: {mean_cv_score}')

    return fitted_model

