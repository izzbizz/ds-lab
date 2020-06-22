import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, plot_confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE


def run_baseline_model(features_train, features_test, target_train, target_test, model, random_state=None, class_weight=None, max_fpr=None, max_iter=None, n_estimators=None, max_leaf_nodes=None, n_features_to_select=None):
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

    max_iter: int, default=100
        logreg - Maximum number of iterations taken for the solvers to converge.

    max_leaf_nodes: int, default=None
        rfclassifier - Grow trees with max_leaf_nodes in best-first fashion.

    n_estimators: int, default=100
        rfclassifier - The number of trees in the forest.

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

    elif model == "rf":
        rf = RandomForestClassifier(class_weight=class_weight,n_estimators=n_estimators,max_leaf_nodes=max_leaf_nodes)

        clf_pipeline = make_pipeline(MinMaxScaler(), rf)

    elif model == "rfe":
        rfe = RFE(RandomForestClassifier(n_estimators=n_estimators),n_features_to_select=n_features_to_select)

        clf_pipeline = make_pipeline(MinMaxScaler(), rfe)

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
    mean_cv_score = round(np.mean(cv_score), 2)

    print(f'Cross-validation scores: {cv_score}')
    print(f'Mean cross-validation score: {mean_cv_score}')

    ### summarize feature importance
   # importance = clf_pipeline.coef_
    #for i, v in enumerate(importance):
     #   print('Feature: %0d, Score: %.5f' % (i, v))

    return fitted_model

