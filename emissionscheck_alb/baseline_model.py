import pandas as pd
import numpy as np


def run_baseline_model(features_train, features_test, target_train, target_test, model, random_state=None, class_weight=None, max_fpr=None):
    """
    Run logistic regression or SVC  
    
    """
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import roc_auc_score, plot_confusion_matrix, classification_report
    from sklearn.model_selection import cross_val_score

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
    
    # cross val score
    cv_score = cross_val_score(clf_pipeline, features_train, target_train, scoring="roc_auc")
    
    mean_cv_score = round(np.mean(cv_score), 2)
    
    print(f'Cross-validation scores: {cv_score}')
    print(f'Mean cross-validation score: {mean_cv_score}')


    return fitted_model
