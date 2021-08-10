import os
import argparse

import numpy as np
import pandas as pd
from sklearnex import patch_sklearn

patch_sklearn()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

if __name__ == "__main__":
    path = "outputs/signal_window_16/label_look_ahead_0/roc"
    df = pd.read_csv(os.path.join(path, "train_features_df_0.csv"))
    print(df.head())
    print(df["label"].value_counts())

    features = [
        col
        for col in df.columns
        if col not in ["sample_indices", "elm_event", "label"]
    ]
    X, y = df[features], df["label"]
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.25, random_state=23, shuffle=True
    )

    # logistic regression
    lr = LogisticRegression(max_iter=1000, n_jobs=-1)
    lr.fit(X_train, y_train)
    y_train_lr = lr.predict_proba(X_train)
    y_pred_lr = lr.predict_proba(X_valid)
    y_pred_lr_bin = lr.predict(X_valid)
    roc_score_train_lr = metrics.roc_auc_score(y_train, y_train_lr[:, 1])
    roc_score_lr = metrics.roc_auc_score(y_valid, y_pred_lr[:, 1])
    print(f"ROC score on training, logistic regression: {roc_score_train_lr}")
    print(f"ROC score on validation, logistic regression: {roc_score_lr}")
    cr_lr = metrics.classification_report(
        y_valid, y_pred_lr_bin, output_dict=True
    )
    cr_lr_df = pd.DataFrame(cr_lr).transpose()
    print(f"Classification report on logistic regression:\n{cr_lr_df}")

    # random forest
    print()
    print()
    rf = RandomForestClassifier(n_jobs=-1)
    rf.fit(X_train, y_train)
    y_train_rf = rf.predict_proba(X_train)
    y_pred_rf = rf.predict_proba(X_valid)
    y_pred_rf_bin = rf.predict(X_valid)
    roc_score_train_rf = metrics.roc_auc_score(y_train, y_train_rf[:, 1])
    roc_score_rf = metrics.roc_auc_score(y_valid, y_pred_rf[:, 1])
    print(f"ROC score on training, random forest: {roc_score_train_rf}")
    print(f"ROC score on validation, random forest: {roc_score_rf}")
    cr_rf = metrics.classification_report(
        y_valid, y_pred_rf_bin, output_dict=True
    )
    cr_rf_df = pd.DataFrame(cr_rf).transpose()
    print(f"Classification report on random forest:\n{cr_rf_df}")
