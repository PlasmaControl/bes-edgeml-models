import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt
from sklearnex import patch_sklearn

patch_sklearn()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier

plt.style.use("/home/lakshya/plt_custom.mplstyle")

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
    print()
    print()
    lr = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=23)
    lr.fit(X_train, y_train)
    y_train_lr = lr.predict_proba(X_train)
    y_pred_lr = lr.predict_proba(X_valid)
    y_pred_lr_bin = lr.predict(X_valid)
    roc_score_train_lr = metrics.roc_auc_score(y_train, y_train_lr[:, 1])
    roc_score_lr = metrics.roc_auc_score(y_valid, y_pred_lr[:, 1])
    print(f"ROC score on training, logistic regression: {roc_score_train_lr}")
    print(f"ROC score on validation, logistic regression: {roc_score_lr}")
    fpr, tpr, thresh = metrics.roc_curve(y_valid, y_pred_lr[:, 1])
    roc_details_lr = pd.DataFrame()
    roc_details_lr["fpr"] = fpr
    roc_details_lr["tpr"] = tpr
    roc_details_lr["threshold"] = thresh

    roc_details_lr.to_csv(
        os.path.join(
            path,
            f"roc_details_lr_lookahead_0.csv",
        ),
        index=False,
    )

    cr_lr = metrics.classification_report(
        y_valid, y_pred_lr_bin, output_dict=True
    )
    cr_lr_df = pd.DataFrame(cr_lr).transpose()
    print(f"Classification report on logistic regression:\n{cr_lr_df}")

    # random forest
    print()
    print()
    rf = RandomForestClassifier(n_jobs=-1, random_state=23)
    rf.fit(X_train, y_train)
    y_train_rf = rf.predict_proba(X_train)
    y_pred_rf = rf.predict_proba(X_valid)
    y_pred_rf_bin = rf.predict(X_valid)
    roc_score_train_rf = metrics.roc_auc_score(y_train, y_train_rf[:, 1])
    roc_score_rf = metrics.roc_auc_score(y_valid, y_pred_rf[:, 1])
    print(f"ROC score on training, random forest: {roc_score_train_rf}")
    print(f"ROC score on validation, random forest: {roc_score_rf}")

    fpr, tpr, thresh = metrics.roc_curve(y_valid, y_pred_rf[:, 1])
    roc_details_rf = pd.DataFrame()
    roc_details_rf["fpr"] = fpr
    roc_details_rf["tpr"] = tpr
    roc_details_rf["threshold"] = thresh

    roc_details_rf.to_csv(
        os.path.join(
            path,
            f"roc_details_rf_lookahead_0.csv",
        ),
        index=False,
    )

    cr_rf = metrics.classification_report(
        y_valid, y_pred_rf_bin, output_dict=True
    )
    cr_rf_df = pd.DataFrame(cr_rf).transpose()
    print(f"Classification report on random forest:\n{cr_rf_df}")
    importances_rf = rf.feature_importances_
    feature_importances_rf = pd.Series(importances_rf, index=features)

    # random forest feature importance using feature permutation
    # result = permutation_importance(
    #     rf, X_valid, y_valid, n_repeats=6, random_state=23, n_jobs=-1
    # )
    # feature_importances_rf_fp = pd.Series(
    #     result.importances_mean, index=features
    # )

    # XGBoost
    print()
    print()
    xgb = XGBClassifier(
        objective="binary:logistic",
        use_label_encoder=False,
        n_jobs=-1,
        random_state=23,
    )
    xgb.fit(X_train, y_train, eval_metric="logloss")
    y_train_xgb = xgb.predict_proba(X_train)
    y_pred_xgb = xgb.predict_proba(X_valid)
    y_pred_xgb_bin = xgb.predict(X_valid)
    roc_score_train_xgb = metrics.roc_auc_score(y_train, y_train_xgb[:, 1])
    roc_score_xgb = metrics.roc_auc_score(y_valid, y_pred_xgb[:, 1])
    print(f"ROC score on training, XGBoost: {roc_score_train_xgb}")
    print(f"ROC score on validation, XGBoost: {roc_score_xgb}")

    fpr, tpr, thresh = metrics.roc_curve(y_valid, y_pred_xgb[:, 1])
    roc_details_xgb = pd.DataFrame()
    roc_details_xgb["fpr"] = fpr
    roc_details_xgb["tpr"] = tpr
    roc_details_xgb["threshold"] = thresh

    roc_details_xgb.to_csv(
        os.path.join(
            path,
            f"roc_details_xgb_lookahead_0.csv",
        ),
        index=False,
    )

    cr_xgb = metrics.classification_report(
        y_valid, y_pred_xgb_bin, output_dict=True
    )
    cr_xgb_df = pd.DataFrame(cr_xgb).transpose()
    print(f"Classification report on XGBoost:\n{cr_xgb_df}")
    importances_xgb = xgb.feature_importances_
    feature_importances_xgb = pd.Series(importances_xgb, index=features)

    # plotting
    fig, ax = plt.subplots()
    feature_importances_rf.plot(kind="bar", figsize=(10, 8), ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.show()

    # fig, ax = plt.subplots()
    # feature_importances_rf_fp.plot(kind="bar", figsize=(10, 8), ax=ax)
    # ax.set_title("Feature importances using feature permutation")
    # ax.set_ylabel("Mean accuracy decrease")
    # fig.tight_layout()
    # plt.show()

    fig, ax = plt.subplots()
    feature_importances_xgb.plot(kind="bar", figsize=(10, 8), ax=ax)
    ax.set_title("Feature importances")
    fig.tight_layout()
    plt.show()

    # fpr/tpr vs threshold for lr
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        roc_details_lr["threshold"].loc[1:],
        roc_details_lr["fpr"].loc[1:],
        "--",
        lw=2,
        c="#636EFA",
        label="fpr lr",
    )
    ax.plot(
        roc_details_lr["threshold"].loc[1:],
        roc_details_lr["tpr"].loc[1:],
        "-",
        lw=2,
        c="#636EFA",
        label="tpr lr",
    )

    # fpr/tpr vs threshold for rf
    ax.plot(
        roc_details_rf["threshold"].loc[1:],
        roc_details_rf["fpr"].loc[1:],
        "--",
        lw=2,
        c="#EF553B",
        label="fpr rf",
    )
    ax.plot(
        roc_details_rf["threshold"].loc[1:],
        roc_details_rf["tpr"].loc[1:],
        "-",
        lw=2,
        c="#EF553B",
        label="tpr rf",
    )

    # fpr/tpr vs threshold for xgb
    ax.plot(
        roc_details_xgb["threshold"].loc[1:],
        roc_details_xgb["fpr"].loc[1:],
        "--",
        lw=2,
        c="#00CC96",
        label="fpr xgb",
    )
    ax.plot(
        roc_details_xgb["threshold"].loc[1:],
        roc_details_xgb["tpr"].loc[1:],
        "-",
        lw=2,
        c="#00CC96",
        label="tpr xgb",
    )
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("TPR/FPR vs threshold", fontsize=16)
    ax.set_xlabel("threshold", fontsize=12)
    plt.tight_layout()
    plt.show()

    # roc curves for different models
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        roc_details_lr["fpr"].loc[1:],
        roc_details_lr["tpr"].loc[1:],
        "-",
        lw=2,
        c="#636EFA",
        label="lr",
    )
    ax.plot(
        roc_details_rf["fpr"],
        roc_details_rf["tpr"],
        "-",
        lw=2,
        c="#EF553B",
        label="rf",
    )
    ax.plot(
        roc_details_xgb["fpr"],
        roc_details_xgb["tpr"],
        "-",
        lw=2,
        c="#00CC96",
        label="xgb",
    )
    ax.plot([0, 1], [0, 1], c="gray", lw=2)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("ROC Curve", fontsize=16)
    ax.set_xlabel("FPR", fontsize=12)
    ax.set_ylabel("TPR", fontsize=12)
    plt.tight_layout()
    plt.show()
