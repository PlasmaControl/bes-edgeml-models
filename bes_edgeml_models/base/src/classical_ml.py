"""Script to train the classical machine learning models like logistic regression,
random forests and XGBoost on the BES features data. Expects various features in
tabular data format. Not actively maintained, expect runtime errors.
"""
print(__doc__)
import os
import sys

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from xgboost import XGBClassifier

# plt.style.use("/home/lakshya/plt_custom.mplstyle")
sns.set_style("darkgrid")
blues = list(sns.dark_palette("#69d", 8, reverse=True).as_hex())
yellows = list(sns.color_palette("YlOrBr", 8).as_hex())

if __name__ == "__main__":
    lookahead = 200
    path = f"outputs/signal_window_16/label_look_ahead_{lookahead}/roc"
    plot_path = f"outputs/signal_window_16/label_look_ahead_{lookahead}/plots"
    # df = pd.read_csv(os.path.join(path, f"train_features_df_{lookahead}.csv"))
    df = pd.read_csv(os.path.join(path, f"cnn_feature_df_{lookahead}.csv"))
    print(df)
    print(df["label"].value_counts())

    train_df = df[df.loc[:, "elm_id"] < 380]
    valid_df = df[df.loc[:, "elm_id"] >= 380]
    print(train_df)
    print(valid_df)
    features = [
        col
        for col in df.columns
        if col
        not in [
            "elm_id",
            "valid_indices",
            "elm_event",
            "label",
            "automatic_label",
        ]
    ]

    label_type = "manual"
    label_dict = {"manual": "label", "automatic": "automatic_label"}
    X_train, y_train = train_df[features], train_df[label_dict[label_type]]
    X_valid, y_valid = valid_df[features], valid_df[label_dict[label_type]]
    # max_pool_features = [f for f in features if f.startswith("max")][::2]
    # avg_pool_features = [f for f in features if f.startswith("avg")][::2]

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
            f"roc_details_lr_{label_type}_lookahead_{lookahead}.csv",
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
            f"roc_details_rf_{label_type}_lookahead_{lookahead}.csv",
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
            f"roc_details_xgb_{label_type}_lookahead_{lookahead}.csv",
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
    valid_df.loc[:, "lr_pred"] = y_pred_lr[:, 1]
    valid_df.loc[:, "rf_pred"] = y_pred_rf[:, 1]
    valid_df.loc[:, "xgb_pred"] = y_pred_xgb[:, 1]
    print(valid_df)

    # plotting
    fig, ax = plt.subplots()
    feature_importances_rf.plot(kind="bar", figsize=(16, 8), ax=ax)
    ax.set_title(f"Feature importances using MDI, lookahead: {lookahead}")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.savefig(
        os.path.join(
            "outputs",
            f"rf_feature_importance_mdi_{label_type}_lookahead_{lookahead}.png",
        )
    )
    plt.show()

    # fig, ax = plt.subplots()
    # feature_importances_rf_fp.plot(kind="bar", figsize=(10, 8), ax=ax)
    # ax.set_title("Feature importances using feature permutation")
    # ax.set_ylabel("Mean accuracy decrease")
    # fig.tight_layout()
    # plt.show()

    fig, ax = plt.subplots()
    feature_importances_xgb.plot(kind="bar", figsize=(16, 8), ax=ax)
    ax.set_title(f"Feature importances, lookahead: {lookahead}")
    fig.tight_layout()
    plt.savefig(
        os.path.join(
            "outputs",
            f"xgb_feature_importance_{label_type}_lookahead_{lookahead}.png",
        )
    )
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
    ax.set_title(f"TPR/FPR vs threshold, lookahead: {lookahead}", fontsize=16)
    ax.set_xlabel("threshold", fontsize=12)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            "outputs",
            f"tpr_fpr_thresh_all_models_{label_type}_lookahead_{lookahead}.png",
        )
    )
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
    ax.set_title(f"ROC Curve, lookahead: {lookahead}", fontsize=16)
    ax.set_xlabel("FPR", fontsize=12)
    ax.set_ylabel("TPR", fontsize=12)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            "outputs",
            f"roc_curve_all_models_{label_type}_lookahead_{lookahead}.png",
        )
    )
    plt.show()

    for elm_id in range(300, 315):
        first_elm_max_pool = valid_df[valid_df["elm_event"] == elm_id].loc[
            :, max_pool_features + ["label", "lr_pred", "rf_pred", "xgb_pred"]
        ]
        first_elm_avg_pool = valid_df[valid_df["elm_event"] == elm_id].loc[
            :, avg_pool_features + ["label", "lr_pred", "rf_pred", "xgb_pred"]
        ]
        index = first_elm_max_pool.index.tolist()
        fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(12, 16))
        axs = axs.flatten()

        for idx, ax in enumerate(axs):
            if idx % 2 != 0:
                if idx == 1:
                    model_out_max_pool = first_elm_max_pool["lr_pred"]
                    label = "lr"
                elif idx == 3:
                    model_out_max_pool = first_elm_max_pool["rf_pred"]
                    label = "rf"
                else:
                    model_out_max_pool = first_elm_max_pool["xgb_pred"]
                    label = "xgb"

                for i in range(8):
                    ax.plot(
                        range(len(index)),
                        first_elm_max_pool[max_pool_features[i]],
                        c=blues[i],
                        label=max_pool_features[i],
                    )
                ax.plot(
                    range(len(index)),
                    first_elm_max_pool["label"],
                    c="crimson",
                    label="ground truth",
                )
                ax.plot(
                    range(len(index)),
                    model_out_max_pool,
                    c="forestgreen",
                    label=label,
                )
            else:
                if idx == 0:
                    model_out_avg_pool = first_elm_avg_pool["lr_pred"]
                    label = "lr"
                elif idx == 2:
                    model_out_avg_pool = first_elm_avg_pool["rf_pred"]
                    label = "rf"
                else:
                    model_out_avg_pool = first_elm_avg_pool["xgb_pred"]
                    label = "xgb"
                for i in range(8):
                    ax.plot(
                        range(len(index)),
                        first_elm_avg_pool[avg_pool_features[i]],
                        c=yellows[i],
                        label=avg_pool_features[i],
                    )
                ax.plot(
                    range(len(index)),
                    first_elm_avg_pool["label"],
                    c="crimson",
                    label="ground truth",
                )
                ax.plot(
                    range(len(index)),
                    model_out_avg_pool,
                    c="forestgreen",
                    label=label,
                )
            ax.legend(frameon=False, fontsize=9, ncol=2)
        plt.suptitle(f"ELM id: {elm_id}, label type: {label_type}, lookahead: {lookahead}", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95], pad=1.5, h_pad=1.5)
        fname = os.path.join(
            plot_path,
            f"classical_ml_model_out_ts_elm_id_{elm_id}_{label_type}_lookahead_{lookahead}.png",
        )
        print(f"Creating file: {fname}")
        plt.savefig(fname, dpi=150)
        if elm_id == 300:
            plt.show()
        plt.close()
