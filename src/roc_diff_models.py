import os
import sys

import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
plt.style.use("/home/lakshya/plt_custom.mplstyle")

if __name__ == "__main__":
    path_0 = "outputs/signal_window_16/label_look_ahead_0/roc"
    path_50 = "outputs/signal_window_16/label_look_ahead_50/roc"
    path_100 = "outputs/signal_window_16/label_look_ahead_100/roc"
    path_200 = "outputs/signal_window_16/label_look_ahead_200/roc"
    path_400 = "outputs/signal_window_16/label_look_ahead_400/roc"

    # lr
    df_lr_0 = pd.read_csv(
        os.path.join(path_0, "roc_details_lr_lookahead_0.csv")
    )
    df_lr_50 = pd.read_csv(
        os.path.join(path_50, "roc_details_lr_lookahead_50.csv")
    )
    df_lr_100 = pd.read_csv(
        os.path.join(path_100, "roc_details_lr_lookahead_100.csv")
    )
    df_lr_200 = pd.read_csv(
        os.path.join(path_200, "roc_details_lr_lookahead_200.csv")
    )
    df_lr_400 = pd.read_csv(
        os.path.join(path_400, "roc_details_lr_lookahead_400.csv")
    )

    # rf
    df_rf_0 = pd.read_csv(
        os.path.join(path_0, "roc_details_rf_lookahead_0.csv")
    )
    df_rf_50 = pd.read_csv(
        os.path.join(path_50, "roc_details_rf_lookahead_50.csv")
    )
    df_rf_100 = pd.read_csv(
        os.path.join(path_100, "roc_details_rf_lookahead_100.csv")
    )
    df_rf_200 = pd.read_csv(
        os.path.join(path_200, "roc_details_rf_lookahead_200.csv")
    )
    df_rf_400 = pd.read_csv(
        os.path.join(path_400, "roc_details_rf_lookahead_400.csv")
    )

    # xgb
    df_xgb_0 = pd.read_csv(
        os.path.join(path_0, "roc_details_xgb_lookahead_0.csv")
    )
    df_xgb_50 = pd.read_csv(
        os.path.join(path_50, "roc_details_xgb_lookahead_50.csv")
    )
    df_xgb_100 = pd.read_csv(
        os.path.join(path_100, "roc_details_xgb_lookahead_100.csv")
    )
    df_xgb_200 = pd.read_csv(
        os.path.join(path_200, "roc_details_xgb_lookahead_200.csv")
    )
    df_xgb_400 = pd.read_csv(
        os.path.join(path_400, "roc_details_xgb_lookahead_400.csv")
    )

    # plotting lr
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        df_lr_0["fpr"],
        df_lr_0["tpr"],
        "-",
        lw=2,
        # c="#636EFA",
        label="lookahead: 0",
    )
    ax.plot(
        df_lr_50["fpr"],
        df_lr_50["tpr"],
        "-",
        lw=2,
        # c="#EF553B",
        label="lookahead: 50",
    )
    ax.plot(
        df_lr_100["fpr"],
        df_lr_100["tpr"],
        "-",
        lw=2,
        # c="#00CC96",
        label="lookahead: 100",
    )
    ax.plot(
        df_lr_200["fpr"],
        df_lr_200["tpr"],
        "-",
        lw=2,
        # c="#00CC96",
        label="lookahead: 200",
    )
    ax.plot(
        df_lr_400["fpr"],
        df_lr_400["tpr"],
        "-",
        lw=2,
        # c="#00CC96",
        label="lookahead: 400",
    )
    ax.plot([0, 1], [0, 1], c="gray", lw=2)
    ax.legend(fontsize=8, frameon=False)
    ax.set_xlabel("FPR", fontsize=12)
    ax.set_ylabel("TPR", fontsize=12)
    ax.set_title("Logistic Regression", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "lr_diff_lookaheads.png"))
    plt.show()

    # plotting rf
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        df_rf_0["fpr"],
        df_rf_0["tpr"],
        "-",
        lw=2,
        # c="#636EFA",
        label="lookahead: 0",
    )
    ax.plot(
        df_rf_50["fpr"],
        df_rf_50["tpr"],
        "-",
        lw=2,
        # c="#EF553B",
        label="lookahead: 50",
    )
    ax.plot(
        df_rf_100["fpr"],
        df_rf_100["tpr"],
        "-",
        lw=2,
        # c="#00CC96",
        label="lookahead: 100",
    )
    ax.plot(
        df_rf_200["fpr"],
        df_rf_200["tpr"],
        "-",
        lw=2,
        # c="#00CC96",
        label="lookahead: 200",
    )
    ax.plot(
        df_rf_400["fpr"],
        df_rf_400["tpr"],
        "-",
        lw=2,
        # c="#00CC96",
        label="lookahead: 400",
    )
    ax.plot([0, 1], [0, 1], c="gray", lw=2)
    ax.legend(fontsize=8, frameon=False)
    ax.set_xlabel("FPR", fontsize=12)
    ax.set_ylabel("TPR", fontsize=12)
    ax.set_title("Random Forest", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "rf_diff_lookaheads.png"))
    plt.show()

    # plotting xgb
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        df_xgb_0["fpr"],
        df_xgb_0["tpr"],
        "-",
        lw=2,
        # c="#636EFA",
        label="lookahead: 0",
    )
    ax.plot(
        df_xgb_50["fpr"],
        df_xgb_50["tpr"],
        "-",
        lw=2,
        # c="#EF553B",
        label="lookahead: 50",
    )
    ax.plot(
        df_xgb_100["fpr"],
        df_xgb_100["tpr"],
        "-",
        lw=2,
        # c="#00CC96",
        label="lookahead: 100",
    )
    ax.plot(
        df_xgb_200["fpr"],
        df_xgb_200["tpr"],
        "-",
        lw=2,
        # c="#00CC96",
        label="lookahead: 200",
    )
    ax.plot(
        df_xgb_400["fpr"],
        df_xgb_400["tpr"],
        "-",
        lw=2,
        # c="#00CC96",
        label="lookahead: 400",
    )
    ax.plot([0, 1], [0, 1], c="gray", lw=2)
    ax.legend(fontsize=8, frameon=False)
    ax.set_xlabel("FPR", fontsize=12)
    ax.set_ylabel("TPR", fontsize=12)
    ax.set_title("XGBoost", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "xgb_diff_lookaheads.png"))
    plt.show()
