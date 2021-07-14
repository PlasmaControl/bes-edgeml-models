import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("white")
colors = ["#ef476f", "#fcbf49", "#06d6a0", "#118ab2", "#073b4c"]

look_aheads = [0, 5, 10, 20, 40, 60, 80, 100, 120]
cnn_2d = {
    "look_ahead_0": {
        "scaled_pos": 0.604,
        "precision": 0.916,
        "recall": 0.902,
        "f1": 0.910,
    },
    "look_ahead_5": {
        "scaled_pos": 0.604,
        "precision": 0.920,
        "recall": 0.900,
        "f1": 0.910,
    },
    "look_ahead_10": {
        "scaled_pos": 0.602,
        "precision": 0.923,
        "recall": 0.886,
        "f1": 0.904,
    },
    "look_ahead_20": {
        "scaled_pos": 0.601,
        "precision": 0.920,
        "recall": 0.890,
        "f1": 0.904,
    },
    "look_ahead_40": {
        "scaled_pos": 0.600,
        "precision": 0.923,
        "recall": 0.880,
        "f1": 0.900,
    },
    "look_ahead_60": {
        "scaled_pos": 0.581,
        "precision": 0.910,
        "recall": 0.845,
        "f1": 0.874,
    },
    "look_ahead_80": {
        "scaled_pos": 0.582,
        "precision": 0.915,
        "recall": 0.840,
        "f1": 0.873,
    },
    "look_ahead_100": {
        "scaled_pos": 0.570,
        "precision": 0.902,
        "recall": 0.820,
        "f1": 0.856,
    },
    "look_ahead_120": {
        "scaled_pos": 0.562,
        "precision": 0.896,
        "recall": 0.800,
        "f1": 0.840,
    },
}
cnn_v2 = {
    "look_ahead_0": {
        "scaled_pos": 0.597,
        "precision": 0.896,
        "recall": 0.907,
        "f1": 0.901,
    },
    "look_ahead_5": {
        "scaled_pos": 0.603,
        "precision": 0.902,
        "recall": 0.913,
        "f1": 0.908,
    },
    "look_ahead_10": {
        "scaled_pos": 0.594,
        "precision": 0.910,
        "recall": 0.886,
        "f1": 0.897,
    },
    "look_ahead_20": {
        "scaled_pos": 0.596,
        "precision": 0.907,
        "recall": 0.894,
        "f1": 0.900,
    },
    "look_ahead_40": {
        "scaled_pos": 0.584,
        "precision": 0.901,
        "recall": 0.870,
        "f1": 0.885,
    },
    "look_ahead_60": {
        "scaled_pos": 0.582,
        "precision": 0.901,
        "recall": 0.866,
        "f1": 0.882,
    },
    "look_ahead_80": {
        "scaled_pos": 0.568,
        "precision": 0.894,
        "recall": 0.830,
        "f1": 0.860,
    },
    "look_ahead_100": {
        "scaled_pos": 0.567,
        "precision": 0.886,
        "recall": 0.836,
        "f1": 0.859,
    },
    "look_ahead_120": {
        "scaled_pos": 0.568,
        "precision": 0.904,
        "recall": 0.811,
        "f1": 0.850,
    },
}
cnn = {
    "look_ahead_0": {
        "scaled_pos": 0.596,
        "precision": 0.900,
        "recall": 0.901,
        "f1": 0.901,
    },
    "look_ahead_5": {
        "scaled_pos": 0.596,
        "precision": 0.913,
        "recall": 0.887,
        "f1": 0.900,
    },
    "look_ahead_10": {
        "scaled_pos": 0.604,
        "precision": 0.914,
        "recall": 0.903,
        "f1": 0.908,
    },
    "look_ahead_20": {
        "scaled_pos": 0.598,
        "precision": 0.911,
        "recall": 0.892,
        "f1": 0.902,
    },
    "look_ahead_40": {
        "scaled_pos": 0.588,
        "precision": 0.906,
        "recall": 0.873,
        "f1": 0.889,
    },
    "look_ahead_60": {
        "scaled_pos": 0.571,
        "precision": 0.898,
        "recall": 0.830,
        "f1": 0.861,
    },
    "look_ahead_80": {
        "scaled_pos": 0.569,
        "precision": 0.893,
        "recall": 0.833,
        "f1": 0.860,
    },
    "look_ahead_100": {
        "scaled_pos": 0.573,
        "precision": 0.911,
        "recall": 0.816,
        "f1": 0.856,
    },
    "look_ahead_120": {
        "scaled_pos": 0.564,
        "precision": 0.902,
        "recall": 0.797,
        "f1": 0.840,
    },
}
feature = {
    "look_ahead_0": {
        "scaled_pos": 0.592,
        "precision": 0.907,
        "recall": 0.884,
        "f1": 0.895,
    },
    "look_ahead_5": {
        "scaled_pos": 0.591,
        "precision": 0.904,
        "recall": 0.885,
        "f1": 0.894,
    },
    "look_ahead_10": {
        "scaled_pos": 0.590,
        "precision": 0.899,
        "recall": 0.890,
        "f1": 0.894,
    },
    "look_ahead_20": {
        "scaled_pos": 0.592,
        "precision": 0.908,
        "recall": 0.881,
        "f1": 0.894,
    },
    "look_ahead_40": {
        "scaled_pos": 0.589,
        "precision": 0.908,
        "recall": 0.876,
        "f1": 0.891,
    },
    "look_ahead_60": {
        "scaled_pos": 0.584,
        "precision": 0.903,
        "recall": 0.866,
        "f1": 0.883,
    },
    "look_ahead_80": {
        "scaled_pos": 0.579,
        "precision": 0.901,
        "recall": 0.854,
        "f1": 0.876,
    },
    "look_ahead_100": {
        "scaled_pos": 0.573,
        "precision": 0.895,
        "recall": 0.843,
        "f1": 0.867,
    },
    "look_ahead_120": {
        "scaled_pos": 0.571,
        "precision": 0.900,
        "recall": 0.830,
        "f1": 0.860,
    },
}


def get_lists(d: dict):
    scaled_pos = []
    precision = []
    recall = []
    f1 = []
    for key, val in d.items():
        for k, v in val.items():
            if k == "scaled_pos":
                scaled_pos.append(v)
            elif k == "precision":
                precision.append(v)
            elif k == "recall":
                recall.append(v)
            elif k == "f1":
                f1.append(v)
    return scaled_pos, precision, recall, f1


if __name__ == "__main__":
    cnn_2d_scaled_pos, cnn_2d_precision, cnn_2d_recall, cnn_2d_f1 = get_lists(
        cnn_2d
    )
    cnn_v2_scaled_pos, cnn_v2_precision, cnn_v2_recall, cnn_v2_f1 = get_lists(
        cnn_v2
    )
    cnn_scaled_pos, cnn_precision, cnn_recall, cnn_f1 = get_lists(cnn)
    (
        feature_scaled_pos,
        feature_precision,
        feature_recall,
        feature_f1,
    ) = get_lists(feature)

    fig = plt.figure(figsize=(8, 6))
    plt.plot(look_aheads, cnn_2d_scaled_pos, color=colors[0], label="cnn_2d")
    plt.show()
