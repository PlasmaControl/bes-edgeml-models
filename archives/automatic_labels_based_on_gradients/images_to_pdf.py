import os
import re
import sys

sys.path.append(os.getcwd())

from PIL import Image


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r"(\d+)", text)]


if __name__ == "__main__":
    root_path = "automatic_label_plots"
    files = os.listdir(root_path)
    files = sorted(files, key=natural_keys)
    images = []
    for i, f in enumerate(files):
        print(f"Opening file: {f}")
        img = Image.open(os.path.join(root_path, f))
        img = img.convert("RGB")
        if i == 0:
            first_img = img
        else:
            images.append(img)
    pdf_fname = os.path.join(root_path, "automatic_label_plots.pdf")
    first_img.save(
        pdf_fname,
        "PDF",
        resolution=1000.0,
        save_all=True,
        append_images=images,
    )

    # img1 = Image.open(os.path.join(root_path, "lr_diff_lookaheads.png"))
    # img2 = Image.open(os.path.join(root_path, "rf_diff_lookaheads.png"))
    # img3 = Image.open(os.path.join(root_path, "xgb_diff_lookaheads.png"))
    # img4 = Image.open(os.path.join(root_path, "fpr_tpr_thresh_lookahead_0.png"))
    # img5 = Image.open(
    #     os.path.join(root_path, "fpr_tpr_thresh_lookahead_50.png")
    # )
    # img6 = Image.open(
    #     os.path.join(root_path, "fpr_tpr_thresh_lookahead_100.png")
    # )
    # img7 = Image.open(
    #     os.path.join(root_path, "roc_curve_lr_rf_xgb_lookahead_0.png")
    # )
    # img8 = Image.open(
    #     os.path.join(root_path, "roc_curve_lr_rf_xgb_lookahead_50.png")
    # )
    # img9 = Image.open(
    #     os.path.join(root_path, "roc_curve_lr_rf_xgb_lookahead_100.png")
    # )

    # img1 = img1.convert("RGB")
    # img2 = img2.convert("RGB")
    # img3 = img3.convert("RGB")
    # img4 = img4.convert("RGB")
    # img5 = img5.convert("RGB")
    # img6 = img6.convert("RGB")
    # img7 = img7.convert("RGB")
    # img8 = img8.convert("RGB")
    # img9 = img9.convert("RGB")

    # img_list = [
    #     img2,
    #     img3,
    #     img4,
    #     img5,
    #     img6,
    #     img7,
    #     img8,
    #     img9,
    # ]
    # pdf_fname = os.path.join(root_path, "classical_ml_models.pdf")

    # img1.save(
    #     pdf_fname,
    #     "PDF",
    #     resolution=400.0,
    #     save_all=True,
    #     append_images=img_list,
    # )
