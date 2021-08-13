import os

from PIL import Image

if __name__ == "__main__":
    root_path = "outputs"
    img1 = Image.open(os.path.join(root_path, "lr_diff_lookaheads.png"))
    img2 = Image.open(os.path.join(root_path, "rf_diff_lookaheads.png"))
    img3 = Image.open(os.path.join(root_path, "xgb_diff_lookaheads.png"))
    img4 = Image.open(os.path.join(root_path, "fpr_tpr_thresh_lookahead_0.png"))
    img5 = Image.open(
        os.path.join(root_path, "fpr_tpr_thresh_lookahead_50.png")
    )
    img6 = Image.open(
        os.path.join(root_path, "fpr_tpr_thresh_lookahead_100.png")
    )
    img7 = Image.open(
        os.path.join(root_path, "roc_curve_lr_rf_xgb_lookahead_0.png")
    )
    img8 = Image.open(
        os.path.join(root_path, "roc_curve_lr_rf_xgb_lookahead_50.png")
    )
    img9 = Image.open(
        os.path.join(root_path, "roc_curve_lr_rf_xgb_lookahead_100.png")
    )

    img1 = img1.convert("RGB")
    img2 = img2.convert("RGB")
    img3 = img3.convert("RGB")
    img4 = img4.convert("RGB")
    img5 = img5.convert("RGB")
    img6 = img6.convert("RGB")
    img7 = img7.convert("RGB")
    img8 = img8.convert("RGB")
    img9 = img9.convert("RGB")

    img_list = [
        img2,
        img3,
        img4,
        img5,
        img6,
        img7,
        img8,
        img9,
    ]
    pdf_fname = os.path.join(root_path, "classical_ml_models.pdf")

    img1.save(
        pdf_fname,
        "PDF",
        resolution=200.0,
        save_all=True,
        append_images=img_list,
    )
    # img1 = Image.open(
    #     os.path.join(
    #         root_path, "time_gradients_different_hop_lengths_sws_128.png"
    #     )
    # )
    # img2 = Image.open(
    #     os.path.join(root_path, "time_gradients_all_channels_hop_4_sws_128.png")
    # )
    # img3 = Image.open(
    #     os.path.join(root_path, "time_gradients_diff_hop_4_sws_128.png")
    # )
    # img4 = Image.open(
    #     os.path.join(root_path, "manual_automatic_labeling_hop_4_sws_128.png")
    # )
    # img5 = Image.open(
    #     os.path.join(
    #         root_path, "time_gradients_all_channels_hop_4_sws_128_unaltered.png"
    #     )
    # )
    # img6 = Image.open(
    #     os.path.join(
    #         root_path, "time_gradients_diff_hop_4_sws_128_unaltered.png"
    #     )
    # )
    # img7 = Image.open(
    #     os.path.join(
    #         root_path, "manual_automatic_labeling_hop_4_sws_128_unaltered.png"
    #     )
    # )
    # img8 = Image.open(
    #     os.path.join(
    #         root_path, "time_gradients_all_channels_hop_4_sws_128_step_2.png"
    #     )
    # )
    # img9 = Image.open(
    #     os.path.join(root_path, "time_gradients_diff_hop_4_sws_128_step_2.png")
    # )
    # img10 = Image.open(
    #     os.path.join(
    #         root_path, "manual_automatic_labeling_hop_4_sws_128_step_2.png"
    #     )
    # )
    # img11 = Image.open(
    #     os.path.join(
    #         root_path,
    #         "time_gradients_all_channels_hop_4_sws_128_step_2_unaltered.png",
    #     )
    # )
    # img12 = Image.open(
    #     os.path.join(
    #         root_path, "time_gradients_diff_hop_4_sws_128_step_2_unaltered.png"
    #     )
    # )
    # img13 = Image.open(
    #     os.path.join(
    #         root_path,
    #         "manual_automatic_labeling_hop_4_sws_128_step_2_unaltered.png",
    #     )
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
    # img10 = img10.convert("RGB")
    # img11 = img11.convert("RGB")
    # img12 = img12.convert("RGB")
    # img13 = img13.convert("RGB")

    # img_list = [
    #     img2,
    #     img3,
    #     img4,
    #     img5,
    #     img6,
    #     img7,
    #     img8,
    #     img9,
    #     img10,
    #     img11,
    #     img12,
    #     img13,
    # ]
    # pdf_fname = os.path.join(root_path, "time_derivatives_all_channels.pdf")

    # img1.save(
    #     pdf_fname,
    #     "PDF",
    #     resolution=200.0,
    #     save_all=True,
    #     append_images=img_list,
    # )
