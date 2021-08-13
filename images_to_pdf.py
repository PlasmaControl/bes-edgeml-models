import os

from PIL import Image

if __name__ == "__main__":
    root_path = "outputs"
    img1 = Image.open(
        os.path.join(
            root_path, "time_gradients_different_hop_lengths_sws_128.png"
        )
    )
    img2 = Image.open(
        os.path.join(root_path, "time_gradients_all_channels_hop_4_sws_128.png")
    )
    img3 = Image.open(
        os.path.join(root_path, "time_gradients_diff_hop_4_sws_128.png")
    )

    img1 = img1.convert("RGB")
    img2 = img2.convert("RGB")
    img3 = img3.convert("RGB")

    pdf_fname = os.path.join(root_path, "time_derivatives_all_channels.pdf")

    img1.save(
        pdf_fname,
        "PDF",
        resolution=200.0,
        save_all=True,
        append_images=[img2, img3],
    )
