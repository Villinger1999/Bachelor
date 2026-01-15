import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from skimage import util, io
from tensorflow.keras.datasets import cifar10
import numpy as np

class NoiseGenerator:
    def __init__(
        self,
        image_output_dir: str = "data/noise/",
        plot_output_dir: str = "data/noise/",
        load_cifar: bool = True,
    ):
        """
        Manages CIFAR-10 loading, noise generation and visualization.

        Args:
            image_output_dir: where to save noisy images
            plot_output_dir: where to save grid plots
            load_cifar: whether to load CIFAR-10 in __init__
        """
        self.image_output_dir = image_output_dir
        self.plot_output_dir = plot_output_dir

        if load_cifar:
            (x_train, y_train), _ = cifar10.load_data()
            self.x_train = x_train.astype("float32") / 255.0
            self.y_train = y_train
        else:
            self.x_train = None
            self.y_train = None

    @staticmethod
    def apply_torch_noise(var: float, orig_img: torch.Tensor) -> torch.Tensor:
        """
        Returns a noisy version of the input tensor with requires_grad=True.
        Accepts (C,H,W) or (1,C,H,W) and returns (1,C,H,W).
        """
        if orig_img.dim() == 3:
            img = orig_img.unsqueeze(0)
        else:
            img = orig_img

        noisy = img + torch.randn_like(img) * var
        noisy = noisy.clamp(0, 1)
        return noisy.requires_grad_(True)

    @staticmethod
    def get_unique_filename(directory: str, filename: str) -> str:
        """
        Ensures a filename does not overwrite an existing file.
        If file exists, appends _1, _2, _3...
        """
        base, ext = os.path.splitext(filename)
        counter = 1
        unique_name = filename

        while os.path.exists(os.path.join(directory, unique_name)):
            unique_name = f"{base}_{counter}{ext}"
            counter += 1

        return os.path.join(directory, unique_name)

    @staticmethod
    def show_noisy_grid(img_records, title_prefix="Image",
                         save_path=None, show=False):
        """
        img_records: list of dicts with keys ["Array", "Variance"]
        """
        fig, axes = plt.subplots(
            1, len(img_records),
            figsize=(3 * len(img_records), 3)
        )

        if len(img_records) == 1:
            axes = [axes]

        for ax, row in zip(axes, img_records):
            ax.imshow(row["Array"])
            ax.set_title(f"{title_prefix}\nvar={row['Variance']}")
            ax.axis("off")

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight")

        if show:
            plt.show()

        plt.close(fig)

    def generate_noisy_cifar_examples(
        self,
        indices=(0, 1, 2),
        variances=(0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5),
        save_images=False,
        save_plots=False,
        show_plots=False,
    ) -> pd.DataFrame:
        """
        Generate noisy versions of selected CIFAR-10 training images.

        Returns:
            df: DataFrame with columns:
                ["Image", "Index", "Variance", "Path", "Array"]
        """
        if self.x_train is None:
            raise RuntimeError(
                "CIFAR-10 not loaded. Set load_cifar=True in __init__ "
                "or load data manually."
            )

        results = []

        if save_images:
            os.makedirs(self.image_output_dir, exist_ok=True)

        for idx in indices:
            base_img = self.x_train[idx]  # shape (32, 32, 3)
            img_name = f"img{idx}"

            for var in variances:
                noisy = util.random_noise(base_img, mode="gaussian", var=var)

                # Filename 
                fname = f"{img_name}_var{var}.jpg"
                img_path = (
                    os.path.join(self.image_output_dir, fname)
                    if save_images else None
                )

                if save_images:
                    io.imsave(img_path, (noisy * 255).astype("uint8"))

                results.append({
                    "Image": img_name,
                    "Index": idx,
                    "Variance": var,
                    "Path": img_path,
                    "Array": noisy,
                })

        df = pd.DataFrame(results)

        # Make plots 
        if save_plots or show_plots:
            os.makedirs(self.plot_output_dir, exist_ok=True)

            for idx in indices:
                img_name = f"img{idx}"
                img_records = df[df["Image"] == img_name].to_dict("records")

                if save_plots:
                    base_name = f"range_noise_{img_name}.jpg"
                    plot_path = self._get_unique_filename(
                        self.plot_output_dir, base_name
                    )
                else:
                    plot_path = None

                self._show_noisy_grid(
                    img_records,
                    title_prefix=f"Image {idx}",
                    save_path=plot_path,
                    show=show_plots,
                )

        return df
