import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_class_legend(tsv_path, ncols=5, save_path=None):
    df = pd.read_csv(tsv_path, sep="\t", header=None, names=["class", "color"])

    # Alphebetize them but put Non-leukemic last
    df = df.sort_values(by="class", key=lambda x: x.str.lower())
    df = pd.concat([df[df["class"] != "Non-leukemic"], df[df["class"] == "Non-leukemic"]])

    patches = [
        mpatches.Patch(color=row["color"], label=row["class"])
        for _, row in df.iterrows()
    ]

    fig, ax = plt.subplots(figsize=(ncols * 3, len(df) / ncols * 0.5))
    ax.axis("off")
    ax.legend(
        handles=patches,
        ncols=ncols,
        loc="center",
        frameon=False,
        fontsize=10,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# Get current dir with Path and then go up one level to get the configs dir
colors_tsv = Path(__file__).resolve().parent.parent / "configs" / "class_colors.tsv"
output_dir = Path(__file__).resolve().parent.parent / \
    "reports" / "output" / "tsnes" / "class_legend_supplementary.svg"
plot_class_legend(colors_tsv, ncols=5, save_path=output_dir)
