"""
Plots the recorded activations: a rotating looping 3D UMAP GIF and a PCA (2
component) scatter. Loads the .pt dumps written by record_activations.py,
reusing the plotting approach from plot_sycophancy_pca_umap.py.

Run from the repo root with the `fl` conda env:
    python backend/ml/plot_activations.py [--activations-file backend/ml/out/layer13_activations.pt]

Outputs (next to the .pt file):
    layer{L}_umap3d.gif   rotating looping 3D UMAP
    layer{L}_pca2d.png    PC1 vs PC2 scatter
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.decomposition import PCA

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "out")

GREEN = "#2ecc71"
RED = "#e74c3c"
BLUE = "#3498db"

# (positive label, negative label) per dataset; matches record_activations.py.
DATASET_LABELS = {
    "sycophancy": ("Sycophantic", "Non-sycophantic"),
    "refusal": ("Refusal", "Non-refusal"),
}
DATASET_COLORS = {
    "sycophancy": (GREEN, RED),
    "refusal": (BLUE, RED),
}


def load(activations_file):
    blob = torch.load(activations_file)
    acts = blob["acts"].numpy()
    labels = blob["labels"].numpy()
    layer = blob["layer"]
    return acts, labels, layer


def plot_pca_2d(acts, labels, layer, dataset, out_path):
    """PC1 vs PC2 scatter of both classes (same style as plot_sycophancy_pca_umap.py)."""
    pos_name, neg_name = DATASET_LABELS[dataset]
    pos_color, neg_color = DATASET_COLORS[dataset]
    pca_2d = PCA(n_components=2).fit_transform(acts)
    plt.figure(figsize=(7, 6))
    for lab, name, color in [(True, pos_name, pos_color), (False, neg_name, neg_color)]:
        sel = labels == lab
        plt.scatter(pca_2d[sel, 0], pca_2d[sel, 1], c=color, s=18, alpha=0.7, label=name)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"Residual-stream PCA ({dataset}, layer {layer})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()
    print(f"saved {out_path}")


def plot_umap3d_gif(acts, labels, layer, dataset, out_path, frames=90, fps=20):
    """3D UMAP scatter, saved as an infinitely looping rotating GIF. The camera
    spins a full 360 degrees across `frames`, then plays the reverse pass so
    the loop is seamless (ping-pong, loop=0)."""
    pos_name, neg_name = DATASET_LABELS[dataset]
    pos_color, neg_color = DATASET_COLORS[dataset]
    emb = umap.UMAP(n_components=3).fit_transform(acts)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    for lab, name, color in [(True, pos_name, pos_color), (False, neg_name, neg_color)]:
        sel = labels == lab
        ax.scatter(emb[sel, 0], emb[sel, 1], emb[sel, 2], c=color, s=18, alpha=0.7, label=name)
    ax.set_title(f"UMAP of residual-stream activations ({dataset}, layer {layer})")
    ax.legend(loc="upper left")
    ax.view_init(elev=25, azim=0)

    def update(i):
        # ping-pong: forward 0..frames-1, then backward 1..frames-1 (endpoints shared)
        azim = (i if i < frames else 2 * frames - 2 - i) * 360 / frames
        ax.view_init(elev=25, azim=azim)
        return []

    anim = FuncAnimation(fig, update, frames=2 * frames - 2, interval=1000 // fps)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sycophancy", choices=["sycophancy", "refusal"])
    ap.add_argument("--layer", type=int, default=13)
    ap.add_argument("--activations-file", default=None,
                    help="override; defaults to out/{dataset}_layer{layer}_activations.pt")
    args = ap.parse_args()

    activations_file = args.activations_file or os.path.join(
        OUT_DIR, f"{args.dataset}_layer{args.layer}_activations.pt")
    acts, labels, layer = load(activations_file)
    print(f"activations: {acts.shape} (dataset={args.dataset}, layer {layer})")

    base = os.path.splitext(activations_file)[0]
    plot_pca_2d(acts, labels, layer, args.dataset, base + "_pca2d.png")
    plot_umap3d_gif(acts, labels, layer, args.dataset, base + "_umap3d.gif")


if __name__ == "__main__":
    main()
