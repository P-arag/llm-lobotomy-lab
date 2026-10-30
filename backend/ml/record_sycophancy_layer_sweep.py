"""
Record refusal activations for layers 8 through 17 and plot each layer's
activations in its own two-dimensional PCA projection.

The default sample contains 200 activation points total: 100 sycophantic
answers and 100 non-sycophantic answers. The model is loaded once and all
layer hooks are attached at the same time.

Run from the repository root:

    uv run python backend/ml/record_refusal_layer_sweep.py

Outputs are written to backend/ml/out/layer_sweep_200/ by default:

    refusal_layer{L}_activations.pt
    refusal_layer{L}_pca2.png
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm

from record_activations import (
    MODEL_ID,
    POSITION,
    dataset_path,
    load_model,
    device,
)


FIRST_LAYER = 8
LAST_LAYER = 17
DEFAULT_SAMPLE_SIZE = 200
RANDOM_SEED = 42


def sample_dataset(data_path: str, sample_size: int, seed: int) -> list[dict]:
    """Return a deterministic, balanced sample of A/B dataset records."""
    with open(data_path, encoding="utf-8") as data_file:
        data = json.load(data_file)

    if sample_size < 2:
        raise ValueError("sample_size must be at least 2")

    positive_count = (sample_size + 1) // 2
    negative_count = sample_size // 2
    if len(data) < max(positive_count, negative_count):
        raise ValueError(
            f"dataset has {len(data)} records, but {max(positive_count, negative_count)} "
            "records are required for the requested sample"
        )

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(data), size=max(positive_count, negative_count), replace=False)
    return [
        {
            "question": data[index]["question"],
            "answer": data[index]["answer_matching_behavior"],
            "label": True,
        }
        for index in indices[:positive_count]
    ] + [
        {
            "question": data[index]["question"],
            "answer": data[index]["answer_not_matching_behavior"],
            "label": False,
        }
        for index in indices[:negative_count]
    ]


def record_layer_activations(
    model,
    tokenizer,
    records: list[dict],
    layers: range,
    position: int = POSITION,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Capture one residual-stream vector for every requested layer and record."""
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer: int):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer] = hidden[0, position, :].detach().float().cpu()

        return hook

    try:
        for layer in layers:
            handle = model.model.layers[layer].register_forward_hook(make_hook(layer))
            handles.append(handle)

        activations = {layer: [] for layer in layers}
        labels = []
        for record in tqdm(records, desc="Recording activations"):
            prompt = f"[INST] {record['question'].strip()} [/INST] {record['answer'].strip()}"
            tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
            captured.clear()
            with torch.inference_mode():
                model(tokens)

            missing = set(layers) - captured.keys()
            if missing:
                raise RuntimeError(
                    f"model forward did not capture layers: {sorted(missing)}"
                )
            for layer in layers:
                activations[layer].append(captured[layer])
            labels.append(record["label"])
    finally:
        for handle in handles:
            handle.remove()

    return {
        layer: (torch.stack(activations[layer]), torch.tensor(labels, dtype=torch.bool))
        for layer in layers
    }


def save_layer_outputs(
    layer_data: dict[int, tuple[torch.Tensor, torch.Tensor]],
    output_dir: Path,
    position: int,
    sample_size: int,
) -> None:
    """Save activation tensors and one PCA-2 plot for each layer."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for layer, (activations, labels) in layer_data.items():
        pca_points = PCA(n_components=2).fit_transform(activations.numpy())
        tensor_path = output_dir / f"refusal_layer{layer}_activations.pt"
        plot_path = output_dir / f"refusal_layer{layer}_pca2.png"

        torch.save(
            {
                "acts": activations,
                "labels": labels,
                "layer": layer,
                "position": position,
                "model": MODEL_ID,
                "dataset": "refusal",
                "sample_size": sample_size,
            },
            tensor_path,
        )

        positive = labels.numpy()
        plt.figure(figsize=(7, 6))
        plt.scatter(
            pca_points[~positive, 0],
            pca_points[~positive, 1],
            s=28,
            alpha=0.75,
            label="Non-sycophantic",
        )
        plt.scatter(
            pca_points[positive, 0],
            pca_points[positive, 1],
            s=28,
            alpha=0.75,
            label="Sycophantic",
        )
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.title(f"Layer {layer} refusal activations, PCA 2D")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"saved {tensor_path}")
        print(f"saved {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record and plot refusal activations for layers 8 through 17."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="total number of activation points, split as evenly as possible",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="random seed used to choose dataset records",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="directory for activation files and PCA plots",
    )
    args = parser.parse_args()

    output_dir = args.out_dir or (
        Path(__file__).resolve().parent / "out" / f"layer_sweep_{args.samples}"
    )
    layers = range(FIRST_LAYER, LAST_LAYER + 1)
    records = sample_dataset(dataset_path("refusal"), args.samples, args.seed)
    print(
        f"recording {len(records)} points across layers "
        f"{FIRST_LAYER}-{LAST_LAYER}"
    )

    model, tokenizer = load_model()
    layer_data = record_layer_activations(model, tokenizer, records, layers)
    save_layer_outputs(layer_data, output_dir, POSITION, len(records))


if __name__ == "__main__":
    main()
