"""
Records residual-stream activations of Llama-2-7b-chat on either the sycophancy
or refusal A/B dataset at a fixed layer and token position, computes the
difference-of-means (direction) vector, and dumps everything as .pt files for
later steering and plotting.

Run from the repo root:
    python backend/ml/record_activations.py --dataset sycophancy --layer 13
    python backend/ml/record_activations.py --dataset refusal    --layer 13

Outputs (in backend/ml/out/):
    {ds}_layer{L}_activations.pt   {"acts": [N,d] float32, "labels": [N] bool, "layer", "position"}
    {ds}_layer{L}_direction.pt     {"direction": [d] float32, "layer", "position"}

Where `ds` is either "sycophancy" or "refusal". `--steering` is an alias for
`--dataset`, making it convenient to select the direction used for steering.
"""

import argparse
import json
import os

import numpy as np
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

load_dotenv()

MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(HERE, "datasets")
OUT_DIR = os.path.join(HERE, "out")

# Supported A/B datasets; each item has `question`, `answer_matching_behavior`,
# `answer_not_matching_behavior`.
SUPPORTED_DATASETS = ("sycophancy", "refusal")

N_PER_CLASS = 1600  # number of matching / non-matching examples to collect
POSITION = -2      # second-to-last token

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("Set HF_TOKEN in .env (needs Llama-2 access on Hugging Face).")

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    # 4-bit NF4 quantization: same setup as plot_sycophancy_pca_umap.py (8 GB GPU).
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, token=HF_TOKEN, quantization_config=bnb_config, device_map="auto"
    )
    model.eval()
    return model, tokenizer


def dataset_path(dataset_name):
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"dataset must be one of {SUPPORTED_DATASETS}, got {dataset_name!r}")
    return os.path.join(DATASETS_DIR, f"{dataset_name}.json")


def record_activations(model, tokenizer, layer, data_path,
                       position=POSITION, n_per_class=N_PER_CLASS):
    """Hook layer `layer`, run every question twice (matching + non-matching
    answer), return (acts, labels): acts [N, d] float32, labels True for the
    matching behavior."""
    captured = []

    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured.append(hidden[0, position, :].detach().float().cpu())

    handle = model.model.layers[layer].register_forward_hook(hook)

    def activation_for(question, answer):
        prompt = f"[INST] {question.strip()} [/INST] {answer.strip()}"
        tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
        captured.clear()
        with torch.no_grad():
            model(tokens)
        return captured[0]

    with open(data_path) as f:
        data = json.load(f)

    pos_acts, neg_acts = [], []
    for item in tqdm(data, desc="Processing data"):
        pos_acts.append(activation_for(item["question"], item["answer_matching_behavior"]))
        neg_acts.append(activation_for(item["question"], item["answer_not_matching_behavior"]))
        if len(pos_acts) >= n_per_class and len(neg_acts) >= n_per_class:
            break

    handle.remove()
    acts = torch.cat([torch.stack(pos_acts), torch.stack(neg_acts)])  # [N, d]
    labels = torch.tensor([True] * len(pos_acts) + [False] * len(neg_acts))
    return acts, labels


def difference_of_means(acts, labels):
    """direction = mean(pos activations) - mean(neg activations), [d] float32."""
    pos = acts[labels]
    neg = acts[~labels]
    return pos.mean(dim=0) - neg.mean(dim=0)


def cosine_similarity(direction, acts):
    """Cosine similarity between a [d] direction and [N, d] or [T, d] activations.
    Returns a [N] (or [T]) float32 tensor of scores in [-1, 1]."""
    d = direction / direction.norm(dim=-1, keepdim=True)
    a = acts / acts.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return a @ d


def score_prompt(model, tokenizer, direction, layer, prompt, position=POSITION):
    """Score every token position of `prompt` by its cosine similarity to the
    direction vector at `layer`. Returns a list of (token_str, score)."""
    captured = []

    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured.append(hidden[0].detach().float().cpu())  # [T, d]

    handle = model.model.layers[layer].register_forward_hook(hook)
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
    captured.clear()
    with torch.no_grad():
        model(tokens)
    handle.remove()

    hs = captured[0]  # [T, d]
    scores = cosine_similarity(direction, hs)  # [T]
    return [(tokenizer.decode([t]), float(s)) for t, s in zip(tokens[0], scores)]


def main():
    ap = argparse.ArgumentParser()
    dataset_group = ap.add_mutually_exclusive_group()
    dataset_group.add_argument("--dataset", choices=SUPPORTED_DATASETS,
                    help="which A/B dataset to record activations for")
    dataset_group.add_argument("--steering", choices=SUPPORTED_DATASETS,
                    help="steering behavior to record (alias for --dataset)")
    ap.add_argument("--layer", type=int, default=13)
    args = ap.parse_args()

    dataset = args.dataset or args.steering
    if dataset is None:
        ap.error("one of --dataset or --steering is required")

    os.makedirs(OUT_DIR, exist_ok=True)
    model, tokenizer = load_model()

    data_path = dataset_path(dataset)
    acts, labels = record_activations(model, tokenizer, args.layer, data_path)
    print(f"activations: {tuple(acts.shape)} (dataset={dataset}, layer {args.layer})")

    direction = difference_of_means(acts, labels)
    print(f"direction norm: {direction.norm().item():.3f}")

    act_path = os.path.join(OUT_DIR, f"{dataset}_layer{args.layer}_activations.pt")
    dir_path = os.path.join(OUT_DIR, f"{dataset}_layer{args.layer}_direction.pt")
    torch.save({"acts": acts, "labels": labels, "layer": args.layer,
                "position": POSITION, "model": MODEL_ID, "dataset": dataset},
               act_path)
    torch.save({"direction": direction, "layer": args.layer,
                "position": POSITION, "model": MODEL_ID, "dataset": dataset},
               dir_path)
    print(f"saved {act_path}")
    print(f"saved {dir_path}")


if __name__ == "__main__":
    main()
