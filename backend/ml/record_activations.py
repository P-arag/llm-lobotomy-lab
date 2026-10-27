"""
Records residual-stream activations of Llama-2-7b-chat on the sycophancy A/B
dataset at a fixed layer and token position, computes the difference-of-means
(direction) vector, and dumps everything as .pt files for later steering and
plotting.

Run from the repo root with the `fl` conda env:
    python backend/ml/record_activations.py [--layer 13]

Outputs (in backend/ml/out/):
    layer{L}_activations.pt   {"acts": [N,d] float32, "labels": [N] bool, "layer", "position"}
    layer{L}_direction.pt     {"direction": [d] float32, "layer", "position"}
"""

import argparse
import json
import os

import numpy as np
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

load_dotenv()

MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
HERE = os.path.dirname(os.path.abspath(__file__))
SYCOPHANCY_DATA_PATH = os.path.join(HERE, "datasets", "sycophancy.json")
OUT_DIR = os.path.join(HERE, "out")

N_PER_CLASS = 200  # number of sycophantic and non-sycophantic examples to collect
POSITION = -2      # second-to-last token, same convention as plot_sycophancy_pca_umap.py

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


def record_activations(model, tokenizer, layer, data_path=SYCOPHANCY_DATA_PATH,
                       position=POSITION, n_per_class=N_PER_CLASS):
    """Hook layer `layer`, run every question twice (sycophantic + honest answer),
    return (acts, labels): acts [N, d] float32, labels True for sycophantic."""
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
    for item in data:
        pos_acts.append(activation_for(item["question"], item["answer_matching_behavior"]))
        neg_acts.append(activation_for(item["question"], item["answer_not_matching_behavior"]))
        print(f"collected {len(pos_acts)} pairs", flush=True)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=13)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    model, tokenizer = load_model()

    acts, labels = record_activations(model, tokenizer, args.layer)
    print(f"activations: {tuple(acts.shape)} (layer {args.layer})")

    direction = difference_of_means(acts, labels)
    print(f"direction norm: {direction.norm().item():.3f}")

    act_path = os.path.join(OUT_DIR, f"layer{args.layer}_activations.pt")
    dir_path = os.path.join(OUT_DIR, f"layer{args.layer}_direction.pt")
    torch.save({"acts": acts, "labels": labels, "layer": args.layer,
                "position": POSITION, "model": MODEL_ID}, act_path)
    torch.save({"direction": direction, "layer": args.layer,
                "position": POSITION, "model": MODEL_ID}, dir_path)
    print(f"saved {act_path}")
    print(f"saved {dir_path}")


if __name__ == "__main__":
    main()
