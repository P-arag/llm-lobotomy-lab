"""
Standalone script: records layer-13 residual-stream activations of Llama-2-7b-chat
on the sycophancy A/B dataset, then plots PCA (2 components) and UMAP (2 and 3
components) of sycophantic vs non-sycophantic activations.

Run from the repo root with the `fl` conda env:
    python plot_sycophancy_pca_umap.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from dotenv import load_dotenv
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

load_dotenv()

MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
SYCOPHANCY_DATA_PATH = os.path.join("datasets", "sycophancy", "refusal", "generate_dataset.json")
LAYER = 10
N_PER_CLASS = 200  # number of sycophantic and non-sycophantic examples to collect
POSITION = -2  # second-to-last token, same convention as generate_vectors.py

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("Set HF_TOKEN in .env (needs Llama-2 access on Hugging Face).")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Each dataset item is one question with two answers: the sycophantic one
# (answer_matching_behavior) and the non-sycophantic one. Both are run through
# the model as a completed chat prompt, and we keep the layer-k activation.
with open(SYCOPHANCY_DATA_PATH) as f:
    sycophancy_data = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
# 4-bit NF4 quantization: a full-precision 7B (13.5 GB) does not fit this 8 GB
# GPU, and loading to CPU RAM first segfaults the safetensors loader under this
# machine's memory pressure. 4-bit loads shard-by-shard straight onto the GPU.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, token=HF_TOKEN, quantization_config=bnb_config, device_map="auto"
)
model.eval()

captured = []


def hook(module, inputs, output):
    # transformers v5 may hand the hook either the hidden-states tensor or a
    # tuple whose first element is that tensor.
    hidden = output[0] if isinstance(output, tuple) else output
    captured.append(hidden[0, POSITION, :].detach().float().cpu().numpy())


model.model.layers[LAYER].register_forward_hook(hook)


def activation_for(question, answer):
    prompt = f"[INST] {question.strip()} [/INST] {answer.strip()}"
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
    captured.clear()
    with torch.no_grad():
        model(tokens)
    return captured[0]


pos_acts, neg_acts = [], []
for item in sycophancy_data:
    pos_acts.append(activation_for(item["question"], item["answer_matching_behavior"]))
    neg_acts.append(activation_for(item["question"], item["answer_not_matching_behavior"]))
    print(f"collected {len(pos_acts)} pairs")
    if len(pos_acts) >= N_PER_CLASS and len(neg_acts) >= N_PER_CLASS:
        break

pos_acts = np.stack(pos_acts)
neg_acts = np.stack(neg_acts)
acts = np.concatenate([pos_acts, neg_acts])
labels = np.array(["Sycophantic"] * len(pos_acts) + ["Non-sycophantic"] * len(neg_acts))
colors = labels == "Sycophantic"
print(f"activations: {acts.shape} (layer {LAYER})")

# PCA to 2 components, one scatter plot for both classes.
pca_2d = PCA(n_components=2).fit_transform(acts)
plt.figure(figsize=(7, 6))
plt.scatter(pca_2d[~colors, 0], pca_2d[~colors, 1], s=25, alpha=0.7, label="Non-sycophantic")
plt.scatter(pca_2d[colors, 0], pca_2d[colors, 1], s=25, alpha=0.7, label="Sycophantic")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.title(f"Layer {LAYER} activations, PCA")
plt.legend()
plt.savefig(f"layer{LAYER}__pca.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"saved layer{LAYER}_pca.png")

# UMAP to 2 and 3 components, one plot each.
umap_2d = umap.UMAP(n_components=2, random_state=42).fit_transform(acts)
plt.figure(figsize=(7, 6))
plt.scatter(umap_2d[~colors, 0], umap_2d[~colors, 1], s=25, alpha=0.7, label="Non-sycophantic")
plt.scatter(umap_2d[colors, 0], umap_2d[colors, 1], s=25, alpha=0.7, label="Sycophantic")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title(f"Layer {LAYER} activations, UMAP 2D")
plt.legend()
plt.savefig(f"layer{LAYER}_umap_2d.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"saved layer{LAYER}_umap_2d.png")

umap_3d = umap.UMAP(n_components=3, random_state=42).fit_transform(acts)
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(projection="3d")
ax.scatter(umap_3d[~colors, 0], umap_3d[~colors, 1], umap_3d[~colors, 2],
           s=25, alpha=0.7, label="Non-sycophantic")
ax.scatter(umap_3d[colors, 0], umap_3d[colors, 1], umap_3d[colors, 2],
           s=25, alpha=0.7, label="Sycophantic")
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.set_zlabel("UMAP 3")
ax.set_title(f"Layer {LAYER} activations, UMAP 3D")
ax.legend()
plt.savefig(f"layer{LAYER}_umap_3d.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"saved layer{LAYER}_umap_3d.png")
