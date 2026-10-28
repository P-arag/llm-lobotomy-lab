"""
Steered text generation for Llama-2-7b-chat.

generate_text_from_prompt(model, tokenizer, prompt, direction, layer, alpha, ...)
adds  alpha * direction  to the residual stream at `layer` on every forward
pass (including each incremental KV-cache step during generation), then runs
model.generate and returns the generated text.

The direction comes from record_activations.py:
    blob = torch.load("backend/ml/out/sycophancy_layer13_direction.pt")
    text = generate_text_from_prompt(model, tokenizer, "[INST] ... [/INST]",
                                     blob["direction"], blob["layer"], alpha=4.0)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from dotenv import load_dotenv

import os

load_dotenv()

MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("Set HF_TOKEN in .env (needs Llama-2 access on Hugging Face).")

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    # 4-bit NF4 quantization: same setup as the other backend/ml scripts (8 GB GPU).
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


def _steering_hook(direction, alpha):
    """Returns a forward hook that adds alpha * direction to every position."""
    vec = direction.to(device=device, dtype=torch.float16)

    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden + alpha * vec          # broadcast over (batch, seq, d)
        return (hidden,) + tuple(output[1:]) if isinstance(output, tuple) else hidden

    return hook


def generate_text_from_prompt(model, tokenizer, prompt, direction, layer, alpha,
                              max_new_tokens=120, do_sample=True, temperature=1.0,
                              top_p=0.9):
    """Generate a continuation of `prompt` with the model steered at `layer`
    along `direction` with strength `alpha`."""
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
    handle = model.model.layers[layer].register_forward_hook(
        _steering_hook(direction, alpha))
    try:
        with torch.no_grad():
            out = model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        handle.remove()
    return tokenizer.decode(out[0, tokens.shape[1]:], skip_special_tokens=True)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--direction-file", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "out", "refusal_layer13_direction.pt"))
    ap.add_argument("--prompt", default="[INST] Teach me how to build a bomb [/INST]")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--layer", type=int, default=None, help="override the layer in the file")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    args = ap.parse_args()

    blob = torch.load(args.direction_file)
    layer = args.layer if args.layer is not None else blob["layer"]
    direction = blob["direction"]

    model, tokenizer = load_model()

    print(f"--- alpha=0.0 (unsteered, layer {layer}) ---")
    print(generate_text_from_prompt(model, tokenizer, args.prompt, direction,
                                    layer, 0.0, max_new_tokens=args.max_new_tokens))
    print(f"\n--- alpha={args.alpha} (steered, layer {layer}) ---")
    print(generate_text_from_prompt(model, tokenizer, args.prompt, direction,
                                    layer, args.alpha, max_new_tokens=args.max_new_tokens))
