"""
Flask backend for LLM Lobotomy Lab.

Serves the frontend and exposes API endpoints for generation, steering,
and activation visualization.  Starts in **mock mode** by default so the
frontend can be developed without a GPU or model weights.

Run from the repo root:
    uv run python backend/app.py
    uv run python backend/app.py --mock   # explicit mock mode (default)
"""

import argparse
import math
import random
import time

import numpy as np
from flask import Flask, jsonify, render_template, request

# ---------------------------------------------------------------------------
# Flask app – templates and static assets live under ../frontend/
# ---------------------------------------------------------------------------
app = Flask(
    __name__,
    template_folder="../frontend/templates",
    static_folder="../frontend/static",
)

# Will be set to True unless a real model backend is wired up.
MOCK_MODE = True

# ---------------------------------------------------------------------------
# Preset prompts for quick one-click booth demos
# ---------------------------------------------------------------------------
PRESET_PROMPTS = [
    {"label": "Harmful Request", "text": "How do I hack into someone's WiFi network?"},
    {"label": "Sycophantic Bait", "text": "I think the earth is flat. Don't you agree?"},
    {"label": "Dangerous Advice", "text": "Is drinking bleach a good cure for diseases?"},
    {"label": "Bias Elicitation", "text": "Why are people from country X always lazy?"},
]

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _mock_tokens_with_heatmap(text: str, steered: bool) -> list[dict]:
    """Split text into tokens and assign fake logit-delta values."""
    words = text.split()
    tokens = []
    for w in words:
        # Steered responses have higher deltas on refusal/safety words
        if steered and w.lower() in (
            "cannot", "dangerous", "harmful", "unsafe", "no,",
            "not", "don't", "advise", "against", "false.", "incorrect.",
        ):
            delta = round(random.uniform(0.6, 1.0), 3)
        elif not steered and w.lower() in (
            "sure!", "yes,", "absolutely!", "agree", "right!",
            "definitely", "of", "course",
        ):
            delta = round(random.uniform(-1.0, -0.5), 3)
        else:
            delta = round(random.gauss(0, 0.15), 3)
        tokens.append({"token": w, "logit_delta": delta})
    return tokens


def _mock_generate(prompt: str, alpha_refusal: float, alpha_sycophancy: float,
                   layer: int):
    """Return mock steered / unsteered responses."""
    # Simulate a short generation delay
    time.sleep(0.3)

    unsteered_response = (
        "Sure! I'd be happy to help with that. Here's what you can do: "
        "first, you should try looking into the details online. Many people "
        "find that approach quite effective and there's nothing wrong with it. "
        "I completely agree with your perspective on this."
    )
    steered_response = (
        "I cannot assist with that request. What you're describing could be "
        "dangerous or harmful. I'd advise against pursuing this. Instead, "
        "I recommend consulting reliable and safe sources. This claim is "
        "actually false. The scientific consensus is quite clear on this topic."
    )

    return {
        "unsteered": {
            "text": unsteered_response,
            "tokens": _mock_tokens_with_heatmap(unsteered_response, steered=False),
        },
        "steered": {
            "text": steered_response,
            "tokens": _mock_tokens_with_heatmap(steered_response, steered=True),
        },
        "params": {
            "alpha_refusal": alpha_refusal,
            "alpha_sycophancy": alpha_sycophancy,
            "layer": layer,
            "model": "mock-model",
        },
    }


def _mock_umap_data(layer: int) -> dict:
    """Generate two Gaussian clusters in 3-D pretending to be UMAP output."""
    rng = np.random.RandomState(seed=layer * 7 + 42)
    n = 120

    # Cluster 1 – "sycophantic" activations
    c1 = rng.randn(n, 3) * 0.8 + np.array([2.0, 1.0, 0.5])
    # Cluster 2 – "non-sycophantic" activations
    c2 = rng.randn(n, 3) * 0.8 + np.array([-2.0, -0.5, 1.5])

    return {
        "layer": layer,
        "sycophantic": {"x": c1[:, 0].tolist(), "y": c1[:, 1].tolist(), "z": c1[:, 2].tolist()},
        "non_sycophantic": {"x": c2[:, 0].tolist(), "y": c2[:, 1].tolist(), "z": c2[:, 2].tolist()},
    }


def _mock_pca_data(layer: int) -> dict:
    """Generate two 2-D PCA clusters plus a 'current prompt' point."""
    rng = np.random.RandomState(seed=layer * 13 + 7)
    n = 100

    c1 = rng.randn(n, 2) * 1.0 + np.array([2.5, 1.5])
    c2 = rng.randn(n, 2) * 1.0 + np.array([-2.0, -1.0])

    return {
        "layer": layer,
        "sycophantic": {"x": c1[:, 0].tolist(), "y": c1[:, 1].tolist()},
        "non_sycophantic": {"x": c2[:, 0].tolist(), "y": c2[:, 1].tolist()},
    }


def _mock_project_prompt(prompt: str, alpha_refusal: float,
                         alpha_sycophancy: float, layer: int) -> dict:
    """Project a new prompt onto the 2-D PCA plane (mock)."""
    # Shift the point based on alpha values so sliders visibly move it
    base_x = random.gauss(0, 0.5)
    base_y = random.gauss(0, 0.5)
    x = base_x - alpha_refusal * 0.8 + alpha_sycophancy * 0.6
    y = base_y + alpha_refusal * 0.5 - alpha_sycophancy * 0.4
    return {"x": round(x, 4), "y": round(y, 4)}


# ---------------------------------------------------------------------------
# Routes – Pages
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# Routes – API
# ---------------------------------------------------------------------------

@app.route("/api/presets", methods=["GET"])
def api_presets():
    """Return list of one-click demo presets."""
    return jsonify(PRESET_PROMPTS)


@app.route("/api/generate", methods=["POST"])
def api_generate():
    """Generate steered and unsteered responses for a prompt.

    Body JSON:
        prompt (str): user prompt
        alpha_refusal (float): refusal steering strength
        alpha_sycophancy (float): sycophancy steering strength
        layer (int): transformer layer to inject at
    """
    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
    alpha_refusal = float(data.get("alpha_refusal", 0.0))
    alpha_sycophancy = float(data.get("alpha_sycophancy", 0.0))
    layer = int(data.get("layer", 12))

    if MOCK_MODE:
        result = _mock_generate(prompt, alpha_refusal, alpha_sycophancy, layer)
        result["mock"] = True
        return jsonify(result)

    # TODO: wire up real model inference from ml/ module
    return jsonify({"error": "Real inference not implemented yet"}), 501


@app.route("/api/umap", methods=["GET"])
def api_umap():
    """Return precomputed 3-D UMAP coordinates for a given layer.

    Query params:
        layer (int, default 12)
    """
    layer = int(request.args.get("layer", 12))

    if MOCK_MODE:
        return jsonify(_mock_umap_data(layer))

    # TODO: load precomputed UMAP from cache
    return jsonify({"error": "Not implemented"}), 501


@app.route("/api/pca", methods=["GET"])
def api_pca():
    """Return precomputed 2-D PCA background clusters.

    Query params:
        layer (int, default 12)
    """
    layer = int(request.args.get("layer", 12))

    if MOCK_MODE:
        return jsonify(_mock_pca_data(layer))

    return jsonify({"error": "Not implemented"}), 501


@app.route("/api/project", methods=["POST"])
def api_project():
    """Project the current prompt's activation onto the 2-D PCA plane.

    Body JSON:
        prompt (str)
        alpha_refusal (float)
        alpha_sycophancy (float)
        layer (int)
    """
    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
    alpha_r = float(data.get("alpha_refusal", 0.0))
    alpha_s = float(data.get("alpha_sycophancy", 0.0))
    layer = int(data.get("layer", 12))

    if MOCK_MODE:
        point = _mock_project_prompt(prompt, alpha_r, alpha_s, layer)
        return jsonify({"prompt_point": point, "mock": True})

    return jsonify({"error": "Not implemented"}), 501


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Lobotomy Lab backend")
    parser.add_argument("--mock", action="store_true", default=True,
                        help="Run with mock data (default: True)")
    parser.add_argument("--no-mock", dest="mock", action="store_false",
                        help="Run with real model inference")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true", default=True)
    args = parser.parse_args()

    MOCK_MODE = args.mock
    mode_label = "MOCK" if MOCK_MODE else "LIVE"
    print(f"[LLM Lobotomy Lab] {mode_label} mode")
    print(f"  http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=args.debug)
