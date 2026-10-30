"""
Flask backend for LLM Lobotomy Lab.

Serves the frontend and exposes API endpoints for generation, steering,
and activation visualization. Use `--mock` for frontend-only development.

Run from the repo root:
    uv run python backend/app.py
    uv run python backend/app.py --mock   # frontend-only mock mode
"""

import argparse
import os
import random
import threading
import time

import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from sklearn.decomposition import PCA
from umap import UMAP

from ml import steer_generate

# ---------------------------------------------------------------------------
# Flask app – templates and static assets live under ../frontend/
# ---------------------------------------------------------------------------
app = Flask(
    __name__,
    template_folder="../frontend/templates",
    static_folder="../frontend/static",
)

# Mock mode remains available for frontend-only development.
MOCK_MODE = True
ML_RUNTIME = None
DEFAULT_LAYER = 13

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


class MLRuntime:
    """Owns the single model instance and the live visualization state."""

    def __init__(self, layer=DEFAULT_LAYER, sample_size=100):
        self.layer = layer
        self.sample_size = sample_size
        self.model, self.tokenizer = steer_generate.load_model()
        self.refusal_direction = steer_generate.load_direction("refusal")
        self.sycophancy_direction = steer_generate.load_direction("sycophancy")
        self._lock = threading.RLock()
        self._pca_activations, self._pca_labels = self._load_initial_activations()
        self._live_activations = []
        self._pca = None
        self._pca_points = None
        self._last_point = None
        self._fit_pca()
        self._umap_data = self._build_umap()

    @property
    def model_name(self):
        return steer_generate.MODEL_ID

    def _load_initial_activations(self):
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "ml", "out", "sycophancy_layer13_activations.pt",
        )
        blob = torch.load(path, map_location="cpu", weights_only=True)
        acts = blob["acts"].numpy()
        labels = blob["labels"].numpy().astype(bool)
        rng = np.random.default_rng(42)
        selected = np.concatenate([
            rng.choice(np.flatnonzero(labels), min(self.sample_size, labels.sum()), replace=False),
            rng.choice(np.flatnonzero(~labels), min(self.sample_size, (~labels).sum()), replace=False),
        ])
        return acts[selected], labels[selected]

    def _fit_pca(self):
        self._pca = PCA(n_components=2)
        all_activations = np.vstack([self._pca_activations, *self._live_activations])
        points = self._pca.fit_transform(all_activations)
        base_points = points[:len(self._pca_activations)]
        self._pca_points = {
            "sycophantic": base_points[self._pca_labels].tolist(),
            "non_sycophantic": base_points[~self._pca_labels].tolist(),
        }

    def _build_umap(self):
        reducer = UMAP(n_components=3, random_state=42)
        points = reducer.fit_transform(self._pca_activations)
        return {
            "layer": self.layer,
            "sycophantic": {
                "x": points[self._pca_labels, 0].tolist(),
                "y": points[self._pca_labels, 1].tolist(),
                "z": points[self._pca_labels, 2].tolist(),
            },
            "non_sycophantic": {
                "x": points[~self._pca_labels, 0].tolist(),
                "y": points[~self._pca_labels, 1].tolist(),
                "z": points[~self._pca_labels, 2].tolist(),
            },
        }

    def generate(self, prompt, alpha_refusal, alpha_sycophancy, layer):
        with self._lock:
            steered_direction = (
                alpha_refusal * self.refusal_direction["direction"]
                + alpha_sycophancy * self.sycophancy_direction["direction"]
            )
            unsteered = steer_generate.generate_text_from_prompt(
                self.model, self.tokenizer, prompt, steered_direction,
                layer, 0.0)
            steered = steer_generate.generate_text_from_prompt(
                self.model, self.tokenizer, prompt, steered_direction,
                layer, 1.0)

        return {
            "unsteered": {"text": unsteered, "tokens": _tokenize_response(unsteered, False)},
            "steered": {"text": steered, "tokens": _tokenize_response(steered, True)},
            "params": {
                "alpha_refusal": alpha_refusal,
                "alpha_sycophancy": alpha_sycophancy,
                "layer": layer,
                "model": self.model_name,
            },
        }

    def project_prompt(self, prompt, layer):
        with self._lock:
            prompt_activation = steer_generate.capture_prompt_activation(
                self.model, self.tokenizer, prompt, layer)
            self._live_activations.append(prompt_activation.numpy())
            self._fit_pca()
            point = self._pca.transform(prompt_activation.numpy().reshape(1, -1))[0]
            self._last_point = {"x": float(point[0]), "y": float(point[1])}
            pca = self.pca_data()
        return self._last_point, pca

    def pca_data(self):
        with self._lock:
            return {
                "layer": self.layer,
                "sycophantic": {
                    "x": [point[0] for point in self._pca_points["sycophantic"]],
                    "y": [point[1] for point in self._pca_points["sycophantic"]],
                },
                "non_sycophantic": {
                    "x": [point[0] for point in self._pca_points["non_sycophantic"]],
                    "y": [point[1] for point in self._pca_points["non_sycophantic"]],
                },
            }

    def umap_data(self):
        return self._umap_data

    def last_point(self):
        with self._lock:
            return self._last_point


def _tokenize_response(text, steered):
    return _mock_tokens_with_heatmap(text, steered)


# ---------------------------------------------------------------------------
# Routes – Pages
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Render the dashboard UI."""
    return render_template("index.html")


@app.route("/api/presets", methods=["GET"])
def api_presets():
    """Return preset prompts for testing."""
    return jsonify(mock_data.PRESET_PROMPTS)


@app.route("/api/generate", methods=["POST"])
def api_generate():
    """Generate unsteered and steered responses for a prompt.

    Request Body (JSON):
        prompt (str): The input prompt from user.
        alpha_refusal (float): Steering strength for refusal vector (typically -5.0 to +5.0).
        alpha_sycophancy (float): Steering strength for sycophancy vector (typically -5.0 to +5.0).
        layer (int): Transformer layer index for steering injection (e.g. 10 to 16).
        model (str): Selected model identifier (e.g. 'qwen-3b', 'llama-3b', 'qwen-27b').

    Response (JSON):
        unsteered:
            text (str): Baseline completion without activation steering.
            tokens (list[dict]): Tokens with activation/logit deltas:
                [{"token": "word", "logit_delta": float}, ...]
        steered:
            text (str): Completion with residual stream forward hook applied.
            tokens (list[dict]): Tokens with activation/logit deltas:
                [{"token": "word", "logit_delta": float}, ...]
        params:
            Echo of applied steering parameters.
    """
    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
    alpha_refusal = float(data.get("alpha_refusal", 0.0))
    alpha_sycophancy = float(data.get("alpha_sycophancy", 0.0))
    layer = int(data.get("layer", DEFAULT_LAYER))

    if MOCK_MODE:
        result = mock_data.generate_mock_response(prompt, alpha_refusal, alpha_sycophancy, layer, model)
        result["mock"] = True
        return jsonify(result)

    try:
        result = ML_RUNTIME.generate(
            prompt, alpha_refusal, alpha_sycophancy, layer)
        result["mock"] = False
        return jsonify(result)
    except Exception as exc:
        app.logger.exception("Live generation failed")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/pca", methods=["GET"])
def api_pca():
    """Return 2D PCA projection coordinates of activation clusters for a layer.

    Query params:
        layer (int, default 13)
    """
    layer = int(request.args.get("layer", DEFAULT_LAYER))

    if MOCK_MODE:
        return jsonify(mock_data.get_mock_pca(layer))

    return jsonify(ML_RUNTIME.umap_data())


@app.route("/api/umap", methods=["GET"])
def api_umap():
    """Return 3D UMAP coordinates of activation clusters for a layer.

    Query params:
        layer (int, default 13)
    """
    layer = int(request.args.get("layer", DEFAULT_LAYER))

    if MOCK_MODE:
        return jsonify(mock_data.get_mock_umap(layer))

    return jsonify(ML_RUNTIME.pca_data())


@app.route("/api/project", methods=["POST"])
def api_project():
    """Project the user's current prompt activation onto the 2D PCA plane.

    Request Body (JSON):
        prompt (str): Current prompt text.
        alpha_refusal (float): Refusal steering strength.
        alpha_sycophancy (float): Sycophancy steering strength.
        layer (int): Layer index.

    Response (JSON):
        prompt_point:
            x (float): Projected X coordinate.
            y (float): Projected Y coordinate.
    """
    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
    alpha_r = float(data.get("alpha_refusal", 0.0))
    alpha_s = float(data.get("alpha_sycophancy", 0.0))
    layer = int(data.get("layer", DEFAULT_LAYER))

    if MOCK_MODE:
        point = mock_data.project_mock_prompt(prompt, alpha_r, alpha_s, layer)
        return jsonify({"prompt_point": point, "mock": True})

    try:
        point, pca = ML_RUNTIME.project_prompt(prompt, layer)
        return jsonify({"prompt_point": point, "pca": pca, "mock": False})
    except Exception as exc:
        app.logger.exception("Live prompt projection failed")
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Lobotomy Lab backend")
    parser.add_argument("--mock", action="store_true", default=False,
                        help="Run with mock data instead of loading the model")
    parser.add_argument("--no-mock", dest="mock", action="store_false",
                        help="Run with real model inference")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true", default=True)
    args = parser.parse_args()

    MOCK_MODE = args.mock
    mode_label = "MOCK" if MOCK_MODE else "LIVE"
    if not MOCK_MODE:
        print("  loading model and activation projections...")
        ML_RUNTIME = MLRuntime()
        print("  model loaded")
    print(f"[LLM Lobotomy Lab] {mode_label} mode")
    print(f"  http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=args.debug, use_reloader=False)
