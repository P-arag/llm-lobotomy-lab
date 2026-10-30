"""
Backend application for LLM Lobotomy Lab.
Serves the web dashboard and provides REST API endpoints for steered text generation,
residual-stream activation projections (2D PCA), and cluster representations (3D UMAP).

NOTE FOR BACKEND DEVELOPERS:
This file connects the frontend to model inference. The mock data calls currently delegate
to `backend/mock_data.py`. Follow the inline comments in each endpoint to connect real
PyTorch / Hugging Face models (e.g., Llama-2-7b-chat, Qwen-2.5) and activation vectors.
Once real inference is wired up, `backend/mock_data.py` can be safely deleted.
"""

import argparse
import os
import sys

from flask import Flask, jsonify, render_template, request

# Temporary mock data import — delete once real model inference is connected
try:
    from backend import mock_data
except ImportError:
    import mock_data

app = Flask(
    __name__,
    template_folder="../frontend/templates",
    static_folder="../frontend/static",
)

MOCK_MODE = True


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
    data = request.get_json(force=True, silent=True) or {}
    prompt = str(data.get("prompt", "")).strip()

    try:
        alpha_refusal = float(data.get("alpha_refusal", 0.0))
    except (ValueError, TypeError):
        alpha_refusal = 0.0

    try:
        alpha_sycophancy = float(data.get("alpha_sycophancy", 0.0))
    except (ValueError, TypeError):
        alpha_sycophancy = 0.0

    try:
        layer = int(float(data.get("layer", 12)))
    except (ValueError, TypeError):
        layer = 12
    layer = max(0, min(31, layer))
    model = str(data.get("model", "qwen-3b"))

    if MOCK_MODE:
        result = mock_data.generate_mock_response(prompt, alpha_refusal, alpha_sycophancy, layer, model)
        result["mock"] = True
        return jsonify(result)

    # ─────────────────────────────────────────────────────────────────
    # BACKEND INTEGRATION POINT:
    # 1. Load model and tokenizer (see backend/ml/steer_generate.py).
    # 2. Register forward hook at `layer` adding:
    #    alpha_refusal * refusal_vec + alpha_sycophancy * sycophancy_vec
    # 3. Generate steered and unsteered tokens.
    # 4. Compute token-level logit_delta or activation delta.
    # ─────────────────────────────────────────────────────────────────
    return jsonify({"error": "Real inference not implemented yet"}), 501


@app.route("/api/pca", methods=["GET"])
def api_pca():
    """Return 2D PCA projection coordinates of activation clusters for a layer.

    Query Parameters:
        layer (int, default=12): Layer index.

    Response (JSON):
        layer (int): Active layer.
        sycophantic:
            x (list[float]): X coordinates of sycophantic activations.
            y (list[float]): Y coordinates of sycophantic activations.
        non_sycophantic:
            x (list[float]): X coordinates of non-sycophantic activations.
            y (list[float]): Y coordinates of non-sycophantic activations.
    """
    try:
        layer = int(float(request.args.get("layer", 12)))
    except (ValueError, TypeError):
        layer = 12
    layer = max(0, min(31, layer))

    if MOCK_MODE:
        return jsonify(mock_data.get_mock_pca(layer))

    # ─────────────────────────────────────────────────────────────────
    # BACKEND INTEGRATION POINT:
    # Load precomputed PCA coordinates from backend/ml/out/{layer}_pca.json
    # or compute via sklearn.decomposition.PCA over recorded activations.
    # ─────────────────────────────────────────────────────────────────
    return jsonify({"error": "Not implemented"}), 501


@app.route("/api/umap", methods=["GET"])
def api_umap():
    """Return 3D UMAP coordinates of activation clusters for a layer.

    Query Parameters:
        layer (int, default=12): Layer index.

    Response (JSON):
        layer (int): Active layer.
        sycophantic:
            x (list[float]): X coordinates.
            y (list[float]): Y coordinates.
            z (list[float]): Z coordinates.
        non_sycophantic:
            x (list[float]): X coordinates.
            y (list[float]): Y coordinates.
            z (list[float]): Z coordinates.
    """
    try:
        layer = int(float(request.args.get("layer", 12)))
    except (ValueError, TypeError):
        layer = 12
    layer = max(0, min(31, layer))

    if MOCK_MODE:
        return jsonify(mock_data.get_mock_umap(layer))

    # ─────────────────────────────────────────────────────────────────
    # BACKEND INTEGRATION POINT:
    # Load precomputed 3D UMAP coordinates from backend/ml/out/{layer}_umap.json
    # (see backend/ml/plot_sycophancy_pca_umap.py).
    # ─────────────────────────────────────────────────────────────────
    return jsonify({"error": "Not implemented"}), 501


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
    data = request.get_json(force=True, silent=True) or {}
    prompt = str(data.get("prompt", "")).strip()

    try:
        alpha_r = float(data.get("alpha_refusal", 0.0))
    except (ValueError, TypeError):
        alpha_r = 0.0

    try:
        alpha_s = float(data.get("alpha_sycophancy", 0.0))
    except (ValueError, TypeError):
        alpha_s = 0.0

    try:
        layer = int(float(data.get("layer", 12)))
    except (ValueError, TypeError):
        layer = 12
    layer = max(0, min(31, layer))

    if MOCK_MODE:
        point = mock_data.project_mock_prompt(prompt, alpha_r, alpha_s, layer)
        return jsonify({"prompt_point": point, "mock": True})

    # ─────────────────────────────────────────────────────────────────
    # BACKEND INTEGRATION POINT:
    # 1. Forward prompt through model up to `layer`.
    # 2. Extract residual stream activation at second-to-last token position.
    # 3. Apply fitted PCA transformer (`pca.transform(act.reshape(1, -1))`).
    # 4. Return {"prompt_point": {"x": float(x), "y": float(y)}}.
    # ─────────────────────────────────────────────────────────────────
    return jsonify({"error": "Not implemented"}), 501


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
