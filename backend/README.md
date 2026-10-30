# Backend Architecture & API Specification

This directory contains the Flask server (`app.py`), the temporary mock data provider (`mock_data.py`), and the machine learning scripts (`ml/`).

## Quick Start

Run the server with:
```sh
uv run python backend/app.py --port 5000
```

By default, the server runs in `--mock` mode so the frontend can function without a GPU.
To switch to real inference once implemented, launch with `--no-mock`.

---

## File Structure

- `app.py`: The main REST API entrypoint. Handles input sanitization, error reporting, and routing.
- `mock_data.py`: **Temporary mock data module**. Provides simulated responses, token heatmap values, and cluster coordinates so the frontend can be run and developed independently. **Delete this file once real model inference is connected.**
- `ml/`: Scripts for recording activations, computing steering directions, and plotting:
  - `record_activations.py`: Dumps residual-stream activations and difference-of-means vectors (`.pt`).
  - `steer_generate.py`: Implements residual-stream forward hooks (`_steering_hook`) for generation.
  - `plot_sycophancy_pca_umap.py`: Computes 2D PCA and 3D UMAP clusters.

---

## API Contract (What the Frontend Consumes)

### 1. `POST /api/generate`
Generates both baseline (unsteered) and steered text with token-level activation values.

**Request Body:**
```json
{
  "prompt": "How do I hack a WiFi network?",
  "alpha_refusal": 2.5,
  "alpha_sycophancy": 0.0,
  "layer": 12,
  "model": "qwen-3b"
}
```

**Response Format:**
```json
{
  "unsteered": {
    "text": "Sure! To analyze network security...",
    "tokens": [
      { "token": "Sure!", "logit_delta": 0.25 },
      { "token": "To", "logit_delta": -0.01 }
    ]
  },
  "steered": {
    "text": "I cannot assist with unauthorized access...",
    "tokens": [
      { "token": "I", "logit_delta": 0.02 },
      { "token": "cannot", "logit_delta": 0.72 }
    ]
  },
  "params": {
    "alpha_refusal": 2.5,
    "alpha_sycophancy": 0.0,
    "layer": 12,
    "model": "Qwen2.5-3B-abliterated"
  }
}
```
*Note on `logit_delta`: Values range from `-1.0` (suppressed) to `+1.0` (amplified). The frontend uses these values to color-code the real-time token heatmap.*

---

### 2. `GET /api/pca?layer=12`
Returns 2D PCA projection coordinates for the background clusters.

**Response Format:**
```json
{
  "layer": 12,
  "sycophantic": {
    "x": [1.2, 2.4, 0.8],
    "y": [0.5, 1.1, -0.3]
  },
  "non_sycophantic": {
    "x": [-1.5, -2.1, -0.9],
    "y": [-0.4, -1.2, 0.2]
  }
}
```

---

### 3. `GET /api/umap?layer=12`
Returns 3D UMAP coordinates for the rotating cluster plot.

**Response Format:**
```json
{
  "layer": 12,
  "sycophantic": {
    "x": [1.5, 2.1, 0.9],
    "y": [0.8, 1.3, -0.2],
    "z": [1.1, 0.4, 0.7]
  },
  "non_sycophantic": {
    "x": [-1.2, -1.9, -0.8],
    "y": [-0.5, -1.1, 0.3],
    "z": [0.2, 0.9, -0.4]
  }
}
```

---

### 4. `POST /api/project`
Projects the user's active prompt activation onto the 2D PCA plot (rendered as the yellow diamond).

**Request Body:**
```json
{
  "prompt": "User prompt text",
  "alpha_refusal": 1.0,
  "alpha_sycophancy": 0.0,
  "layer": 12
}
```

**Response Format:**
```json
{
  "prompt_point": {
    "x": 0.4521,
    "y": -0.3129
  }
}
```

---

## Instructions for Connecting Real Model Inference

1. In `app.py`, locate the `# BACKEND INTEGRATION POINT` comments in each endpoint.
2. Replace the call to `mock_data.*` with your inference pipeline:
   - For generation: Import `generate_text_from_prompt` from `ml/steer_generate.py`.
   - For PCA/UMAP: Load precomputed coordinates from `ml/out/` or compute them on demand.
3. Once all endpoints use real weights, delete `backend/mock_data.py`.
