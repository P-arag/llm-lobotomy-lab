# Backend

The backend provides the model-serving, activation-recording, and visualization
API used by the frontend.

## Features

- Loads Llama 2 once when the live server starts and keeps it in memory.
- Supports refusal and sycophancy activation directions.
- Generates paired unsteered and steered responses.
- Captures each prompt's residual-stream activation at the selected layer.
- Computes a fixed, deterministic 3-D UMAP from a 100-example-per-class
  subsample at startup.
- Maintains a live PCA projection that is refit after each prompt.
- Serves the generation, UMAP, PCA, and prompt-projection endpoints.
- Provides mock mode for frontend development without model weights.

Record a steering direction from either behavior dataset:

```sh
conda run -n lobo-lab python backend/ml/record_activations.py --steering sycophancy --layer 13
conda run -n lobo-lab python backend/ml/record_activations.py --steering refusal --layer 13
```

Generate with a recorded direction (refusal remains the default):

```sh
conda run -n lobo-lab python backend/ml/steer_generate.py --steering sycophancy --alpha 1.0
```

Record a balanced 200-point sycophancy sample at layers 8 through 17 and
create one PCA-2 plot for each layer:

```sh
conda run -n lobo-lab python backend/ml/record_sycophancy_layer_sweep.py
```

The activation files and plots are written to
`backend/ml/out/layer_sweep_200/`. Use `--samples` to change the total sample
size, or `--out-dir` to choose another output directory.

## Usage

Run from the repository root. Create a `.env` file containing a Hugging Face
token with access to `meta-llama/Llama-2-7b-chat-hf`:

```text
HF_TOKEN=your_hugging_face_token
```

Start the live backend:

```sh
conda run -n lobo-lab python backend/app.py
```

The server is available at `http://localhost:5000`. Live startup loads the
model and precomputes the initial UMAP before accepting requests.

For frontend-only development, use mock mode:

```sh
conda run -n lobo-lab python backend/app.py --mock
```

Useful API routes:

- `GET /api/presets` — demo prompts.
- `POST /api/generate` — paired generated responses.
- `GET /api/umap?layer=13` — fixed 3-D UMAP coordinates.
- `GET /api/pca?layer=13` — current PCA background.
- `POST /api/project` — capture a prompt activation, update PCA, and return
  the prompt's PCA point and refreshed background.

The activation recorder writes `.pt` files to `backend/ml/out/`. Direction
files are named `{steering}_layer{layer}_direction.pt`; activation files are
named `{dataset}_layer{layer}_activations.pt`.