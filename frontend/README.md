# Frontend

The frontend is a browser-based interface for exploring refusal and
sycophancy steering in Llama 2.

## Features

- Dual response panels for unsteered and steered generations.
- Token-level heatmaps showing activation effects in each response.
- Interactive refusal, sycophancy, and transformer-layer controls.
- Preset prompts for quick demonstrations.
- Rotating 3-D UMAP visualization of the fixed activation sample.
- PCA visualization with separate sycophantic and non-sycophantic classes.
- A highlighted point for the current prompt activation.
- Light and dark themes with responsive panel layout.

## Usage

The Flask backend serves the frontend assets, so start the backend from the
repository root:

```sh
conda run -n lobo-lab python backend/app.py
```

Open [http://localhost:5000](http://localhost:5000) in a browser. Enter a
prompt or choose a preset, adjust the steering sliders, and submit it. The
interface displays the paired responses and updates the PCA projection after
the prompt is processed.

For UI development without a model or Hugging Face credentials, start mock
mode:

```sh
conda run -n lobo-lab python backend/app.py --mock
```

In live mode, the backend loads the model once at startup. The 3-D UMAP
remains fixed during the server session, while the PCA background is refreshed
after each prompt.