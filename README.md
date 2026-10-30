# LLM Lobotomy Lab

What happens when you nudge a language model's internal activations in one
direction or another?

That is the question behind **LLM Lobotomy Lab**, an interactive demo for
exploring two recognizable model behaviors: refusal and sycophancy. Instead of
only looking at the text a model produces, we also look inside the model and
visualize what its activations are doing.

![LLM Lobotomy Lab interface](assets/screenshot.png)

## The problem, in simple terms

Language models can be surprisingly agreeable. They may say yes to a bad idea,
agree with an obviously incorrect claim, or refuse a request depending on small
changes in their internal state.

It is difficult to see why this happens by reading generated text alone. The
interesting information is hidden in the model's intermediate activations,
which are large numerical representations that are not easy to inspect
directly.

## What we do

This project turns those hidden representations into something you can explore
in a browser.

- Compare an unsteered response with a response generated after steering the
  model's activations.
- Adjust refusal and sycophancy strength with interactive sliders.
- Choose the transformer layer where steering is injected.
- Capture the activation produced by each new prompt.
- See prompt and dataset activations in PCA plots.
- Explore a fixed 3-D UMAP of refusal and sycophancy examples.
- Keep the model loaded for the whole session instead of loading it for every
  request.
- Use mock mode to try the interface without downloading model weights.

The UMAP is precomputed from a deterministic sample when the server starts, so
it stays stable while you experiment. The PCA is updated as new prompts are
submitted, allowing you to see where each prompt lands relative to the
recorded examples.

## Running the project

### 1. Create the environment

From the repository root, create and activate the Conda environment:

```sh
conda create --name lobo-lab python=3.11
conda activate lobo-lab
```

Install the project dependencies:

```sh
uv sync
```

The project is configured to install PyTorch from the CUDA 12.8 wheel index.
Use a compatible CUDA setup for live model inference.

### 2. Add your Hugging Face token

Live mode uses `meta-llama/Llama-2-7b-chat-hf`. Make sure your Hugging Face
account has access to the model, then create a `.env` file in the repository
root:

```text
HF_TOKEN=your_hugging_face_token
```

### 3. Start the app

Run the live backend from the repository root:

```sh
uv run python backend/app.py
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.
The first startup loads the model and precomputes the initial visualizations,
so it may take a little while.

### Running without the model

If you want to explore the interface without model weights or a Hugging Face
token, start mock mode:

```sh
uv run python backend/app.py --mock
```

Mock mode is useful for checking the frontend and trying the controls quickly.

## Record activation directions

Activation direction files can be recorded for either behavior (these are essential to be run for plotting the PCA and UMAP):

```sh
uv run python backend/ml/record_activations.py --steering sycophancy --layer 13
uv run python backend/ml/record_activations.py --steering refusal --layer 13
```

To try a recorded direction directly:

```sh
uv run python backend/ml/steer_generate.py --steering sycophancy --alpha 1.0
```

Recorded `.pt` files are written to `backend/ml/out/`.

## Project layout

- `backend/app.py` runs the Flask server and model runtime.
- `backend/ml/` contains activation recording, steering, and model utilities.
- `frontend/` contains the templates, styles, and browser-side interaction code.
- `assets/screenshot.png` shows the dashboard interface.
