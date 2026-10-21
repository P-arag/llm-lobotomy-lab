# LLM Lobotomy Lab

## Backend setup

Create and activate the Conda environment:

```sh
conda create --name llm-lobo python=3.11
conda activate llm-lobo
```

Install the Python dependencies with [uv](https://docs.astral.sh/uv/):

```sh
uv sync
```

The project is configured to install PyTorch from the CUDA 12.8 wheel index.
Before running the activation script, create a `.env` file containing
an `HF_TOKEN` with access to the required Hugging Face model:

```text
HF_TOKEN=your_hugging_face_token
```

Run the current activation analysis from the repository root:

```sh
uv run python backend/plot_sycophancy_pca_umap.py
```