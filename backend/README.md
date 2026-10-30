Should implement recording of activations across atleast two models (Llama 2 and something else). 
- Figure out which layer gives best steering for what task for the other model
- Dump all the activations for the tasks so that the frontend can plot the rotating umap
- provide a clean api for steering and recording activations that the frontend will use

Record a steering direction from either behavior dataset:

```sh
uv run python backend/ml/record_activations.py --steering sycophancy --layer 13
uv run python backend/ml/record_activations.py --steering refusal --layer 13
```

Generate with a recorded direction (refusal remains the default):

```sh
uv run python backend/ml/steer_generate.py --steering sycophancy --alpha 1.0
```