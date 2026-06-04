# Dynamic IDK Cascades

Simple research workspace for implementing the Random Forest skip-decision IDK cascade paper step by step.

## Files

- `research_pipeline.ipynb` — self-contained notebook with Hugging Face streaming, model-output caching, RF skip training, cascade evaluation, and smoke tests.
- `dataset.py` — optional copy of the streaming dataset helpers for testing outside Jupyter.
- `artifacts/` — cached model probability/timing files. Raw images are not required in the project.

## Dataset access

The notebook streams ImageNet-V2 from Hugging Face using `datasets.load_dataset("vaishaal/ImageNetV2", streaming=True)`.
It does not require `data/imagenetv2` or any raw dataset folder in this repo.

Install the needed packages in your venv if they are missing:

```bash
pip install datasets torch torchvision timm scikit-learn numpy pillow
```

## How to work

1. Open `research_pipeline.ipynb` in VS Code or Jupyter.
2. Run the setup and streaming check cells.
3. Run the fake-cache logic smoke test.
4. Optionally run the tiny ResNet-18 streaming inference smoke test.
5. Cache one full model at a time into `artifacts/`.
6. Train the Random Forest skipper and compare `no-skip`, `threshold`, and `rf` cascade results.

Full experiments may be slower with streaming. Once logits are cached in `artifacts/`, later RF/cascade experiments do not need to stream the images again.
