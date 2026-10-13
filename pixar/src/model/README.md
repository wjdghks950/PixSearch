# PixSearch model code

See the [main README](../../../README.md) for installation, data preparation,
training, checkpoint export, VQA with/without retrieval, and segmentation evaluation.
The Python package and class names retain `pixar` / `PIXAR` for checkpoint compatibility.

Use `bash scripts/model.sh` from the repository root. The older `run_*.sh`
files are historical experiment recipes; the root launcher accepts explicit paths
and forwards arguments to the actual Python entry points.
