# Deliberation Controller

This repository contains data processing, controller modeling, and supervised training code for a Deliberation Controller.

## Project Layout

- `deliberation_controller/data/`: signal extraction, normalization, dataset preparation
- `deliberation_controller/model/`: controller model definitions
- `deliberation_controller/train/`: training scripts
- legacy analysis scripts at repository root are kept for reference

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you train on GPU, install a CUDA-compatible PyTorch build for your target server.

## Prepare Dataset

```bash
python -m deliberation_controller.data.prepare_dataset \
  --input "<path_to_trajectory_json>" \
  --output "./deliberation_controller/data/dataset.json" \
  --reference-distribution "./deliberation_controller/data/reference_distribution.json" \
  --window-size 5 \
  --seed 42 \
  --rebuild-reference
```

## Train (Supervised)

```bash
python -m deliberation_controller.train.train_sl \
  --data_path ./deliberation_controller/data/dataset.json \
  --epochs 100 \
  --batch_size 256 \
  --lr 1e-3 \
  --patience 10 \
  --save_dir ./deliberation_controller/checkpoints \
  --gate_threshold 0.5
```

## Notes

- `action_label = -100` indicates gate=0 samples and is ignored in action loss.
- Large datasets/checkpoints are ignored by default via `.gitignore`.

