# MoWaveQFormer · Python implementation

Documented modules extracted from the supplied research notebook, with portable entry points for training, evaluation and prediction. The original model names and checkpoint keys are retained; `MoWaveQFormer` aliases `MoWaveNet`.

## Modules

| File | Responsibility |
| --- | --- |
| `mowave/config.py` | Sampling rates, window dimensions and activity mapping |
| `mowave/preprocessing.py` | SWT filtering, bandpass, normalization, peak detection and PTT |
| `mowave/motion.py` | Optional accelerometer rule classifier and signal features |
| `mowave/data.py` | WFDB loading, CSV preparation, validated subject split and augmentation |
| `mowave/model.py` | Motion-specific learnable FIR banks and quality-gated Transformer |
| `mowave/losses.py` | Valid-target physiological PTT consistency loss |
| `mowave/baselines.py` | Supplied neural and spectral comparison implementations |
| `mowave/training.py` | AdamW training, warmup, plateau scheduling and checkpoint selection |
| `mowave/evaluation.py` | Record predictions, MAE, RMSE, correlation and agreement metrics |
| `train.py` | Training CLI and three ablation switches |
| `evaluate.py` | Test-set metrics, quality/group breakdowns and predictions CSV |
| `predict.py` | Single-window HR inference with a trained checkpoint |

## Setup

Use Python 3.10 or later. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r python/requirements.txt
```

Provide a locally obtained BUT PPG v2.0 dataset. `--data-root` must contain `quality-hr-ann.csv`, `subject-info.csv`, and record directories such as `100001/100001_PPG.hea` with their WFDB signal files. Optional ECG peak annotations are `100001/100001.qrs`. Labels must include `id`, `hr`, `quality` in the annotation CSV and `id`, `motion` in the subject CSV. Alternatively pass an already prepared `--metadata` CSV with `base_id,subject_id,hr,quality,motion`.

## Train and evaluate

```bash
python python/train.py --data-root /path/to/but-ppg --output runs/mowave --epochs 80
python python/evaluate.py --data-root /path/to/but-ppg --run runs/mowave
```

CPU is the portable default; add `--device cuda` to both commands for a suitable GPU. Training uses batch size 32, AdamW at 3e-4, weight decay 1e-4, five-epoch linear warmup, plateau reduction and PTT weight 0.1. The default split uses the notebook's seed-42 subject shuffle. `split_ids.json` is saved and checked for subject leakage; exact paper partition counts require the original metadata.

For ablations add `--fixed-wavelet`, `--no-quality-gate` or `--no-ptt`, and use a separate output directory. Evaluation reconstructs the matching model from the saved configuration. Runs produce `best.pt`, `model_config.json`, `master_ann.csv`, `split_ids.json`, `history.csv`, then `test_predictions.csv` and `test_metrics.json`.

## Predict a window

```bash
python python/predict.py --ppg window.csv --run runs/mowave --motion-group 0 --quality 1
```

The input is a headerless single-column CSV containing exactly 300 PPG samples at 30 Hz. Groups are 0=subtle/rest, 1=walking, 2=burst. Quality must be supplied as a binary label; this release does not infer SQI. This CLI accepts the motion group directly. The optional ACC rule function is separate and is not used to claim the paper's annotated-group results. ECG is used for training supervision, not prediction.

## Verification and limits

```bash
PYTHONPATH=python python -m unittest discover -s python/tests -v
```

Checks cover all three motion groups, the 816,445-parameter architecture, 768-parameter filter bank, finite gradients, checkpoint reload, preprocessing, split isolation and a one-epoch synthetic training run. Model outputs were also checked against the original notebook classes with identical weights and inputs and matched exactly. These checks do not rerun the dataset benchmarks.

The refactor rejects unreadable records and invalid labels instead of silently substituting zero PPG or HR=78. The notebook SWT operation and physiological coefficient are retained, not independently revalidated. PTT absence is explicitly masked. Agreement limits use sample standard deviation. Spectral functions are the notebook comparators: `troika_hr` is not the complete original TROIKA algorithm; `harmonic_sp_hr` aliases the legacy `tapir_hr` implementation.

No dataset, trained weights, raw private notebook, personal Kaggle paths or credentials are bundled. Paper tables in the main README remain paper-reported evidence. See [the implementation notes](../docs/IMPLEMENTATION.md).
