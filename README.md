# MoWaveQFormer

<div align="center">

  <img src="https://img.shields.io/badge/Research-PPG%20Heart%20Rate-8A2BE2?style=for-the-badge" alt="Research area" />
  <img src="https://img.shields.io/badge/Mode-Motion%20Conditioned-00C853?style=for-the-badge" alt="Mode" />
  <img src="https://img.shields.io/badge/Quality-Gated-FF6D00?style=for-the-badge" alt="Quality gated" />
  <img src="https://img.shields.io/badge/Signals-PPG%20%2B%20ECG%20%2B%20ACC-1E88E5?style=for-the-badge" alt="Signals" />

  <p>
    <strong>Motion-conditioned, quality-gated smartphone PPG heart-rate estimation</strong>
  </p>

  <p>
    A public research summary for robust heart-rate estimation from smartphone-acquired physiological and motion signals.
  </p>

</div>

## Why this matters

Heart-rate estimation from smartphone PPG is highly sensitive to motion artifacts and signal quality changes. This project studies how to condition estimation on motion context and quality information so the model can remain reliable in realistic, noisy recording environments.

## Project snapshot

- Estimates heart rate from smartphone-based photoplethysmography (PPG)
- Uses motion context and quality gating to suppress unreliable segments
- Combines multi-sensor context from PPG, ECG, and accelerometry
- Evaluates with subject-aware validation and clinically relevant metrics
- Keeps the public repo lightweight while documenting the research workflow and reproducibility notes

## High-level pipeline

```mermaid
flowchart LR
    A[Smartphone sensors\nPPG + ECG + Accelerometer + metadata] --> B[Data validation\nand synchronization]
    B --> C[Quality screening\nremove low-confidence segments]
    C --> D[Motion-aware preprocessing\nfilter + normalize + align]
    D --> E[Feature extraction\ntime + frequency + quality features]
    E --> F[Quality gating\n& motion conditioning]
    F --> G[Heart-rate estimation]
    G --> H[Subject-aware evaluation]
    H --> I[Report\nMAE / RMSE / Pearson r]

    classDef sensor fill:#1f6feb,stroke:#0b3b8f,color:#fff,stroke-width:1px;
    classDef proc fill:#0ea5e9,stroke:#075985,color:#fff,stroke-width:1px;
    classDef model fill:#8b5cf6,stroke:#4c1d95,color:#fff,stroke-width:1px;
    classDef result fill:#10b981,stroke:#065f46,color:#fff,stroke-width:1px;

    class A,B,C,D,E sensor;
    class F,G model;
    class H,I result;
```

## Key idea

The project treats motion and signal quality as first-class contextual variables rather than nuisance noise. By screening low-quality segments and conditioning on motion context, the pipeline aims to stabilize heart-rate estimation under realistic usage conditions.

## Public benchmark snapshot

The repository includes a compact result summary in `results/metrics.csv`. Below is the current public benchmark snapshot for the reported methods.

| Method | MAE (bpm) | RMSE (bpm) | Pearson r | N |
|---|---:|---:|---:|---:|
| TROIKA-labelled SP1 | 20.559 | 26.129 | 0.1238 | 598 |
| TAPIR-style | 21.497 | 26.686 | 0.1799 | 598 |
| DeepPPG | 8.430 | 13.846 | 0.1515 | 598 |
| CNN-BiLSTM | 8.479 | 14.015 | 0.0670 | 598 |
| ResNet1D Q-PPG-style | 8.175 | 14.246 | 0.1009 | 598 |
| MoWaveNet | 7.851 | 13.377 | 0.2920 | 598 |

<div align="center">

  <p><strong>Best public result in this snapshot:</strong> <code>MoWaveNet</code> with <strong>7.851 bpm MAE</strong> and <strong>0.292 Pearson r</strong></p>

</div>

## Scientific framing

This work sits at the intersection of:

- physiological signal processing
- motion artifact handling
- quality-aware estimation
- smartphone sensing and robustness analysis
- reproducible research in mobile health

## Repository structure

```text
.
├── docs/
│   ├── METHODS.md
│   ├── REPRODUCIBILITY.md
│   └── PRIVATE_ARCHITECTURE.md
├── notebooks/
│   └── brnodatajournal.ipynb
├── results/
│   ├── metrics.csv
│   └── logs/
├── README.md
├── requirements.txt
├── CITATION.cff
├── citation.bib
├── LICENSE
└── .gitignore
```

## Public documentation

- `docs/METHODS.md` — public method overview and algorithm sketch
- `docs/REPRODUCIBILITY.md` — reproducibility notes and run setup details
- `notebooks/brnodatajournal.ipynb` — research notebook summary
- `results/metrics.csv` — public benchmark summary

## Quick start

1. Review the public methods summary in `docs/METHODS.md`
2. Inspect the notebook in `notebooks/brnodatajournal.ipynb`
3. Check `results/metrics.csv` for the benchmark snapshot
4. Use the reproducibility notes for environment and execution context

## Citation

If you use this project in your work, please cite the repository metadata in `CITATION.cff` or the BibTeX file `citation.bib`.

## Notes

This repository is intentionally kept concise in its public view while still preserving enough methodological and evaluation context for others to understand the work, inspect the workflow, and reproduce the broader research narrative.

---

<p align="center">
  <sub>Built for clarity, motion-aware sensing, and reproducible research.</sub>
</p>














































































































































































































































































