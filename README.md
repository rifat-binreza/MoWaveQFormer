# MoWaveQFormer

<div align="center">

  <img src="https://img.shields.io/badge/Research-PPG%20Heart%20Rate-8A2BE2?style=for-the-badge&logo=heart&logoColor=white" alt="PPG heart-rate research" />
  <img src="https://img.shields.io/badge/Mode-Motion%20Conditioned-00C853?style=for-the-badge" alt="Motion conditioned" />
  <img src="https://img.shields.io/badge/Quality-Gated-FF6D00?style=for-the-badge" alt="Quality gated" />
  <img src="https://img.shields.io/badge/Signals-PPG%20%2B%20ECG%20%2B%20ACC-1E88E5?style=for-the-badge" alt="Multi-sensor" />

  <h3><strong>Motion-conditioned, quality-gated smartphone PPG heart-rate estimation</strong></h3>

  <p>
    A public research summary for robust heart-rate estimation from smartphone-acquired physiological and motion signals.
  </p>

</div>

## ✨ Overview

MoWaveQFormer investigates how motion context and signal quality affect heart-rate estimation from smartphone-acquired PPG recordings. The project is designed around the idea that physiological signals are not just noisy — they are context-dependent: motion, drift, and signal reliability all shape how reliable the estimate should be.

This repository presents a compact public-facing summary of the research workflow, the evaluation narrative, and the benchmark snapshot, while keeping the richer implementation details in the broader project context.

## 🎯 Why this matters

Smartphone PPG is attractive because it is easy to collect and highly scalable, but it is also highly sensitive to:

- motion artifacts
- sensor instability
- low-quality segments
- subject variability
- environmental or device-induced drift

Instead of treating these as isolated nuisances, the project explicitly models motion and signal quality as part of the estimation process.

## 🧠 Core idea

The main hypothesis is simple but powerful:

- quality-aware conditioning improves robustness
- motion-aware preprocessing stabilizes the signal stream
- subject-aware validation gives a more trustworthy estimate of real-world performance

In practice, the system screens low-confidence signal windows, accounts for motion context, and evaluates the output under realistic subject-level splits.

## 🔁 End-to-end pipeline

```mermaid
flowchart LR
    A[Smartphone sensors<br/>PPG + ECG + Accelerometer + metadata] --> B[Validation & synchronization]
    B --> C[Signal quality screening]
    C --> D[Motion-aware preprocessing<br/>filter + normalize + align]
    D --> E[Feature extraction<br/>time + frequency + quality features]
    E --> F[Quality gating & motion conditioning]
    F --> G[Heart-rate estimation]
    G --> H[Subject-aware evaluation]
    H --> I[Reported metrics<br/>MAE / RMSE / Pearson r]

    classDef sensor fill:#2563eb,stroke:#1d4ed8,color:#fff,stroke-width:1px;
    classDef process fill:#0ea5e9,stroke:#0369a1,color:#fff,stroke-width:1px;
    classDef model fill:#8b5cf6,stroke:#6d28d9,color:#fff,stroke-width:1px;
    classDef result fill:#10b981,stroke:#047857,color:#fff,stroke-width:1px;

    class A,B,C,D,E sensor;
    class F,G process;
    class H,I result;
```

## 📊 Method summary

The public workflow follows a structured research pipeline:

1. Data validation and subject-level checks
2. Synchronization across sensor streams
3. Signal quality screening and motion-aware filtering
4. Feature extraction across time and frequency domains
5. Quality-gated and motion-conditioned analysis
6. Subject-aware evaluation and comparative reporting

## 📈 Benchmark snapshot

The public results in `results/metrics.csv` summarize a compact comparison of several baseline and motion-aware methods.

| Method | MAE (bpm) | RMSE (bpm) | Pearson r | N |
|---|---:|---:|---:|---:|
| TROIKA-labelled SP1 | 20.559 | 26.129 | 0.1238 | 598 |
| TAPIR-style | 21.497 | 26.686 | 0.1799 | 598 |
| DeepPPG | 8.430 | 13.846 | 0.1515 | 598 |
| CNN-BiLSTM | 8.479 | 14.015 | 0.0670 | 598 |
| ResNet1D Q-PPG-style | 8.175 | 14.246 | 0.1009 | 598 |
| MoWaveNet | 7.851 | 13.377 | 0.2920 | 598 |

<div align="center">

  <p><strong>Best public result in this snapshot:</strong> <code>MoWaveNet</code></p>
  <p><strong>MAE:</strong> 7.851 bpm &nbsp;•&nbsp; <strong>RMSE:</strong> 13.377 bpm &nbsp;•&nbsp; <strong>Pearson r:</strong> 0.292</p>

</div>

## 🧪 Data and signals

This study uses synchronized multimodal recordings, including:

- PPG signal
- ECG signal
- tri-axial accelerometer data
- subject metadata and quality labels

The goal is to better understand how motion and sensor quality alter the reliability of HR estimation in realistic smartphone conditions.

## 📁 Repository layout

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
├── .gitignore
└── .github/
```

## 📚 Public documentation

- `docs/METHODS.md` — public method overview and algorithm sketch
- `docs/REPRODUCIBILITY.md` — execution and reproducibility notes
- `docs/PRIVATE_ARCHITECTURE.md` — internal architecture context
- `notebooks/brnodatajournal.ipynb` — notebook-based research narrative
- `results/metrics.csv` — benchmark snapshot and summary metrics

## 🚀 Quick start

1. Read the method summary in `docs/METHODS.md`
2. Inspect the notebook in `notebooks/brnodatajournal.ipynb`
3. Review the result summary in `results/metrics.csv`
4. Use `docs/REPRODUCIBILITY.md` for execution and environment details

## 📌 Scientific framing

This work sits at the intersection of:

- physiological signal processing
- quality-aware sensing
- motion artifact mitigation
- smartphone-based health monitoring
- reproducible computational research

## 🏷️ Citation

If you use this project in your research, please refer to the metadata in `CITATION.cff` or the BibTeX file `citation.bib`.

## 📝 Notes

This repository is intentionally compact in its public presentation, while still preserving the essential scientific story: the dataset, the method, the motion-aware logic, and the evaluation context needed to understand the work.

---

<p align="center">
  <sub>Built for clarity, robustness, and reproducible research.</sub>
</p>
