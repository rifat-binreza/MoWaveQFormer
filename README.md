<div align="center">

# 🫀 MoWaveQFormer
### Motion-conditioned signals. Quality-gated attention.
**Smartphone PPG heart-rate estimation with learned motion-specific filters and a Transformer.**

[![arXiv](https://img.shields.io/badge/arXiv-2609.16248-B31B1B?style=for-the-badge)](https://arxiv.org/abs/2609.16248)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](notebooks/brnodatajournal.ipynb)
[![Dataset](https://img.shields.io/badge/BUT_PPG-v2.0.0-16B8A6?style=for-the-badge)](https://physionet.org/content/butppg/2.0.0/)

**Swandip Singha · Rifat Bin Reza · Saifur Rahman Sabuj**

[Paper](https://arxiv.org/abs/2609.16248) · [Notebook](notebooks/brnodatajournal.ipynb) · [Recorded run](results/logs/kaggle-run.txt) · [Reproducibility notes](docs/REPRODUCIBILITY.md)

</div>

---

## Research snapshot

Research materials associated with **MoWaveQFormer: A Motion-Conditioned Quality-Gated Transformer for Smartphone-Based PPG Heart Rate Estimation**.

This release preserves the supplied experiment notebook and execution log. The notebook uses the earlier implementation name **MoWaveNet**; class and checkpoint names are retained for traceability. Exact correspondence between this snapshot and every experiment in the preprint has not been independently verified. Results below are transcribed from the supplied run, not newly reproduced.

## Method at a glance

| Component | Implementation in this snapshot |
| :--- | :--- |
| Motion conditioning | Group-specific learned 1-D filter bank: eight filters, kernel length 31 |
| Quality gating | Quality-conditioned Transformer with patch size 10, width 128, four heads and four layers |
| Prediction | Heart-rate regression from 300-sample, 10-second PPG windows |
| Auxiliary training | PTT-related loss using the heuristic `PTT_pred = 0.35 × 60 / HR_pred` |
| Evaluation | Subject-block split, ablations, baseline comparisons, Bland–Altman and outlier analysis |

**Conditioning matters:** the reported model evaluation uses annotation-derived motion groups and signal-quality labels. The separate ACC detector experiment does not establish an end-to-end automatic conditioning pipeline.

## Recorded results

BUT PPG v2.0.0 · **598 test records** · errors in beats per minute. Baseline names describe the implementations in this notebook, not verified reproductions of their original publications.

| Notebook method | MAE ↓ | RMSE ↓ | Pearson r ↑ |
| :--- | ---: | ---: | ---: |
| TROIKA-labelled SP1 | 20.559 | 26.129 | 0.1238 |
| TAPIR-style | 21.497 | 26.686 | 0.1799 |
| DeepPPG | 8.430 | 13.846 | 0.1515 |
| CNN–BiLSTM | 8.479 | 14.015 | 0.0670 |
| ResNet1D / Q-PPG-style | 8.175 | 14.246 | 0.1009 |
| **MoWaveNet (this snapshot)** | **7.851** | **13.377** | **0.2920** |

The MAE difference from ResNet1D is **0.324 bpm (about 4.0%)**. The run's paired Wilcoxon test gives **p = 0.31871**, so it does **not** establish statistical significance against that baseline. See the [reporting caveats](docs/REPRODUCIBILITY.md) before interpreting the raw log's auto-generated paper text.

[Machine-readable results](results/metrics.csv) · [Complete original log](results/logs/kaggle-run.txt)

## Run the notebook

The original workflow targets **Kaggle** and contains training, ablation and additional baseline stages. It is an experiment notebook, not an installable inference package.

1. Obtain [BUT PPG v2.0.0](https://physionet.org/content/butppg/2.0.0/) and retain its record directory structure and annotation CSVs.
2. Open [brnodatajournal.ipynb](notebooks/brnodatajournal.ipynb) in Kaggle or Jupyter. Install the dependencies from `requirements.txt`; select an appropriate PyTorch build for your CPU/CUDA environment.
3. Adjust **every `ROOT` assignment** to the extracted dataset directory. The saved notebook expects `/kaggle/input/datasets/swandipsingha/brnounidataset/brno-university-of-technology-smartphone-ppg-database-but-ppg-2.0.0`.
4. For local execution, replace all `/kaggle/working` paths with one writable output directory and create it first. On Kaggle, the existing paths can be retained.
5. Execute the eight code cells in order. Training generates `mowavenet_final_best.pt`, split IDs, CSVs and figures used by later cells. Use a fresh output directory when intentionally generating a new split.

Dependencies are unpinned because the original environment lockfile was not supplied. No trained checkpoints or raw dataset are bundled. Full training has not been rerun as part of repository preparation.

## Repository guide

| Path | Purpose |
| :--- | :--- |
| `notebooks/brnodatajournal.ipynb` | Original notebook, including saved outputs, unchanged |
| `results/logs/kaggle-run.txt` | Original supplied execution log, unchanged |
| `results/metrics.csv` | Main comparison table transcribed from that log |
| `docs/REPRODUCIBILITY.md` | Data assumptions, limitations and provenance |
| `CITATION.cff` / `citation.bib` | Preprint citation |
| `requirements.txt` | Dependencies inferred from notebook imports |

## Citation and data credit

Please cite the [MoWaveQFormer preprint](https://arxiv.org/abs/2609.16248) when referring to this research; BibTeX is available in [citation.bib](citation.bib).

The dataset is **Brno University of Technology Smartphone PPG Database (BUT PPG), v2.0.0**, DOI [10.13026/tn53-8153](https://doi.org/10.13026/tn53-8153), distributed by PhysioNet under CC BY 4.0. Follow the dataset's citation and attribution requirements. Data ownership remains with its original contributors.

**Software licensing:** this repository's software is available under the [MIT License](LICENSE). The BUT PPG dataset retains its separate CC BY 4.0 license, and the linked paper retains its own terms. Research code has not been clinically validated.
