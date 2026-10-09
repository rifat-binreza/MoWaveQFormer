# MoWaveQFormer paper guide

This document follows arXiv:2609.16248v1 and takes precedence over earlier summary notes when describing the published preprint's architecture and evaluation.

## Inputs and architecture

A 10-second smartphone PPG window has 300 samples at 30 Hz; synchronized ACC has 1,000 samples on each of three axes at 100 Hz. ECG supplies reference HR and R-peak timing for supervision. It is not an inference-time waveform input to the HR regressor. A binary signal-quality label conditions the soft quality gate.

Per-window min-max normalization maps channels to [-1, 1]. Motion index m chooses one of three banks: subtle/rest, walking, and burst (coughing/laughing). Each bank has eight learnable 31-tap FIR filters and gains, for 768 parameters total. The 8×300 sub-band output is reshaped into 30 patches of width 10; each 80-dimensional patch is projected to 128 dimensions.

A learned sigmoid quality gate reweights embeddings without dropping poor-quality tokens. A four-layer, four-head Transformer uses dimension 128, feed-forward size 512 and dropout 0.1. Average pooling and a 128→64→1 regression head produce HR. Training uses L1 HR error plus a PTT consistency term with weight 0.1. Invalid PTT matches do not contribute to the auxiliary term.

## Motion assignment: evaluation versus deployment

The reported experiments use activity annotations for group assignment. The accelerometer-derived rule-based classifier is evaluated separately and has 56% held-out agreement with annotated groups. Reported HR metrics should not be described as the performance of an annotation-free deployed system.

## Subject-independent evaluation

BUT PPG v2.0 contains 3,888 recordings from 50 subjects. Training uses 35 subjects/2,742 records; validation 7/548; test 8/598. Low-quality records number 3,058 (78.7%). Training group counts are 2,436 subtle/rest, 102 walking and 204 burst. Test counts are 532, 22 and 44 respectively.

## Interpreting improvements

Table II reports MAE 7.851 bpm, RMSE 13.377 bpm and Pearson r 0.292 for MoWaveQFormer. The Wilcoxon comparison is significant against Harmonic-SP, DeepPPG and CNN-BiLSTM, but not ResNet1D (p=0.319). Table IV reports 8.805 bpm MAE on poor-quality and 4.609 bpm on good-quality records; ResNet1D is marginally better on good-quality records (4.606 bpm).

Table VI is a VALIDATION ablation: full 7.162; no motion conditioning 7.079; no quality gate 7.283; no PTT 7.305 bpm. Motion conditioning therefore does not improve this pooled ablation metric. The paper discusses gains in walking and burst activities alongside group imbalance. Avoid saying every component improves every aggregate metric.

Table V reports MoWaveQFormer MAE 10.962 bpm for walking, 9.880 for coughing and 9.442 for laughing, all the lowest among the compared methods for those activities. It does not show a consistent advantage across all subtle/rest activities.

Figures 1–4 in assets/ are reproduced from the paper. Figure 3 shows walking-band specialization but not equally clear burst-band specialization. Figure 4 reports bias -5.54 bpm and limits [-29.42,18.35] bpm. Cloud CPU/GPU timing is not a smartphone-device benchmark.

## Available repository artifacts

The existing notebook, logs and results/metrics.csv are retained. The older snapshot uses MoWaveNet and TAPIR-style labels; results/paper-benchmarks.csv follows the preprint's Table II naming and displayed precision. This is a documentation update, not a new training run. Existing method/reproducibility notes should be read as earlier artifact context rather than a complete verified implementation of the paper.

## Citation

See ../CITATION.cff and ../citation.bib for full attribution. Source: https://arxiv.org/abs/2609.16248
