# MoWaveQFormer

A compact public-facing summary of a motion-aware physiological signal study built around smartphone PPG and related sensor streams.

## Overview

MoWaveQFormer explores how motion context and signal quality affect physiological estimation from smartphone-acquired signals. The public repository is intentionally minimal: it highlights the research framing, the experimental setup, and the methodological approach without exposing the full implementation or executable notebook internals.

## Research focus

- Estimate physiological response under motion and quality variation
- Combine PPG, ECG, and accelerometer-derived context
- Use motion-aware filtering and quality-aware analysis to improve robustness
- Evaluate performance with subject-aware splits and clinically relevant summaries

## Dataset and signals

This work uses a smartphone-based Brno-style dataset containing synchronized physiological and motion recordings. The public summary focuses on the data modalities and the analysis protocol, rather than the raw implementation details.

- PPG signal
- ECG signal
- Accelerometer signals
- Subject metadata and quality annotations

## Method summary

The public methodology follows a structured pipeline:

1. Data ingestion and subject-level validation
2. Signal cleaning and quality screening
3. Feature extraction across time and frequency domains
4. Motion-aware grouping and baseline normalization
5. Rule- and model-based assessment of physiological states
6. Validation using subject-aware evaluation and class summaries

## Public documentation

- Methods: `docs/METHODS.md`
- Reproducibility notes: `docs/REPRODUCIBILITY.md`
- Notebook summary: `notebooks/brnodatajournal.ipynb`

## Repository structure

- `notebooks/` — public, high-level research note
- `docs/` — public methods and reproducibility documentation
- `results/` — saved metrics and recorded run artifacts

## Notes

This repository is intentionally kept lightweight in the public view to protect implementation details while preserving the scientific context, methodology, and evaluation narrative.
