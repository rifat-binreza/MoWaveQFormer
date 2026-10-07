# Reproducibility and reporting notes

## Release provenance

The notebook and log are byte-for-byte copies of materials supplied by Rifat Bin Reza. Repository preparation added documentation, dependencies and a transcribed metrics CSV; it did not change model code, labels, splits or recorded outputs.

SHA-256:
- `notebooks/brnodatajournal.ipynb`: `2635fa9e231a4ab8c4e806a4f05ee54fc962512be212c205c24541c775f6032d`
- `results/logs/kaggle-run.txt`: `562c5ca83be2a3adf5ecf27809b6a89bbb0f0eaf9934755e66d1a41c46c21961`

Validation during packaging: notebook JSON parsed and all eight code cells passed Python syntax parsing after excluding the shell installation command. This is not a successful training or numerical reproduction claim. Exact package versions, original runtime hardware and trained weights were not supplied.

## Data and split

The log records 3,888 records and 50 subject blocks derived from record ID integer division by 1,000. It reports PPG at 30 Hz, ECG at 1,000 Hz and ACC at 100 Hz, with 10-second windows and ACC available for 3,840 records. Verify the record-to-subject mapping against dataset metadata before making independent-subject claims.

The code shuffles subject blocks with seed 42 and uses 70%/15%/remainder for train/validation/test. The recorded counts are 2,742 / 548 / 598 records. Existing `split_ids.json` is reused, so archive it with any future checkpoint and validate its provenance. CUDA deterministic operations use warning-only settings; identical seeds do not promise bit-identical results.

## Interpretation caveats

- **Annotation conditioning:** training and evaluation supply annotation-derived motion group and quality. The separately evaluated ACC rule-based detector records 51.35% accuracy on 594 test records; its predictions are not substituted for annotations in the main reported model evaluation.
- **Ablation naming:** the “fixed BPF” or fixed-wavelet label is misleading in this source: the ablation selects group zero while retaining trainable filters. Interpret it as a shared-filter configuration rather than a frozen classical band-pass filter.
- **PTT auxiliary term:** predicted PTT is derived heuristically from predicted HR, not an independent PTT prediction head. ECG-derived quantities are used during preparation and auxiliary training.
- **Statistical reporting:** the raw log's suggested manuscript wording overstates the evidence against ResNet1D. The recorded paired Wilcoxon p-value is 0.31871. Values against DeepPPG and CNN–BiLSTM are 0.016772 and 0.015911; these are uncorrected per-record comparisons. Repeated records from the same subject and multiple comparisons require appropriate treatment before stronger inferential claims.
- **Baseline comparability:** the notebook contains adapted baseline implementations. TAPIR-style continuity carries `prev_hr` across loader records without resetting per subject; ordering may affect that comparison. Do not present these results as canonical published baseline scores.
- **Reference audit:** eleven large underestimation outliers belong to subject block 142. The log reports discrepancies between annotation HR and ECG-derived estimates; their cause remains unresolved. No records were removed or labels corrected during packaging.
- **Latency:** the log lists 816,445 parameters, CUDA latency 2.04 ± 0.11 ms/window and CPU latency 2.63 ± 0.21 ms/window. Hardware is unspecified; these figures are not evidence of measured on-phone performance.
- **Fallback behavior:** some signal-loading paths return zero signals or annotation fallbacks after exceptions. Audit failed reads before interpreting a new run; successful completion alone does not verify data integrity.
- **Saved outputs:** the notebook and log preserve historical output and prose, including the above caveats. Documentation takes precedence over unsupported wording in auto-generated suggested paper text.

## Before a reproducible release

Archive the environment lockfile, verified split IDs, checkpoint hashes, per-record predictions and hardware details from the same run. Reconcile the reference-HR outliers, verify subject IDs and baseline protocols, and evaluate predicted conditioning if claiming a fully automatic deployment pipeline. Match the source snapshot explicitly to a preprint version before describing it as an exact reproduction.
