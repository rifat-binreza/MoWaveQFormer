# Public implementation notes

The executable source is split into the [Python modules](../python/README.md). It derives from the executable research notebook supplied for this repository, rather than being inferred solely from paper prose. The previously published notebook remains a summary artifact.

The main architecture is retained: three banks of eight 31-tap filters, sigmoid gains, 30 ten-sample patches, scalar-quality-to-patch gating, four pre-normalized Transformer layers and the HR regression head. Checkpoint parameter names are unchanged. The legacy `hard_discard` flag has no forward effect. Motion routing uses activity annotations for training/evaluation; raw ACC rules are an optional separate utility.

Portability changes replace notebook global paths with arguments, save model configuration beside weights, validate subject split membership, and report missing/invalid labels explicitly. The dataset no longer invents an HR target when metadata fails. ACC tensors were omitted from training batches because the supplied model consumes their annotation-derived group rather than the raw axes. Missing ECG peak annotations disable PTT supervision for that record.

The supplied training defaults are retained, but full training was not repeated here. CPU verification covers parameter counts, exact forward parity against original notebook classes, gradients, checkpoint round-trip and one synthetic training epoch. The source-derived baselines have not been retrained. Dataset-level reproduction still requires the original data, split and training run. Published numerical evidence remains separate from any newly generated run outputs.
