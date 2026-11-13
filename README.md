# Speech Emotion Recognition: WavLM-Large and Hierarchical Fusion

Research code and output files for two speech-only models evaluated on six datasets.

## Models

- **Architecture A:** WavLM-Large + Grouped Hierarchical Fusion (GHF) + Dual-Path Temporal Pooling (DPTP) + supervised contrastive learning (SupCon).
- **Baseline:** WavLM-Large + masked mean pooling + classification head.

## Evaluations

| Protocol | Test targets | Runs |
|---|---|---:|
| Within-corpus (recording-disjoint, speaker-overlapping) | RAVDESS, CREMA-D, EMOVO, EmoDB, EAV, PME4 | 12 |
| Cross-corpus 3-to-1 (six-class ontology) | RAVDESS, CREMA-D, EMOVO, EmoDB | 8 |
| Cross-corpus 5-to-1 (four-class ontology) | RAVDESS, CREMA-D, EMOVO, EmoDB, EAV, PME4 | 12 |

**32 total runs (16 matched model pairs).** The 3-to-1 and 5-to-1 protocols have different class definitions; their scores must not be treated as a controlled source-data scaling study.

## Navigating the repository

- `01_within_corpus/<dataset>/<model>/source/`: code used for that experiment.
- `01_within_corpus/<dataset>/<model>/results/test/`: test metrics and recording predictions.
- `02_cross_corpus_3_to_1/<model>/source/`: 3-to-1 source code.
- `02_cross_corpus_3_to_1/<model>/results/test_<dataset>/test/`: 3-to-1 test results.
- `03_cross_corpus_5_to_1/<model>/source/`: 5-to-1 source code.
- `03_cross_corpus_5_to_1/<model>/results/folds/<dataset>/external_test/`: 5-to-1 test results.
- `04_paper_assets/`: consolidated paper tables and figures (where present).
- `REPRODUCIBILITY_AUDIT.json`: structural and recording-pair alignment verification.

This public export excludes model checkpoints, raw/processed audio, embedding matrices, large caches, archived ZIP files, and training logs. It **does not** reproduce the original dataset licenses or distribute the original speech datasets. Consult `DATASETS_AND_LIMITATIONS.md`.

## Results and interpretation

Report accuracy, unweighted average recall (UAR), and macro-F1 per corpus. The strongest within-corpus EAV result in the archived runs is 630/630; however the split is speaker- and conversation-group-overlapping and must **not** be presented as evidence of perfect unseen-speaker or independent-conversation generalization. A checkpoint-provenance discrepancy in the available EAV records also requires review.

## Reproducibility

Training code reflects separate experiment generations and environments. Examine each script's arguments and its `resolved_config.json` / `experiment_config.json`; paths in original scripts refer to server-specific dataset locations. Obtain data from the authorized original sources, use the documented splits, and verify compatible PyTorch/Transformers versions before attempting reruns.

The CSV split/prediction files in this **public export** may have absolute local-audio path columns removed. Source experiment files and the private complete archive are not modified.

## Publication and attribution

Authors, in manuscript order: Poonam Rani, Avyakt Jain, Snehil Tripathi (Department of Computer Science, Netaji Subhas University of Technology, Delhi, India).

Use a permanent tagged release (and ideally an archive DOI) for the paper's *Code and Data Availability* link. Repository license and dataset source mirrors must be approved by the authors before publication. No license is automatically assigned to external datasets or third-party code.
