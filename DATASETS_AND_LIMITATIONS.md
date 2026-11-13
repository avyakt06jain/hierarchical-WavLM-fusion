# Dataset sources and reporting constraints

| Corpus | Typical source used by this project | Redistribution advice |
|---|---|---|
| RAVDESS | Kaggle copy of original RAVDESS | Link both the original publication and exact Kaggle source; check their licenses. |
| CREMA-D | Kaggle copy of original CREMA-D | Link both the original publication and exact Kaggle source; check their licenses. |
| EmoDB | Kaggle copy of original EmoDB | Link the original corpus and the exact Kaggle mirror; check licenses. |
| EMOVO | Kaggle copy of original EMOVO | Link the original corpus and the exact Kaggle mirror; check licenses. |
| EAV | Official dataset source | Do not redistribute raw participant recordings unless explicitly permitted. |
| PME4 | Official dataset source | Do not redistribute raw participant recordings unless explicitly permitted. |

This repository contains derived evaluation records, **not original licensed speech data**. Dataset source URLs were not included in the supplied file listing and must be confirmed by the authors rather than guessed.

## Evaluation limitations

- Within-corpus: per-speaker recording splits; speaker overlap across training/validation/test. This is not held-out-speaker evaluation.
- EAV within-corpus: an audit found all 630 test recordings share subject-conversation groups with training; see manuscript limitations.
- EAV Architecture A checkpoint provenance requires review (the archived test record and training-completion record may reference different best epochs/checkpoint hashes).
- Cross-corpus 3-to-1: six classes; cross-corpus 5-to-1: four classes, so raw protocol scores are not directly comparable.
- Prediction CSVs preserve recording IDs, but some absolute source file path fields are omitted from this public copy for portability and privacy.
