# Within-corpus per-speaker 70/15/15 (speaker-overlapping) benchmark

Architecture: architecture_a
Label ontology: Neutral, Sad, Angry, Happy, Fear, Disgust
All speakers occur in train, validation and test; only recordings are mutually disjoint.
Per-speaker splits are 70/15/15 rounded to whole recordings; exact percentages are integer-limited.
See per-corpus actual fractions, immutable manifests, and excluded audio audit.
The test set is evaluated once after validation-only checkpoint selection.
Confidence intervals are speaker-cluster bootstrap intervals (dependent on number of distinct speakers).
Only one random split and seed are reported unless experiments are independently repeated.

## Completed tests

| Corpus | Train / Val / Test | Accuracy | UAR | Macro F1 |
|---|---|---:|---:|---:|
| ravdess | 743 / 167 / 145 | 84.14% | 84.08% | 84.02% |
| crema-d | 5171 / 1089 / 1174 | 81.43% | 81.61% | 81.37% |
| emodb | 319 / 68 / 67 | 97.01% | 96.67% | 96.63% |
| emovo | 354 / 78 / 72 | 87.50% | 87.50% | 88.13% |
