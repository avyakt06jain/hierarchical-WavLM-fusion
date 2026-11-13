# Within-corpus per-speaker 70/15/15 (speaker-overlapping) benchmark

Architecture: baseline
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
| ravdess | 743 / 167 / 145 | 65.52% | 65.33% | 62.47% |
| crema-d | 5171 / 1089 / 1174 | 65.84% | 66.21% | 65.50% |
| emodb | 319 / 68 / 67 | 65.67% | 67.87% | 67.02% |
| emovo | 354 / 78 / 72 | 54.17% | 54.17% | 54.39% |
