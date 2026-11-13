# WavLM-Large mean-pooling baseline: strict cross-corpus six-class SER

Training source corpora: three; validation: speaker-disjoint recordings from each source;
testing: the entire fourth corpus, only once after validation checkpoint selection.

**Shared labels:** Neutral, Sad, Angry, Happy, Fear, Disgust

## Test results

| Held-out corpus | N | Accuracy | UAR | Macro F1 | Accuracy 95% CI |
|---|---:|---:|---:|---:|---:|
| RAVDESS | 1055 | 50.24% | 52.79% | 46.01% | [46.21%, 54.17%] |
| CREMA-D | 7434 | 36.14% | 35.41% | 32.36% | [34.50%, 37.82%] |
| EMODB | 454 | 69.38% | 70.42% | 67.98% | [65.03%, 73.58%] |
| EMOVO | 504 | 49.60% | 49.60% | 47.41% | [43.45%, 56.35%] |

## Reporting notes
- All percentages in this table are computed from original recordings, not audio windows.
- UAR is macro-average recall across six shared classes; macro-F1 likewise weights classes equally.
- CIs are speaker-cluster bootstraps, not seed-to-seed variability. For EMOVO (six speakers), expect wide CIs.
- Results across corpora reflect language, speakers, acoustics and label-subjectivity shifts simultaneously.
- This is **one fixed seed**, not a multi-seed estimate. Repeat with independent pre-registered seeds for variance.
- Validation speakers belong only to training-source corpora. No target-corpus adaptation, tuning or leakage.
- Experiments are not directly comparable to PME4 four-class, speaker-overlapping trials.
- Dataset-level labels may differ semantically despite harmonized emotion names.
- Excluded: RAVDESS calm/surprise, EMOVO surprise, EmoDB boredom.
- WavLM-Large initialized with pinned pretrained weights; only final 6 encoder layers are fine-tuned after frozen warmup; masked mean pooling and linear classifier are trained on source labels.

## Figures
1. figure_1_cross_corpus_metrics.png — accuracy, UAR and macro-F1 across the four target corpora.
2. figure_2_accuracy_with_speaker_bootstrap_ci.png — uncertainty with speaker-cluster resampling.
3. figure_3_per_class_recall_by_test_corpus.png — recall heatmap by class and corpus.
4. figure_4_per_class_f1_by_test_corpus.png — macro-class F1 heatmap.
5. figure_5_four_normalized_confusions.png — corpus-level normalized confusion matrices.
6. Each experiment contains learning_curves.png, source-validation plots, ROC/PR and reliability figures.
