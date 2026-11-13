# Leave-one-corpus-out evaluation — plain WavLM-Large

| Held-out corpus | Speakers | Recordings | MSA (%) | Macro-F1 (%) | UAR (%) | Pooled Acc. (%) |
|---|---|---|---|---|---|---|
| EAV | 42 | 3360 | 49.73 | 43.84 | 49.73 | 49.73 |
| CREMA-D | 91 | 4897 | 59.19 | 55.42 | 57.9 | 59.18 |
| EmoDB | 10 | 339 | 85.23 | 84.74 | 85.1 | 85.25 |
| EMOVO | 6 | 336 | 49.4 | 45.84 | 49.4 | 49.4 |
| PME4 | 11 | 2189 | 59.42 | 59.36 | 59.47 | 59.48 |
| RAVDESS | 24 | 670 | 53.81 | 49.64 | 58.02 | 53.88 |

**Macro average across 6 held-out corpora**: MSA 59.46%, Macro-F1 56.47%, UAR 59.94%.

Test uses 100% of target-corpus usable four-class recordings, no target adaptation. Source train/val are speaker-disjoint within each source corpus. PME4 may have excluded missing/duplicate audio. The six folds have separate model initialization.
