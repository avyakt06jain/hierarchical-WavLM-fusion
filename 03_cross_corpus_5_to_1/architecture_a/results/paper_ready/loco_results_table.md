# Leave-one-corpus-out evaluation — WavLM-Large + GHF + DPTP + SupCon

| Held-out corpus | Speakers | Recordings | MSA (%) | Macro-F1 (%) | UAR (%) | Pooled Acc. (%) |
|---|---|---|---|---|---|---|
| EAV | 42 | 3360 | 46.34 | 40.52 | 46.34 | 46.34 |
| CREMA-D | 91 | 4897 | 61.87 | 60.49 | 61.35 | 61.87 |
| EmoDB | 10 | 339 | 96.02 | 95.44 | 95.65 | 95.87 |
| EMOVO | 6 | 336 | 57.14 | 56.25 | 57.14 | 57.14 |
| PME4 | 11 | 2189 | 58.2 | 58.27 | 58.33 | 58.34 |
| RAVDESS | 24 | 670 | 62.77 | 62.28 | 65.73 | 62.84 |

**Macro average across 6 held-out corpora**: MSA 63.73%, Macro-F1 62.21%, UAR 64.09%.

Test uses 100% of target-corpus usable four-class recordings, no target adaptation. Source train/val are speaker-disjoint within each source corpus. PME4 may have excluded missing/duplicate audio. The six folds have separate model initialization.
