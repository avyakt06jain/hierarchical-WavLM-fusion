# Full four-corpus data quality audit

- Audit policy: `FULL_SCAN_QUARANTINE_AMBIGUOUS_GROUPS_V2`
- Input: `/workspace/data`
- Total valid eligible before group checks: **10510**
- Retained: **9447**
- Rejected invalid audio/filenames: **2**
- Safe redundant copies removed: **1057**
- Ambiguous recordings quarantined: **6**, in **3** groups

## Per corpus

| Corpus | Scanned paths | Excluded labels | Invalid | Benign removed | Conflict quarantined | Retained |
|---|---:|---:|---:|---:|---:|---:|
| ravdess | 2880 | 768 | 0 | 1057 | 0 | 1055 |
| crema-d | 7442 | 0 | 2 | 0 | 6 | 7434 |
| emodb | 535 | 81 | 0 | 0 | 0 | 454 |
| emovo | 588 | 84 | 0 | 0 | 0 | 504 |

## Integrity policy

Exact repeat, same corpus/speaker/emotion: retain one. Conflicting emotion, speaker, corpus, or repeated recording identity with different audio: quarantine **the complete connected group**, never guess the correct label. Compare SHA-256 of WAV bytes and decoded mono PCM at the original sample rate. These checks do not detect near-duplicates, audio with different sample rates, or semantic annotation mistakes without repeated audio.

Files: `full_scan_inventory.csv`, `rejected_audio.csv`, `duplicate_audio.csv`, `duplicate_groups.csv`, `quarantined_audio.csv`, `class_counts_before_after_dedup.csv`, `duplicate_group_details.json`.

## Warnings

- ravdess: 50.0% of eligible files excluded by duplicates/quarantine. Check quality and report exclusion bias.

## Per-speaker recording-disjoint train/val/test viability

- Pass: every speaker has recordings in each partition; all recordings are disjoint and all six emotions are covered.
