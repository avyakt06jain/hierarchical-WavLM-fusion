# Before making this repository public

1. Read all 21 Python files, configuration JSON files, and CSV data tables for hard-coded tokens, passwords, private paths, identifiable recording metadata, and redistribution restrictions.
2. Confirm all 16 paired experiments share the intended original-recording test IDs and true labels (see `REPRODUCIBILITY_AUDIT.json`).
3. Confirm the six-dataset 5-to-1 ontology and four-dataset 3-to-1 ontology separately; never claim the difference isolates source-corpus quantity.
4. Investigate the EAV 630/630 result's conversation-group overlap and inconsistent checkpoint epoch/provenance records before describing it as confirmatory generalization.
5. Specify each experiment's software environment and how to obtain the six original corpora legally. Kaggle mirrors are third-party copies; their URLs and licensing should be checked.
6. Decide on a license for original code only. This archive does not grant permissions for third-party dataset material or copied model weights.
7. Add a tagged GitHub release (for example `v1.0-paper`) and cite its immutable release URL or DOI in the paper's Data/Code Availability section.
8. **Do not embed the entire ZIP in the PDF.** Add a clickable, versioned GitHub/DOI URL to the availability section.

Large model checkpoint files are intentionally omitted. Retain a separate private archive or selectively publish approved checkpoints via an external artifact store / GitHub Release / Git LFS.
