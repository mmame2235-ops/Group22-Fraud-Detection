# Group 22 — Module 6 Submission Package

**Project:** AI-Based Credit Card Fraud Detection Using Transaction Data  
**Course:** FUSE AI-201 — Module 6  
**Team:** Yihays Fente Tarekegn; Mohammed Tiku Tareke; Abraham Alem Tafes; Asnakew Mekonnen Tegegne.

## Deliverables
- `Deliverable_A_Project_Portfolio.docx` — portfolio, evaluation, deployment, monitoring and maintenance.
- `Deliverable_B_Video_Demonstration_Script.docx` — timed script for a demonstration of up to 10 minutes.
- `Deliverable_C_Project_Retrospective.docx` — one-page retrospective draft.
- `Deliverable_D_Data_Provenance_and_Synthetic_Data_Statement.docx` — data source and synthetic-data statement.
- `Group22_Module5_Implementation_Notebook.ipynb` — supplied Module 5 notebook copy, if present.
- `recorded_test_metrics.csv` — metrics recorded in the project notebook.

## Recorded results to preserve accurately
Selected model: XGBoost; validation PR-AUC: 0.8072; validation-selected threshold: approximately 0.9331. Test accuracy 0.9995; precision 0.8977; recall 0.7980; F1 0.8449; PR-AUC 0.8557; ROC-AUC 0.9763. Confusion matrix [TN, FP, FN, TP] = [56854, 9, 20, 79]. Test recall is below the proposed 0.85 target; do not claim that the target was met.

## Final checks required before upload
1. Run the notebook from top to bottom and confirm all results.
2. Verify the dataset's exact download date, source, current license/terms and attribution.
3. Verify the actual contributions of each named member; names were supplied, but individual roles were not confirmed.
4. Record the video using real notebook outputs. The optional Flask API is a local demo only; do not claim it is publicly deployed.
5. Generate/attach any model artifact or screenshots only if actually created and required.
6. Review the retrospective and data statement as a team before signing/submission.

## Reproducibility
Use a compatible Jupyter/Colab environment, install the packages listed in the notebook, obtain `creditcard.csv` from the stated Kaggle source subject to its terms, place it where the notebook expects it, and execute cells in order. Confirm outputs before submission.

**Evidence limitation:** This package documents the recorded notebook outputs and the project proposal. It does not claim a live production deployment, a running public API, an independent security audit, or a verified separate model artifact.
