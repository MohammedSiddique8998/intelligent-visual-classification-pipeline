# Consistency Audit

## Issues Found

- The first portfolio version reported fixed test metrics while the dataset and trained checkpoint were not included.
- Old plot images were copied from previous report assets rather than produced from the current repository scripts.
- The README presented DenseNet121 and custom CNN result comparisons without shipping the exact experiment outputs needed to verify them.
- The model card gave a specific performance claim without enough local evidence to reproduce it.
- The evaluation script did not report enough imbalance-aware metrics for a three-class medical-imaging problem.
- The training script selected checkpoints by validation accuracy rather than macro F1, which is less suitable when class balance matters.

## Fixes Completed

- Removed old screenshots that could not be proven to match the current code.
- Rewrote README to state that results must be regenerated locally because the dataset and checkpoints are not included.
- Rewrote model card to remove unsupported benchmark claims and clarify non-clinical use.
- Updated training to track accuracy, balanced accuracy and macro F1.
- Updated checkpoint selection to use validation macro F1.
- Added optional class-weighted loss for imbalanced training data.
- Expanded evaluation to include balanced accuracy, macro precision, macro recall, macro F1, weighted F1, Cohen's kappa, optional macro ROC-AUC, class-wise metrics, raw confusion matrix and normalised confusion matrix.
- Added generated plot outputs from current code: `training_curves.png`, `confusion_matrix.png` and `normalised_confusion_matrix.png`.

## Current Truth Standard

The repository now separates implementation from unverified historical results. A recruiter or reviewer can inspect the code, run it with a permitted dataset and generate metrics/plots directly from the current pipeline.
