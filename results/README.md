# Results Folder

This folder is intentionally empty in the committed repository.

Generate results locally with a permitted dataset:

```bash
python src/train.py --data-dir data --model custom_cnn --epochs 30 --output-dir outputs/custom_cnn
python src/evaluate.py --data-dir data --checkpoint outputs/custom_cnn/best_model.pt --model custom_cnn --output-dir outputs/custom_cnn
```

Expected generated artefacts:

- `training_history.json`
- `training_curves.png`
- `metrics.json`
- `confusion_matrix.png`
- `normalised_confusion_matrix.png`

Only plots generated from the current code should be used in the README, portfolio or CV.
