# Applied AI Image Classification

Clean, reproducible computer vision portfolio project for multiclass chest X-ray image classification.

This repository is a corrected portfolio version of an Applied AI learning project. It is not a copy of a coursework submission, and it does not include university briefs, answer sheets, grading feedback, raw reports, zipped submissions, restricted datasets or trained checkpoint files.

## Current Repository Status

This repository contains the cleaned project structure and reproducible training/evaluation pipeline. Dataset access and training outputs must be regenerated locally with a permitted dataset.

No fixed benchmark claim is made in this README because the dataset and trained model checkpoint are not included. Any metrics, confusion matrices and plots should be produced from the current code.

## Problem Statement

Chest X-ray image classification is a useful learning task for practising medical-image preprocessing, convolutional neural networks, transfer learning and careful model evaluation.

The project is framed as an educational portfolio workflow for classifying images into:

- Normal
- Opacity
- Pneumonia

It is not intended for diagnosis, clinical triage or medical decision-making.

## Why This Task Matters

Medical imaging models can appear strong when evaluated only with accuracy. This project therefore emphasises:

- reproducible training settings,
- class-aware evaluation metrics,
- balanced accuracy and macro F1 for imbalanced classes,
- confusion matrix interpretation,
- clear limitations and non-clinical use.

## Dataset

The original learning experiment used a three-class chest X-ray dataset with the following class balance:

| Class | Images |
|---|---:|
| Normal | 1,250 |
| Opacity | 1,125 |
| Pneumonia | 1,100 |
| Total | 3,475 |

The intended split is stratified:

| Split | Ratio |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Test | 15% |

The dataset is not included. To run the project, place a permitted image dataset in this folder layout:

```text
data/
  train/
    Normal/
    Opacity/
    Pneumonia/
  val/
    Normal/
    Opacity/
    Pneumonia/
  test/
    Normal/
    Opacity/
    Pneumonia/
```

## Experimental Protocol

The current pipeline uses these defaults:

| Setting | Value |
|---|---|
| Random seed | 42 |
| Image size | 224 x 224 |
| Batch size | 32 |
| Default custom CNN epochs | 30 |
| Default DenseNet121 epochs | User supplied, commonly shorter for transfer learning |
| Optimiser for custom CNN | AdamW |
| Optimiser for DenseNet121 | Adam |
| Model selection | Highest validation macro F1 |
| Final evaluation | Run once on the held-out test set using the selected checkpoint |

Class imbalance can be handled with optional inverse-frequency class weights:

```bash
python src/train.py --data-dir data --model custom_cnn --use-class-weights
```

## Preprocessing

Training images:

- convert to three-channel RGB,
- random resized crop to 224 x 224,
- mild rotation,
- mild brightness/contrast jitter,
- normalise with ImageNet mean and standard deviation.

Validation and test images:

- convert to three-channel RGB,
- resize to 224 x 224,
- normalise with ImageNet mean and standard deviation,
- no augmentation.

## Models

### Custom CNN

The implemented custom CNN uses five convolutional blocks:

- block channels: 3 -> 32 -> 64 -> 128 -> 256 -> 512,
- each block has two 3 x 3 convolution layers,
- batch normalisation and ReLU after each convolution,
- max pooling at the end of each block,
- dropout inside convolutional blocks,
- adaptive average pooling,
- dropout,
- final linear classifier for three classes.

### DenseNet121

The repository also supports DenseNet121 using torchvision pretrained ImageNet weights:

- `feature_extractor`: freeze DenseNet feature layers and train only the classifier head,
- `finetune`: train the full network end to end.

DenseNet121 results should only be reported after running the current training and evaluation scripts.

## Evaluation Metrics

The evaluation script reports:

- accuracy,
- balanced accuracy,
- macro precision,
- macro recall,
- macro F1,
- weighted F1,
- Cohen's kappa,
- macro ROC-AUC one-vs-rest when valid,
- class-wise precision/recall/F1/support,
- raw confusion matrix,
- normalised confusion matrix.

Balanced accuracy and macro F1 are prioritised because the classes are not perfectly balanced and class-level reliability matters.

## Results

No regenerated result table is committed in this corrected version because the dataset and trained checkpoints are not included.

After training and evaluation, the pipeline writes current outputs to your chosen output directory:

```text
outputs/
  best_model.pt
  training_history.json
  training_curves.png
  metrics.json
  confusion_matrix.png
  normalised_confusion_matrix.png
```

All plots should be generated from the current code and current experiment outputs. Old screenshots from the earlier coursework-style report were removed to avoid mismatched or unsupported results.

## Confusion Matrix Interpretation Guide

When you regenerate results, review:

- whether Opacity is confused with Normal,
- whether Pneumonia recall is stable,
- whether one class dominates the model's errors,
- whether balanced accuracy is meaningfully lower than standard accuracy,
- whether class-wise F1 reveals weaknesses hidden by aggregate accuracy.

## How To Run

Create an environment:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Train the custom CNN:

```bash
python src/train.py --data-dir data --model custom_cnn --epochs 30 --output-dir outputs/custom_cnn
```

Train DenseNet121 as a feature extractor:

```bash
python src/train.py --data-dir data --model densenet121 --strategy feature_extractor --epochs 10 --output-dir outputs/densenet_feature_extractor
```

Fine-tune DenseNet121:

```bash
python src/train.py --data-dir data --model densenet121 --strategy finetune --epochs 10 --output-dir outputs/densenet_finetune
```

Evaluate a checkpoint:

```bash
python src/evaluate.py --data-dir data --checkpoint outputs/custom_cnn/best_model.pt --model custom_cnn --output-dir outputs/custom_cnn
```

## Repository Structure

```text
applied-ai-image-classification/
  README.md
  requirements.txt
  src/
    config.py
    data.py
    evaluate.py
    models.py
    train.py
  notebooks/
    README.md
  reports/
    consistency_audit.md
    model_card.md
  results/
    README.md
```

## Limitations

- Educational portfolio project only; not clinically validated.
- Dataset and checkpoints are not included, so results must be regenerated locally.
- A single train/validation/test split is less robust than repeated splits or cross-validation.
- Medical-image datasets may contain scanner, site, demographic and labelling biases.
- No external validation dataset is currently included.
- Explainability methods such as Grad-CAM are not implemented yet.
- Model performance may change when the dataset source, split, seed, hardware or library versions change.

## Future Improvements

- Add a dataset preparation script for permitted public datasets.
- Add Grad-CAM visual explanations.
- Add repeated stratified split evaluation.
- Add calibration and confidence analysis.
- Add MLflow or Weights & Biases experiment tracking.
- Add a lightweight Streamlit demo with clear non-clinical disclaimers.

## Academic and Data Ethics

This public repository is designed as a portfolio-safe implementation. It intentionally excludes private university material and restricted data. Any future dataset use should respect licensing, consent, privacy and clinical governance requirements.
