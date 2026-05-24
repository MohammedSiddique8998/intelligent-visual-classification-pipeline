# Model Card

## Model Family

Three-class chest X-ray image classification using:

- a custom convolutional neural network,
- optional DenseNet121 feature extraction,
- optional DenseNet121 fine-tuning.

## Intended Use

This project is intended for educational and portfolio demonstration purposes. It shows how to structure a reproducible image-classification workflow, evaluate imbalanced multiclass performance and document limitations.

## Not Intended Use

This model is not intended for:

- clinical diagnosis,
- medical triage,
- patient management,
- replacing radiologists or clinicians,
- use on real patient data without governance, validation and approval.

## Classes

- Normal
- Opacity
- Pneumonia

## Data Limitations

The dataset is not included in this repository. The original learning experiment used a moderate-size three-class chest X-ray dataset, but public users must supply their own permitted data.

Known risks include:

- class imbalance,
- label noise,
- scanner and hospital-site bias,
- demographic bias,
- distribution shift between datasets,
- limited generalisation without external validation.

## Training Settings

The current code documents and stores:

- random seed,
- image size,
- batch size,
- epoch count,
- learning rate,
- weight decay,
- class-weight setting,
- model type and DenseNet strategy.

## Metrics

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

Balanced accuracy and macro F1 should be reviewed alongside standard accuracy because the classes are not perfectly balanced.

## Ethical Considerations

Medical imaging models can create harm if used outside their validated context. False negatives may miss serious disease; false positives may cause unnecessary anxiety or follow-up. This project should be treated as a learning artefact only.

## Failure Risks

- Mild opacity may be confused with normal images.
- Model confidence may be poorly calibrated.
- Performance may fall on external datasets.
- Artefacts, acquisition settings or labels may influence predictions.

## Future Improvements

- External validation on a permitted dataset.
- Grad-CAM or similar explainability.
- Calibration and uncertainty analysis.
- Bias and subgroup analysis where ethically available.
- Repeated stratified splits or cross-validation.
- Clear deployment guardrails for any demo interface.
