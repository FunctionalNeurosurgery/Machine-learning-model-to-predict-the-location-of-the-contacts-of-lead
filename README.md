# Machine-learning-model-to-predict-the-location-of-the-contacts-of-lead
# DBS Electrode Localization Using Stimulation-Induced Side Effects and Machine Learning

## Overview

This repository contains the implementation of a machine learning framework for predicting three-dimensional (3D) Deep Brain Stimulation (DBS) electrode coordinates using stimulation-induced side effects (SISE).

The framework evaluates multiple machine learning models, including:

- Lasso Regression
- Random Forest
- XGBoost

A nested cross-validation strategy is used to ensure robust performance estimation. Model performance is evaluated using coordinate prediction accuracy and spatial localization error.

---

## Study Objective

Accurate DBS electrode localization is critical for optimal postoperative programming and therapeutic outcomes. Conventional imaging-based localization methods may be affected by imaging artifacts and anatomical variability.

This work investigates whether stimulation-induced side effects contain sufficient spatial information to predict electrode contact coordinates using machine learning.

---

## Dataset

The dataset consists of DBS stimulation records collected from:

- 73 DBS patients
- Stimulation parameters
- Electrode contact information
- Side-effect responses
- Ground-truth electrode coordinates

### Input Features

| Feature | Description |
|----------|------------|
| side | Brain hemisphere |
| Contact_pt | Electrode contact point |
| current_1 | First stimulation current |
| side_effect_1 | First side-effect response |
| current_2 | Second stimulation current |
| side_effect_2 | Second side-effect response |
| side_effect_diff | Absolute difference between side effects |
| current_diff | current difference |
| current_interaction | Product of currents |
| side_effect_interaction | Product of side effects |

### Target Variables

| Variable | Description |
|----------|------------|
| Medio_lateral_X | X coordinate (mm) |
| AP_Y | Y coordinate (mm) |
| supro_inferior_Z | Z coordinate (mm) |

---

## Methodology

### Data Preprocessing

- Removal of unnamed columns
- Side-effect category remapping
- Exclusion of invalid side-effect entries
- Contact point encoding
- Feature engineering

### Feature Engineering

Additional features:

```python
side_effect_diff
current_diff
current_interaction
side_effect_interaction
```

---

## Machine Learning Models

### 1. Lasso Regression

Used as a linear baseline model.

### 2. Random Forest

Ensemble tree-based regression model.

### 3. XGBoost

Gradient boosting model optimized for nonlinear spatial prediction.

---

## Model Evaluation

### Nested Cross-Validation

Outer loop:

- 10-fold cross-validation

Inner loop:

- 3-fold GridSearchCV

Purpose:

- Hyperparameter optimization
- Unbiased performance estimation

---

## Evaluation Metrics

### Coordinate Prediction Metrics

- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

### Spatial Localization Metric

Euclidean Distance Error

\[
Error = \sqrt{(x_{true}-x_{pred})^2 +
(y_{true}-y_{pred})^2 +
(z_{true}-z_{pred})^2}
\]

---

## Statistical Analysis

### Friedman Test

Used to compare multiple machine learning models across folds.

### Post-hoc Analysis

Wilcoxon signed-rank test with Bonferroni correction:

- XGB vs RF
- XGB vs Lasso
- RF vs Lasso

---

## Visualization

### Bland–Altman Analysis

Generated for:

- Medio-Lateral (X)
- Anterior-Posterior (Y)
- Superior-Inferior (Z)

and

- Euclidean distance prediction

### Violin Plot

Visualization of Euclidean localization errors across models.
