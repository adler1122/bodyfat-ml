# Body Fat Percentage Predictor

A machine learning project that predicts body fat percentage from simple body circumference measurements, with a local web app interface. Trained separately for male and female subjects using real density-based body fat measurements as ground truth.

---

## Overview

The project takes a practical approach to body fat estimation: instead of using the US Navy circumference formula directly, a **Linear Regression model** is trained on actual body density measurements (converted to body fat % via Siri's equation — the same method used as a reference standard in clinical settings), giving a data-driven prediction that outperforms simple formula-based approaches.

Two separate models are trained — one for males, one for females — because body fat distribution differs significantly between sexes, and a combined model washes out within-group signal (Simpson's paradox observed during correlation analysis).

A **US Navy formula** is also computed and included as a feature/comparison column to assess how well the formula tracks the measured values (correlation: ~0.87 for males, ~0.79 for females, with a consistent overestimation bias).

---

## Dataset

**Source:** [Body Fat Extended Dataset — simonezappatini (Kaggle)](https://www.kaggle.com/datasets/simonezappatini/body-fat-extended-dataset)

- 436 subjects (252 male, 184 female)
- Body fat % derived from underwater weighing (hydrostatic weighing) via Siri's equation — the DXA-equivalent gold standard for this type of dataset
- Measurements: Age, Weight, Height, Neck, Chest, Abdomen, Hip, Thigh, Knee, Ankle, Biceps, Forearm, Wrist

> **Note:** `Density` is excluded from features (it is a 1:1 transform of the target via Siri's equation — pure leakage). `NavyBodyFat` is excluded from features (derived from the same measurements — circular). Both are retained in the engineered CSV for reference only.

---

## Project Structure

```
bodyfat-ml/
├── BodyFat.csv                     # raw dataset from Kaggle
├── BodyFat_engineered.csv          # preprocessed dataset with NavyBodyFat column
├── pre_process.py                  # preprocessing: drops indicator column, converts units, adds NavyBodyFat
├── feature_engineering.py          # correlation analysis, overall and split by sex
├── linear_regression_final.py      # model training, learning curve diagnostics, saves .pkl files
├── model_male.pkl                  # trained pipeline (StandardScaler + LinearRegression) for males
├── model_female.pkl                # trained pipeline (StandardScaler + LinearRegression) for females
├── app.py                          # Flask web app backend
└── templates/
    └── index.html                  # frontend: measurement inputs, prediction, lean mass & goal calculator
```

---

## Methodology

### Preprocessing (`pre_process.py`)
- Drops the dataset's "in original dataset" indicator column (no predictive value)
- Converts `Height` from meters to cm for unit consistency
- Adds `NavyBodyFat` column using the US Navy circumference formula (imperial conversion applied internally):
  - **Male:** `86.010 × log10(Abdomen − Neck) − 70.041 × log10(Height) + 36.76`
  - **Female:** `163.205 × log10(Waist + Hip − Neck) − 97.684 × log10(Height) − 78.387`

### Feature Selection
Pearson correlation with `BodyFat` was computed overall and split by sex. Key findings:
- The all-rows correlations were misleadingly weak (e.g. Abdomen: +0.36) compared to within-sex correlations (Abdomen: +0.81 male, +0.74 female) — confirming sex-stratified modelling is necessary
- `Age` showed the largest male vs. female difference (+0.29 male, −0.06 female)
- `Height` was essentially useless for both sexes

**Final feature sets (6 each — practical for home measurement with a tape measure and scale):**

| Sex    | Features |
|--------|----------|
| Male   | Abdomen, Chest, Hip, Weight, Thigh, Neck |
| Female | Abdomen, Hip, Weight, Biceps, Thigh, Forearm |

Reducing from 13 to 6 features cost less than 0.002 R² — effectively zero accuracy loss.

### Model Selection
10 model families were compared using 10-fold cross-validation. Linear models consistently outperformed all nonlinear models (RandomForest, GradientBoosting, ExtraTrees, SVR, KNN, DecisionTree) on both sexes — consistent with the underlying physiology (body fat formulas are inherently linear/log-linear combinations of circumferences). Regularization (Ridge/Lasso/ElasticNet) offered marginal improvement (<0.02 R²) over plain Linear Regression, so **LinearRegression** was chosen for simplicity and interpretability.

**Final model performance (10-fold CV):**

| Sex    | R² (mean ± std) | MAE   | RMSE  |
|--------|-----------------|-------|-------|
| Male   | 0.679 ± 0.089   | 3.69  | 4.50  |
| Female | 0.516 ± 0.234   | 2.87  | 3.65  |

### Overfitting / Underfitting Diagnosis
Learning curves (training R² vs CV R² across training set sizes) showed:
- **Male:** gap of ~0.05 at full training size, curves converge cleanly → good fit
- **Female:** gap of ~0.11, CV curve still slightly rising at n=184 → slightly underfit due to limited data, not overfitting

Neither model shows signs of overfitting. The female model's higher variance is attributed to smaller sample size and genuinely more complex fat distribution patterns in women.

---

## Web App

Built with **Flask** (backend) and plain HTML/CSS/JS (frontend). No database, no user data, no external API calls.

### Run locally

```bash
pip install flask pandas scikit-learn
python linear_regression_final.py   # trains models and saves .pkl files
python app.py                        # starts server and opens browser automatically
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000)

### Features
- Male / Female toggle (swaps feature inputs and loads the correct model)
- Input sliders/fields for the 6 relevant measurements
- Displays predicted body fat %, reference category (Essential / Athletic / Fitness / Average / Above average)
- **Lean mass & goal calculator panel** — shows current fat mass and lean mass, lets you enter a target body fat %, and calculates exactly how many kg of fat to gain or lose to reach it (lean mass assumed constant)

> Lean mass formula: `lean_mass = weight × (1 − bf% / 100)`
> Goal weight: `goal_weight = lean_mass / (1 − target_bf% / 100)`

---

## Future Work

- **CNN model** for body fat estimation from photos (separate project branch — dataset sourcing in progress)
- **Flutter mobile app** — Android + iOS, fully offline, hardcoded linear coefficients (no server required)
- More data, especially female subjects, to reduce CV variance for the female model

---

## Dependencies

```
pandas
numpy
scikit-learn
flask
matplotlib
```

Install all:
```bash
pip install pandas numpy scikit-learn flask matplotlib
```

---

## Disclaimer

This is a personal ML project. Predictions are estimates based on circumference measurements and should not be used as a substitute for clinical body composition assessment.
