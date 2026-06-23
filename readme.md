# Body Fat Percentage Predictor

A machine learning project that predicts body fat percentage from simple body circumference measurements, with a local web app interface. Trained separately for male and female subjects using real density-based body fat measurements as ground truth.

---

## Overview

Instead of using the US Navy circumference formula directly, a **Linear Regression model** is trained on actual body density measurements (converted to body fat % via Siri's equation — the same method used as a reference standard in clinical settings), giving a data-driven prediction calibrated on real measurements.

Two separate models are trained — one for males, one for females — because body fat distribution differs significantly between sexes, and a combined model washes out within-group signal (Simpson's paradox observed during correlation analysis).

A **US Navy formula** column is also computed and included for comparison to assess how well the formula tracks the measured values (consistent overestimation bias observed, worse for females than males).

---

## Dataset

**Source:** [Body Fat Extended Dataset — simonezappatini (Kaggle)](https://www.kaggle.com/datasets/simonezappatini/body-fat-extended-dataset)

- 436 subjects (252 male, 184 female)
- Body fat % derived from underwater weighing (hydrostatic weighing) via Siri's equation
- Measurements: Age, Weight, Height, Neck, Chest, Abdomen, Hip, Thigh, Knee, Ankle, Biceps, Forearm, Wrist

> `Density` is excluded from features — it is a 1:1 transform of the target via Siri's equation (pure leakage). `NavyBodyFat` is also excluded from features — it is derived from the same input measurements (circular). Both columns are retained in the engineered CSV for reference only.

---

## Project Structure

```
bodyfat-ml/
├── BodyFat.csv               # raw dataset downloaded from Kaggle
├── BodyFat_engineered.csv    # preprocessed dataset with NavyBodyFat added
├── data.py                   # Kaggle dataset download script
├── preprocessing.ipynb       # full preprocessing and EDA notebook:
│                             #   unit conversion, Navy formula, correlation
│                             #   analysis by sex, model comparison, learning curves
├── train&save.py             # trains final models on full dataset, saves .pkl files
├── model_male.pkl            # trained pipeline (StandardScaler + LinearRegression) for males
├── model_female.pkl          # trained pipeline (StandardScaler + LinearRegression) for females
├── app.py                    # Flask web app (auto-opens browser on run)
└── templates/
    └── index.html            # frontend: inputs, prediction, lean mass & fat goal calculator
```

---

## Methodology

### Preprocessing (`preprocessing.ipynb`)

- Downloads dataset via `data.py` (uses Kaggle API, saves as `BodyFat.csv`)
- Drops the "in original dataset" indicator column (no predictive value)
- Fixes unit inconsistency: `Height` was stored in meters, converted to cm for consistency with circumference columns
- Adds `NavyBodyFat` using the US Navy circumference formula:
  - **Male:** `86.010 × log10(Abdomen − Neck) − 70.041 × log10(Height) + 36.76`
  - **Female:** `163.205 × log10(Waist + Hip − Neck) − 97.684 × log10(Height) − 78.387`
  - All measurements converted to inches before applying the formula

### Feature Selection

Pearson correlation with `BodyFat` was computed overall and split by sex. Key findings:

- All-rows correlations were misleadingly weak (Abdomen: +0.36 overall vs +0.81 male-only, +0.74 female-only) — confirming sex-stratified modelling is necessary
- `Age` showed the largest sex difference: +0.29 for males, −0.06 for females
- `Height` was essentially useless for both sexes (r < 0.1)

**Final feature sets — 6 measurements each, practical to take at home with a tape measure and scale:**

| Sex    | Features |
|--------|----------|
| Male   | Abdomen, Chest, Hip, Weight, Thigh, Neck |
| Female | Abdomen, Hip, Weight, Biceps, Thigh, Forearm |

Reducing from 13 to 6 features cost less than 0.002 R² — effectively zero accuracy loss.

### Model Selection

10 model families were compared using 10-fold cross-validation (in `preprocessing.ipynb`). Results:

- Linear models (LinearRegression, Ridge, Lasso, ElasticNet) consistently outperformed all nonlinear models on both sexes
- Tree-based models (RandomForest, GradientBoosting, ExtraTrees) and distance/kernel models (KNN, SVR) underperformed — dataset is too small (~250/180 rows per sex) for them to learn nonlinear patterns over noise
- Regularization (Ridge/Lasso/ElasticNet) offered <0.02 R² improvement over plain LinearRegression — not worth the added complexity

**→ Plain LinearRegression chosen for both models.**

**Final performance (10-fold CV):**

| Sex    | R² (mean ± std) | MAE   | RMSE  |
|--------|-----------------|-------|-------|
| Male   | 0.679 ± 0.089   | 3.69  | 4.50  |
| Female | 0.516 ± 0.234   | 2.87  | 3.65  |

### Overfitting / Underfitting Check

Learning curves were generated in `preprocessing.ipynb` (training R² vs CV R² across training sizes):

- **Male:** ~0.05 gap at full training size, curves converge cleanly → good fit
- **Female:** ~0.11 gap, CV curve slightly still rising at n=184 → mild underfitting from limited data, not overfitting

Neither model overfits. Female model variance is higher due to smaller sample size and more complex fat distribution patterns.

---

## Running the App

```bash
pip install flask pandas scikit-learn numpy matplotlib kagglehub

# Step 1: download dataset (requires Kaggle API token in ~/.kaggle/kaggle.json)
python data.py

# Step 2: run preprocessing notebook to generate BodyFat_engineered.csv
# Open preprocessing.ipynb in VS Code or Jupyter and run all cells

# Step 3: train models and save .pkl files
python "train&save.py"

# Step 4: launch web app (opens browser automatically)
python app.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000)

### App Features

- Male / Female toggle — swaps input fields and loads the correct model
- 6 number inputs per sex with sensible defaults
- Predicts body fat % with a reference category (Essential / Athletic / Fitness / Average / Above average)
- **Lean mass & goal calculator** — appears after prediction, shows fat mass and lean mass, lets you type a target body fat % and calculates exactly how many kg of fat to gain or lose:

```
lean_mass  = weight × (1 − bf% / 100)
goal_weight = lean_mass / (1 − target_bf% / 100)
fat_change  = goal_weight − current_weight
```

---

## Dependencies

```
pandas
numpy
scikit-learn
flask
matplotlib
kagglehub
```

---

## Future Work

- CNN model for body fat estimation from images (dataset sourcing in progress)
- Flutter mobile app — Android + iOS, fully offline, model coefficients hardcoded as a formula

---

## Disclaimer

Personal ML project. Predictions are estimates based on circumference measurements and should not be used as a substitute for clinical body composition assessment.