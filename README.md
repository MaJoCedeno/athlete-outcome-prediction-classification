# Athlete Competition Outcome Prediction

Predicting whether an athlete achieves a winning outcome in a competition, using sociodemographic, health, and training profile data. Built as part of the Predictive Methods of Data Mining (PMDM) course at Nova IMS, structured as a Kaggle-style classification competition.

## Problem

Given an athlete's profile — training regimen, health history, sociodemographic background, and competition context — predict a binary outcome: **Winner** vs **No Winner**. The goal was to identify which factors most influence competitive success and build a model that generalizes well to unseen athletes.

## Data

- **18,055 records, 30 raw features** (32 after encoding), spanning four domains:
  - **Sociodemographic:** age group, education, income, region, sex, disability
  - **Health:** past injuries, physiotherapy, recovery, supplements
  - **Competition:** athlete score, edition, previous attempts, enrollment status
  - **Training:** cardiovascular, strength, sand, plyometric, sport-specific training, mental preparation, coaching status
- Delivered as a train/test split with a held-out Kaggle-style test set for final submission scoring.

## My Contribution

I led data cleaning, feature engineering, feature selection (RFE), and model development/selection (KNN, Decision Trees, Random Forest) as part of a 3-person team. Teammates contributed the ANOVA and Chi-Square statistical feature selection methods.

## Approach

1. **Data Cleaning** — handled missing values with domain-aware imputation (e.g., forward/backward-filling per athlete, mode-based imputation for categorical fields), resolved data entry errors (e.g., miscoded competition years), removed duplicates.
2. **Outlier Treatment** — winsorized extreme values in continuous training/health variables to reduce distortion from data entry errors while preserving distribution shape.
3. **Feature Engineering & Encoding** — ordinal/binary mapping for ordered categories, one-hot encoding for nominal variables (e.g., region), standardization (Z-score) for continuous features.
4. **Feature Selection** — compared filter methods (ANOVA, Chi-Square) against wrapper methods (Recursive Feature Elimination). RFE identified an optimal set of **14 features**, achieving a **0.88 cross-validation score** — the strongest feature-selection result of the methods tested.
5. **Modeling** — benchmarked against ZeroR/OneR baselines, then compared:
   - Naive Bayes (Gaussian, Bernoulli)
   - K-Nearest Neighbors (tuned across k, including distance-weighted variants)
   - Decision Trees (tuned for max depth and minimum samples split)
   - Random Forest
6. **Model Selection** — Decision Tree with selected features reached **0.81 validation accuracy**. After comparing all candidates on validation performance, **KNN (k=34)** was selected as the final model and used to generate predictions for the held-out Kaggle test set.

## Results

| Model | Validation Score |
|---|---|
| RFE feature subset (14 features) | 0.88 (cross-val) |
| Decision Tree (selected features) | 0.81 |
| **KNN (k=34) — final submitted model** | Used for final Kaggle submission |

## Tools

Python (pandas, NumPy, scikit-learn), matplotlib/seaborn for visualization, Google Colab.

## What I'd Improve

- Complete the Random Forest hyperparameter tuning (estimator count, max depth) — it was fit but not fully benchmarked against KNN before final model selection.
- Add cross-validated confidence intervals rather than single train/validation splits, to better assess model stability.
- Test ensemble/stacking approaches combining KNN and Decision Tree, given their complementary error patterns.
