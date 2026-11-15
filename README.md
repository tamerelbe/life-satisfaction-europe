# What Predicts Life Satisfaction in Europe?

**Bachelor thesis · IE University · Data & Business Analytics · 2026 · Grade 8.2/10**

## The question

What actually drives how satisfied people in Europe are with their lives: money, health, relationships, or trust in institutions?

## The data

- **Source:** European Social Survey (ESS), Round 10
- **Size:** 34,117 respondents across European countries
- **Outcome:** self-reported life satisfaction (0–10 scale)

## The approach

1. **Cleaning and exploration** of the survey data in Python
2. **Four predictive models**, compared side by side:
   - OLS regression
   - Ordinal logistic regression
   - Random Forest
   - Gradient Boosting
3. **Outcome transformation:** a rank-based normal transformation of life satisfaction, to check results don't depend on treating a 0–10 scale as continuous
4. **Segmentation:** K-Means clustering to group respondents into well-being profiles

## Key findings

- **Health and satisfaction with democracy are stronger predictors of life satisfaction than income** — in every one of the four models.
- Clustering identified **four distinct well-being profiles** across Europe.
- The results point to **four policy directions** for improving well-being beyond income growth.

## Results (test set)

| Model | Outcome | RMSE | R² |
|-------|---------|------|----|
| OLS regression | Raw (0–10) | 1.808 | 0.239 |
| Ordinal logistic | Raw (ordered) | 1.808 | 0.239 |
| Random Forest | Latent score | 0.823 | 0.242 |
| Gradient Boosting | Latent score | 0.825 | 0.239 |

All four models land at R² ≈ 0.24, so the main relationships are close to linear and additive. Permutation importance ranks **self-rated health, satisfaction with democracy, income, fairness and trust in parliament** as the top five predictors.

## Repository structure

```
├── run_all.py                     # runs every step in order
├── src/
│   ├── config.py                  # paths, seed, variables, model settings
│   └── data.py                    # loading, missing-code recoding, imputation, rank-normal transform
├── scripts/
│   ├── 01_data_cleaning.py        # analytic sample + missingness report
│   ├── 02_eda.py                  # descriptives and exploratory figures
│   ├── 03_rank_transformation.py  # latent outcome and category thresholds
│   ├── 04_regression_models.py    # OLS (country-clustered SEs) + ordinal logistic
│   ├── 05_ml_models.py            # Random Forest + Gradient Boosting, permutation importance
│   ├── 06_clustering.py           # K-Means: elbow + silhouette, cluster profiles
│   └── 07_robustness_checks.py    # composite trust, raw outcome, gender, complete cases, weights, VIF
├── data/README.md                 # how to get the ESS file (not redistributed)
├── figures/                       # generated charts
└── output/                        # generated tables (CSV)
```

## How to run

1. Download the ESS Round 10 integrated file as CSV from the [European Social Survey data portal](https://www.europeansocialsurvey.org/) (free registration) and save it as `data/ESS10.csv`. The data is not redistributed here.
2. Install the requirements and run the pipeline:

```bash
pip install -r requirements.txt
python run_all.py
```

Every step uses a fixed random seed (42) and an 80/20 train-test split. Tables are written to `output/` and charts to `figures/`.

## Tools

Python · pandas · scikit-learn · statsmodels · SciPy · matplotlib / seaborn

---

*Author: Tamer Elbenghazi · [github.com/tamerelbe](https://github.com/tamerelbe)*
