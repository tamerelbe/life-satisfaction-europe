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

## Getting the data

The ESS Round 10 data is free to download from the [European Social Survey data portal](https://www.europeansocialsurvey.org/) after registration. It is not redistributed here.

## Tools

Python · pandas · scikit-learn · statsmodels · matplotlib / seaborn

---

*Author: Tamer Elbenghazi · [github.com/tamerelbe](https://github.com/tamerelbe)*
