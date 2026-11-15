# Data

This project uses the **European Social Survey (ESS) Round 10** integrated file (fieldwork 2020–2022).
The ESS data is free for non-commercial use but cannot be redistributed, so it is not included here.

1. Register at [europeansocialsurvey.org](https://www.europeansocialsurvey.org/) (free).
2. Download the Round 10 integrated file in CSV format.
3. Save it as `data/ESS10.csv`.

## Variables used

| Role | Variable | ESS code | Scale |
|------|----------|----------|-------|
| Outcome | Life satisfaction | `stflife` | 0–10 |
| Demographic | Age | `agea` | years |
| Demographic | Gender | `gndr` | 1 male, 2 female |
| Demographic | Education | `eduyrs` | years of full-time education |
| Socioeconomic | Household income | `hinctnta` | decile 1–10 |
| Socioeconomic | Employment relation | `emplrel` | employee / self-employed / other |
| Socioeconomic | Self-rated health | `health` | 1 very good – 5 very bad |
| Attitudinal | Interpersonal trust | `ppltrst` | 0–10 |
| Attitudinal | Trust in parliament | `trstprl` | 0–10 |
| Attitudinal | Trust in political parties | `trstprt` | 0–10 |
| Attitudinal | Most people try to be fair | `pplfair` | 0–10 |
| Attitudinal | Satisfaction with democracy | `stfdem` | 0–10 |
| Weight | Post-stratification weight | `pspwght` | |
| Grouping | Country | `cntry` | |

ESS codes such as 77 (refusal), 88 (don't know) and 99 (no answer) are treated as missing (see `src/config.py`).

*Citation: European Social Survey ERIC (2022). ESS Round 10 Data. https://www.europeansocialsurvey.org*
