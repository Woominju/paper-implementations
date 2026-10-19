# Recent developments in robust portfolios with a worst-case approach

JOTA, Vol. 161, 2014 | [Notion summary](https://road-lamp-b0c.notion.site/3fa64a44030c810b9bcbcd48ff6aa45e) | [Paper](https://doi.org/10.1007/s10957-013-0329-1)

## What is in the code
- Mean-variance optimization with estimated parameters
- Robust counterpart with an ellipsoidal uncertainty set for expected returns, solved as a second-order cone program in cvxpy
- Comparison on the true distribution over 200 resampled estimation windows

## Differences from the paper
Only uncertainty in the mean is modeled (the survey also covers covariance and other uncertainty sets).

## Run
```bash
python main.py
```

## Output
```text
oracle (true parameters): objective 0.822%

strategy           true obj %  obj std max weight instability
mean-variance           0.425    0.201       0.44        1.41
robust (kappa=1)        0.574    0.145       0.36        1.21
robust (kappa=2)        0.658    0.097       0.30        1.03
```
