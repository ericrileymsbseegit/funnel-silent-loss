# Funnel Silent Loss

A demonstration of how to find a loss that no team can see: completed insurance applications that quietly never become policies, even though IT, P&C and Marketing are each right about their own numbers.

> **Note on data:** Everything here runs on synthetic data generated to mimic the structure of an insurance issuance funnel. It contains no proprietary or real customer information.

**Start here:** `Start Here Read Me Funnel Evaluation Framework Synthetic Example` (the executive deck). The other two decks go deeper into the survival analysis and the nightly monitoring design.

## The problem

Policy issuance runs below expectation while application volume looks normal. There are no complaints, no refunds and no error flags. Each team reads the funnel its own way:

- **IT** tracks pass rates at each step and sees a system that meets its acceptance criteria.
- **P&C** tracks losses as a share of the book and sees fewer policies than it should.
- **Marketing** sees a top-of-funnel volume problem.

Everyone is right about their own numbers, so the conversation stalls. The loss that matters sits between *application complete* and *policy issued*, the step every current report treats as unremarkable.

## What this shows

- **Where the loss happens.** Applications that hit a system trap move through every earlier step at the same rate as everyone else (within about 2 points, in both directions). At issuance they fall about 20 points behind: 87.5% issued on the clean path vs 67.0% after a trap.
- **That it isn't chance.** Simulating the funnel as if the trap made no difference never reproduces what was observed. Letting trapped applications issue at their own rate does.
- **That it doesn't build up.** There is no sign of damage accumulating along the funnel, and a second, expected loss (underwriting declines) doesn't stack on top of the trap. The search narrows to one step instead of the whole pipeline.

## Approach

- **The funnel as a survival curve.** Each stage is a step in time and leaving the funnel is the event. This explains why IT's pass rates and P&C's loss shares disagree, and why late-stage losses always look small as a share of the book.
- **Posterior predictive checks.** Each step's issuance rate gets a Beta posterior. Simulated outcomes under "the trap makes no difference" are compared with what actually happened.
- **Stability checks.** Bootstrap resampling shows how much the findings move from sample to sample.
- **Built to run nightly.** Validation, drift checks and clear exit codes, so a scheduler can rerun a failed job or alert a person instead of failing silently.

## Why it's framed this way

The hardest part of this problem wasn't the math. It was that no one was technically wrong. Comparing trapped applications with untrapped ones at the same step, in the same period, gives a baseline every team can accept: the system's own clean path. That turns an argument about whose numbers are right into a specific, testable fix.

## Files

 File | What it does |
________________________________________________________________________________________________________________________________
 `generate.py` | Creates the synthetic funnel data 
 `validate.py` | Checks the data adds up before anything else runs 
 `predictive_gap.py` | Simulates expected issuance and compares it with what happened 
 `trap.py` | Compares "the trap makes no difference" with "the trap has its own rate" 
 `analyze_traps.py` | Step-by-step pass rates, trapped vs not, and issuance by trap type 
 `bootstrap_traps.py` | Resamples the data 1,000 times to check how stable the findings are 
 `diagnose.py` | Tests whether the funnel as declared can reproduce what was observed (posterior predictive check) 
 `monitor.py` | Drift checks for the nightly run (Automated ongoing monitoring)
 `make_figure.py` | Plots 40,000 posterior draws of expected issuance against the actual count 
 `funnel_synthetic.csv` | Synthetic sample of 10,000 members 

## Running it

```bash
pip install numpy pandas matplotlib scipy statsmodels
python analyze_traps.py
python bootstrap_traps.py
```

## Stack

Python · NumPy · pandas · SciPy · statsmodels · Matplotlib

---

*Built as a portfolio demonstration of funnel diagnostics and Bayesian model checking.*
