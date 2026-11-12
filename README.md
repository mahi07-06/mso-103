# Seasonal inventory model for steel raw materials: working prototype

MSO-103 group project. Prototype of Kawakami, Kobayashi & Nakata (2021),
*Seasonal Inventory Management Model for Raw Materials in Steel Industry*,
INFORMS Journal on Applied Analytics 51(4): 312-324.

## Run it (VS Code)

1. Open this folder in VS Code and open a terminal (`` Ctrl+` ``).
2. Once only: `python -m pip install -r requirements.txt`
3. `python main.py` (about one minute)

Results appear in the terminal and in `outputs/`: eight charts, a copy of the console
output, and `Steel_Inventory_Prototype.xlsx`. The workbook's **Calculator** sheet is live:
change a yellow cell and the safety stock recalculates in Excel.

`pulp` is pinned to 2.9.0 on purpose. Newer versions and scipy are blocked by Windows
on this laptop, so the gamma distribution and KS test are coded by hand in `stats_utils.py`.

## What the prototype does

The real shipping records are confidential, so `inventory_model.py` first **simulates 10 years
of ship arrivals** from a hidden seasonal process (fewer ships in the cyclone / rainy season).
The model then has to recover everything from that data, as it would with real records.

| Part | Model | Course topic | File |
|---|---|---|---|
| 1 | Seasonal arrival rate λ(t) = exp(Fourier series), fitted by maximum likelihood with Newton's method; harmonics chosen by AIC | Nonlinear programming | `inventory_model.py` |
| 1 | Interval at mill i ~ Gamma(N_i, λ(t)); safety stock Z = G⁻¹(α) − G⁻¹(50%); optimal stock = cycle + Z | The paper's model (eq. 1-8) | `inventory_model.py` |
| 1 | KS test and out-of-sample stock-out check | Validation, as in the paper | `inventory_model.py` |
| 2A | 12-month shipping and yard plan with whole ships, minimising holding + demurrage + shortfall cost | Integer programming, branch & bound | `optimisation.py` |
| 2A | LP relaxation, shadow price of yard space, what-if on yard capacity | LP, duality, sensitivity analysis | `optimisation.py` |
| 2B | Which fine-ore brands to manage as one group (set partitioning) | Binary programming | `optimisation.py` |
| 2C | Total stock vs stock-out risk, ε-constraint method, Pareto frontier | Multi-objective optimisation | `optimisation.py` |

Part 1 is what the paper did. Part 2 is our extension: 2B is the paper's stated future work
("find the optimal grouping strategy") and 2C its stated limitation (the 5% stock-out rate
was fixed, not optimised against cost).

## Simplifying assumptions (all editable in `config.py`)

- 3 brand groups and 4 mills instead of 10 and 8; synthetic data.
- Constant demand and a fixed lot of 170 kt per ship, as in the paper.
- "Manager policy" = one flat stock level all year, sized for the worst season plus 5%.
  The paper does not publish the real rule, so this is our assumption and it drives the
  size of the reduction.
- Yard capacity equals the stock the manager policy wants to hold; cargo that does not fit
  waits at sea and pays demurrage.
- Costs: holding US$0.7, demurrage US$3.5, shortfall US$25 per tonne-month (our estimates).
- The shipping plan works in months and looks at month-end stock only.

## Results of the current run

| | Prototype | Paper |
|---|---|---|
| Inventory reduction, disruption season | 4.6% | 6% |
| Inventory reduction, calm season | 27.9% | 14% |
| Inventory reduction, average | 18.5% | 11% |
| Mill / brand-group pairs passing the KS test | 10 of 10 | most |
| Stock-out rate with the seasonal model (target 5%) | 5.6% / 7.0% | within threshold except one group |
| Demurrage saved | US$12.4 M a year | at least US$4.5 M a year |
| Stock saved by regrouping brands | 38.3% | 18.1% |

The direction matches the paper; the sizes differ because the data and the manager rule are
ours. Two things to say honestly if asked: the seasonal model's stock-out rate comes out a
little above the 5% target (the gamma model slightly understates the variability), and the
two integer plans are the best found in 30 seconds of branch and bound, not proven optimal.
