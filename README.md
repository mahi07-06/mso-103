# Seasonal inventory model for steel raw materials: working prototype

MSO-103 group project. Prototype of Kawakami, Kobayashi & Nakata (2021),
*Seasonal Inventory Management Model for Raw Materials in Steel Industry*,
INFORMS Journal on Applied Analytics 51(4): 312-324.

The prototype follows our paper presentation slide by slide.

## Run it (VS Code)

1. Open this folder in VS Code and open a terminal (`` Ctrl+` ``).
2. Once only: `python -m pip install -r requirements.txt`
3. `python main.py` (takes about one second)

Results appear in the terminal and in `outputs/`: six charts, a copy of the console
output, and `Steel_Inventory_Prototype.xlsx`. The workbook's **Calculator** sheet is live:
change a yellow cell (service level, ships per year) and the safety stock recalculates in Excel.

## How it works

The real Nippon Steel shipping records are confidential, so the code first **simulates
10 years of ship arrivals** with a hidden seasonal dip (fewer ships in the cyclone and rainy
seasons). The model never sees that hidden pattern; it has to find it in the ship records,
exactly as it would with real data.

| Step | What happens | Slide | File |
|---|---|---|---|
| 0 | Simulate 10 years of ship arrivals for 3 brand groups and 4 mills | Data & Input | `inventory_model.py` |
| 1 | **Stage 1 (AAM):** ships arrive in Japan as a Poisson process whose rate λ(t) changes with the season. λ(t) is an exponential Fourier series fitted by maximum likelihood (Newton's method); AIC picks the number of terms | Methodology | `inventory_model.py` |
| 2 | **Stage 2 (AEM):** mill i receives every N_i-th ship, N_i = Q / q_i, so the gap between its ships follows a Gamma(N_i, λ(t)) distribution | Methodology | `inventory_model.py` |
| 3 | **Safety stock** Z = D − A = G⁻¹(α) − G⁻¹(50%): the gap for a late ship minus the gap for a normal ship. **Optimal stock** = cycle stock (half a shipload) + Z | Calculating Safety Stock | `inventory_model.py` |
| 4 | **KS test:** do the real gaps at each mill look like the gamma model? | Validation | `inventory_model.py` |
| 5 | **Inventory simulation:** on 30 fresh years, each ship tops the yard up and the mill burns one day of stock per day; count how often stock runs out before the next ship | Validation | `inventory_model.py` |
| 6 | **Results:** add all mills together and compare with the manager's flat policy | Results | `inventory_model.py` |
| 7 | **Brand grouping:** recalculate the stock with the original and the revised brand groups | Results | `brand_grouping.py` |

Why the stock changes with the season: in winter fewer ships arrive (λ(t) is low), so the
gap between ships is longer and more variable, so Z is larger. In summer λ(t) is high, so
Z is smaller. The manager holds the winter level all year; the model does not.

Link to the course: step 1 is an unconstrained **nonlinear optimisation** (maximise the
log-likelihood), solved with Newton's method, with AIC for model selection.

## Simplifying assumptions (all in `config.py`)

- 4 mills and 3 brand groups instead of 8 and 10; synthetic data.
- Constant demand and a fixed lot of 170 kt per ship, as in the paper.
- Service level 95% (5% stock-out rate), as in the paper.
- "Manager policy" = one flat stock level all year, sized for the worst season plus 5%.
  The paper does not publish the real rule, so this is our assumption.
- The strength of the synthetic seasonal dip was chosen so the size of the effect is similar
  to Nippon Steel's.

## Results of the current run

| | Prototype | Paper |
|---|---|---|
| Inventory reduction in winter | 6.2% | 6% |
| Inventory reduction in summer | 17.4% | 14% |
| Inventory reduction, average | 11.8% | 11% |
| Mill / brand-group pairs passing the KS test | 10 of 10 | most; brand group 7 failed |
| Shortage ratio in simulation (target 5%) | 5.2% (winter season) / 7.1% (calm season) | within threshold except brand group 7 |
| Holding days saved by revised brand grouping | 18.1% | 18.1% |

The paper's US$4.5M demurrage saving is measured on Nippon Steel's real operations and
is not reproduced: our prototype has no cost data.

Limitations to mention if asked: the data are synthetic; the stock-out rate is slightly above
5% in the calm season because some ships break the regular rotation, which the gamma model
does not capture (the paper saw the same effect in brand group 7); demand and lot size are
constant, as in the paper.

## Files

| File | Contents |
|---|---|
| `main.py` | Runs every step and prints the results |
| `config.py` | Every assumption and input |
| `inventory_model.py` | Steps 0 to 6 |
| `brand_grouping.py` | Step 7 |
| `stats_utils.py` | Gamma distribution and KS test, written with numpy only |
| `report.py` | Charts and the Excel workbook |
