"""
inventory_model.py  -  PART 1: the paper's inventory model, step by step.

Read this file from top to bottom. Each STEP is one small function.

  STEP 0  Make synthetic ship-arrival data          (the real data is confidential)
  STEP 1  Fit the seasonal arrival rate lambda(t)   (paper eq. 6, "Stage 1: AAM")
  STEP 2  Find N_i = every N-th ship goes to mill i (paper eq. 3, "Stage 2: AEM")
  STEP 3  Safety stock Z and optimal stock I        (paper eq. 4 and 8)
  STEP 4  KS test: does the gamma model fit?        (paper Table 1)
  STEP 5  Stock-out check on fresh data             (paper Tables 2 and 3)
  STEP 6  Add everything up for all mills           (paper Figure 8)

Units: time in days, stock in "days of usage" (1 day of usage = what the mill burns in a day).
"""
import math
import numpy as np

import config as C
from stats_utils import gamma_cdf, gamma_ppf, ks_test

TWO_PI = 2.0 * math.pi
DAYS = np.arange(C.P) + 0.5          # the middle of each day of the year: 0.5, 1.5, ... 364.5


# =============================================================================
# STEP 0  Synthetic data
# =============================================================================
def true_intensity(cfg, t):
    """
    The "real world" arrival rate in ships per day at time t.
    It dips around cfg["dip_day"] (cyclone / rainy season).
    The model is NOT allowed to look at this - it must estimate it from the ships.
    """
    def seasonal_shape(day):
        angle = TWO_PI * (day - cfg["dip_day"]) / C.P
        return np.exp(-cfg["amp1"] * np.cos(angle) - cfg["amp2"] * np.cos(2 * angle))

    average_rate = cfg["ships_per_year"] / C.P
    t = np.asarray(t, dtype=float)
    return average_rate * seasonal_shape(t) / seasonal_shape(DAYS).mean()


def simulate_arrivals(cfg, years, rng):
    """
    Create ship arrival times for one brand group, and the mill each ship goes to.

    Arrival times: a Poisson process whose rate changes with the season.
    (Method: generate too many candidate ships at the peak rate, then keep each
     one with probability rate(t) / peak rate.  This is called "thinning".)

    Mill choice: the operator rotates ships between mills in proportion to each
    mill's share, but sometimes (ALLOCATION_NOISE) overrides the rotation.
    """
    # ---- when do ships arrive in Japan?
    horizon = years * C.P
    peak_rate = true_intensity(cfg, DAYS).max() * 1.02
    number_of_candidates = rng.poisson(peak_rate * horizon)
    candidates = np.sort(rng.uniform(0, horizon, number_of_candidates))
    keep = rng.uniform(0, peak_rate, number_of_candidates) < true_intensity(cfg, candidates % C.P)
    times = candidates[keep]

    # ---- which mill does each ship go to?
    mills = list(cfg["shares"])
    share = np.array([cfg["shares"][m] for m in mills])
    credit = rng.uniform(0, 1, len(mills))      # how "overdue" each mill is
    destination = np.empty(len(times), dtype=int)
    for j in range(len(times)):
        credit += share
        if rng.uniform() < C.ALLOCATION_NOISE:
            k = rng.choice(len(mills), p=share)     # operator overrides the rotation
        else:
            k = int(np.argmax(credit))              # normal case: most overdue mill
        credit[k] -= 1.0
        destination[j] = k
    return times, destination, mills


# =============================================================================
# STEP 1  Seasonal arrival rate (Stage 1, "arrival at all mills")
# =============================================================================
def fourier_columns(t, harmonics):
    """Columns [1, cos, sin, cos2, sin2, ...] so that log lambda(t) = columns x coefficients."""
    columns = [np.ones_like(t)]
    for h in range(1, harmonics + 1):
        columns.append(np.cos(TWO_PI * h * t / C.P))
        columns.append(np.sin(TWO_PI * h * t / C.P))
    return np.column_stack(columns)


def fit_nhpp(times, years, max_order=3, min_aic_gain=2.0):
    """
    Fit  lambda(t) = exp(A1 + A2 cos(2 pi t/P) + B2 sin(2 pi t/P) + ...)   (paper eq. 6)

    This is an optimisation problem (nonlinear programming, no constraints):
        maximise   log-likelihood = sum over ships of log lambda(t_i)  -  total expected ships
    We solve it with Newton's method:  new = old - (gradient / Hessian).

    We try 0, 1, 2, 3 harmonics and keep the one with the best AIC score
    (AIC = 2 x number of coefficients - 2 x log-likelihood; lower is better).
    """
    day_of_year = times % C.P
    best = None
    aic_table = []

    for harmonics in range(max_order + 1):
        X_ships = fourier_columns(day_of_year, harmonics)   # one row per ship
        X_days = fourier_columns(DAYS, harmonics)           # one row per day of the year

        def log_likelihood(coef):
            return (X_ships @ coef).sum() - years * np.exp(X_days @ coef).sum()

        # start from a flat rate (no seasonality)
        coef = np.zeros(X_days.shape[1])
        coef[0] = math.log(len(times) / (years * C.P))
        current = log_likelihood(coef)

        for iteration in range(1, 101):
            rate = np.exp(X_days @ coef)
            gradient = X_ships.sum(axis=0) - years * X_days.T @ rate
            hessian = -years * X_days.T @ (rate[:, None] * X_days)
            newton_step = np.linalg.solve(-hessian, gradient)

            step_size = 1.0                      # halve the step if it would make things worse
            while log_likelihood(coef + step_size * newton_step) < current and step_size > 1e-8:
                step_size *= 0.5
            coef = coef + step_size * newton_step

            improved_by = abs(log_likelihood(coef) - current)
            current = log_likelihood(coef)
            if improved_by < 1e-10:              # converged
                break

        aic = 2 * len(coef) - 2 * current
        aic_table.append(dict(order=harmonics, n_params=len(coef), loglik=current,
                              aic=aic, newton_iters=iteration))

        # keep this model only if it is clearly better (AIC lower by at least 2)
        if best is None or aic < best["aic"] - min_aic_gain:
            best = dict(order=harmonics, theta=coef, aic=aic, lam_day=np.exp(X_days @ coef))

    best["aic_table"] = aic_table
    return best


# =============================================================================
# STEP 2-5  Everything for one mill
# =============================================================================
def analyse_mill(mill_index, mill, cfg, data):
    """Steps 2 to 5 for one mill of one brand group. `data` holds the group's ships and rates."""
    lam_now = data["lam_now"]        # this year's arrival rate for each day (365 numbers)
    lam_bar = data["lam_bar"]        # this year's average arrival rate (1 number)
    ships_to_mill = data["recent"] & (data["dest"] == mill_index)

    # ---- STEP 2: every N-th ship comes to this mill          N_i = Q / q_i   (eq. 3)
    q_total = int(data["recent"].sum())      # Q:   ships to Japan in the last 2 years
    q_mill = int(ships_to_mill.sum())        # q_i: of those, ships to this mill
    n_i = q_total / q_mill

    # ---- STEP 3: safety stock                                Z = G^-1(alpha) - G^-1(50%)   (eq. 4)
    # The gap between ships at this mill follows Gamma(N_i, lambda).
    # gamma_ppf gives the gap for rate 1; dividing by the real rate converts it to days.
    late_gap = gamma_ppf(C.SERVICE_LEVEL, n_i)     # a "late" ship (95th percentile)
    normal_gap = gamma_ppf(0.50, n_i)              # a "normal" ship (median)

    safety_seasonal = (late_gap - normal_gap) / lam_now     # 365 values: changes with the season
    safety_flat = (late_gap - normal_gap) / lam_bar         # if we ignored the seasons
    safety_manager = (1 + C.MANAGER_MARGIN) * safety_seasonal.max()   # worst season + margin

    cycle_stock = 0.5 * n_i / lam_bar              # half of one shipload, in days of usage
    usage = cfg["ships_per_year"] * cfg["shares"][mill] * C.LOT_KT / C.P      # kt burned per day

    # ---- STEP 4: KS test - do the real gaps look like Gamma(N_i)?
    # First stretch time so the seasonal rate becomes a constant rate of 1
    # ("time-rescaling"); then the gaps should follow Gamma(N_i) with rate 1.
    rescaled_times = data["rescale"](data["times"][ships_to_mill])
    rescaled_gaps = np.diff(rescaled_times)

    def model_cdf(gap):
        return gamma_cdf(gap, n_i)

    ks_statistic, ks_p_value = ks_test(rescaled_gaps, model_cdf)

    # ---- STEP 5: stock-out check on 30 fresh years the model has never seen.
    # A stock-out happens when the gap to the next ship is longer than the stock covers.
    test_arrivals = data["test_times"][data["test_dest"] == mill_index]
    gaps = np.diff(test_arrivals)
    start_day = (test_arrivals[:-1] % C.P).astype(int)
    in_low_season = data["low_season"][start_day]

    days_covered = {
        "Proposed (seasonal)": late_gap / lam_now[start_day],
        "No seasonality": np.full(len(gaps), late_gap / lam_bar),
        "Manager (flat)": np.full(len(gaps), normal_gap / lam_bar + safety_manager),
    }
    stockouts = {}
    for policy, cover in days_covered.items():
        too_long = gaps > cover
        stockouts[policy] = dict(
            low=(int(too_long[in_low_season].sum()), int(in_low_season.sum())),
            high=(int(too_long[~in_low_season].sum()), int((~in_low_season).sum())))

    return dict(
        q_i=q_mill, q_total=q_total, n_i=n_i, k_hi=late_gap, k_md=normal_gap, usage=usage,
        cycle=cycle_stock, z_t=safety_seasonal, z_stat=safety_flat, z_mgr=safety_manager,
        inv_prop=cycle_stock + safety_seasonal,      # optimal stock I(t) = cycle + Z(t)   (eq. 8)
        inv_stat=cycle_stock + safety_flat,
        inv_mgr=cycle_stock + safety_manager,
        gaps_days=np.diff(data["times"][ships_to_mill]), gaps_tau=rescaled_gaps,
        ks_d=ks_statistic, ks_p=ks_p_value, stockouts=stockouts)


def analyse_group(name, cfg, rng_hist, rng_test):
    """Run steps 0 to 5 for one brand group (for example 'Fine ore (Australia)')."""
    # STEP 0: ten years of ship records
    times, dest, mills = simulate_arrivals(cfg, C.YEARS_HISTORY, rng_hist)

    # STEP 1: seasonal arrival rate from those ten years
    fit = fit_nhpp(times, C.YEARS_HISTORY)
    lam_past = fit["lam_day"]

    # This year's rate = same seasonal shape, scaled to this year's contract volume (eq. 7)
    lam_bar = cfg["ships_per_year"] / C.P
    lam_now = lam_bar * lam_past / lam_past.mean()

    # Helper for the KS test: expected number of ships up to time t ("stretched time")
    ships_expected_by_day = np.concatenate([[0.0], np.cumsum(lam_past)])

    def rescale(t):
        full_years = np.floor(t / C.P)
        within_year = np.interp(t % C.P, np.arange(C.P + 1), ships_expected_by_day)
        return full_years * ships_expected_by_day[-1] + within_year

    # the paper uses only the last 2 years to estimate each mill's share
    recent = times >= (C.YEARS_HISTORY - C.YEARS_RECENT) * C.P

    # fresh data for the stock-out check
    test_times, test_dest, _ = simulate_arrivals(cfg, C.YEARS_TEST, rng_test)

    data = dict(times=times, dest=dest, recent=recent, rescale=rescale,
                lam_now=lam_now, lam_bar=lam_bar, low_season=lam_now < lam_bar,
                test_times=test_times, test_dest=test_dest)

    result = dict(name=name, mills=mills, times=times, dest=dest, fit=fit, lam_now=lam_now,
                  lam_bar=lam_bar, lam_true=true_intensity(cfg, DAYS),
                  low_season=data["low_season"], per_mill={})
    for mill_index, mill in enumerate(mills):
        result["per_mill"][mill] = analyse_mill(mill_index, mill, cfg, data)
    return result


# =============================================================================
# STEP 6  All brand groups and all mills together
# =============================================================================
def run_all():
    rng_hist = np.random.default_rng(C.SEED)         # random numbers for the history
    rng_test = np.random.default_rng(C.SEED + 1)     # separate random numbers for the test years

    groups = {}
    for name, cfg in C.BRAND_GROUPS.items():
        groups[name] = analyse_group(name, cfg, rng_hist, rng_test)

    # ---- company-wide stock (paper Figure 8).
    # Average the stock days over all mills, weighting each by how much it burns.
    total_usage = 0.0
    proposed = np.zeros(C.P)
    manager = 0.0
    no_season = 0.0
    for group in groups.values():
        for mill in group["per_mill"].values():
            total_usage += mill["usage"]
            proposed += mill["usage"] * mill["inv_prop"]
            manager += mill["usage"] * mill["inv_mgr"]
            no_season += mill["usage"] * mill["inv_stat"]

    percent_of_manager = 100.0 * proposed / manager          # manager policy = 100 %
    month_of_day = np.repeat(np.arange(12), C.MONTH_DAYS)
    monthly = np.array([percent_of_manager[month_of_day == k].mean() for k in range(12)])

    summary = dict(
        pct_daily=percent_of_manager, pct_monthly=monthly,
        stat_pct=100.0 * no_season / manager,
        mgr_days=manager / total_usage, prop_days=proposed / total_usage,
        avg_reduction=100 - percent_of_manager.mean(),
        max_reduction=100 - percent_of_manager.min(),     # calm season
        min_reduction=100 - percent_of_manager.max())     # disruption season

    # ---- stock-out rate of each policy, all mills together
    counts = {}
    for group in groups.values():
        for mill in group["per_mill"].values():
            for policy, result in mill["stockouts"].items():
                if policy not in counts:
                    counts[policy] = dict(low=[0, 0], high=[0, 0])
                for season in ("low", "high"):
                    counts[policy][season][0] += result[season][0]     # stock-outs
                    counts[policy][season][1] += result[season][1]     # cycles
    summary["stockout_rates"] = {}
    for policy, by_season in counts.items():
        summary["stockout_rates"][policy] = {
            season: 100.0 * c[0] / c[1] for season, c in by_season.items()}

    return groups, summary
