"""
optimisation.py  -  PART 2: three optimisation models built on the paper's result.

The paper stops at "this is how much stock to hold". We go one step further and
use that number inside optimisation models from the course.

  MODEL A  Shipping and yard plan     Integer programming (whole ships)
           + the same model as an LP  LP relaxation, shadow prices, sensitivity
  MODEL B  Brand grouping             Binary programming (yes/no variables)
  MODEL C  Service level vs stock     Multi-objective optimisation (Pareto frontier)

Every model is written the same way, as in class:
    1. decision variables   2. objective   3. constraints   4. solve   5. read the answer

Library: PuLP (the model) + CBC (the solver, uses branch and bound for integers).
"""
from itertools import combinations
import numpy as np
import pulp

import config as C
from stats_utils import gamma_ppf

MANAGER = "Manager (flat)"
PROPOSED = "Proposed (seasonal)"
POLICIES = [MANAGER, PROPOSED]

MONTHS = range(12)
MONTH_OF_DAY = np.repeat(np.arange(12), C.MONTH_DAYS)     # which month each day belongs to


def make_solver():
    """CBC solver. Stops after MIP_TIME_LIMIT seconds or when within 0.5 % of the best bound."""
    return pulp.PULP_CBC_CMD(msg=0, timeLimit=C.MIP_TIME_LIMIT, gapRel=0.005)


def monthly_average(daily_values):
    """Turn 365 daily numbers into 12 monthly averages."""
    return np.array([daily_values[MONTH_OF_DAY == k].mean() for k in MONTHS])


def value_of(variable):
    """The solved value of a PuLP variable (0 if the solver left it empty)."""
    return variable.value() or 0.0


# =============================================================================
# MODEL A   Shipping and yard plan
# =============================================================================
def plan_inputs(groups):
    """Collect the numbers the plan needs, month by month, from the Part 1 results."""
    pairs = []                    # every (brand group, mill) combination that exists
    demand = {}                   # kt the mill burns each month
    ships_available = {}          # ships that can reach Japan each month (seasonal)
    target = {MANAGER: {}, PROPOSED: {}}     # stock the policy wants to hold, in kt

    for group_name, group in groups.items():
        # Ships available per month = the fitted seasonal rate added up over the month,
        # rounded to whole ships so that the year total stays correct.
        exact = np.array([group["lam_now"][MONTH_OF_DAY == k].sum() for k in MONTHS])
        whole = np.floor(exact).astype(int)
        ships_still_to_place = int(round(exact.sum())) - whole.sum()
        months_with_biggest_remainder = np.argsort(-(exact - whole))[:ships_still_to_place]
        for k in months_with_biggest_remainder:
            whole[k] += 1
        ships_available[group_name] = whole

        for mill_name, mill in group["per_mill"].items():
            pairs.append((group_name, mill_name))
            demand[group_name, mill_name] = mill["usage"] * np.array(C.MONTH_DAYS)
            # stock target in kt = stock target in days x kt burned per day
            target[MANAGER][group_name, mill_name] = np.full(12, mill["inv_mgr"] * mill["usage"])
            target[PROPOSED][group_name, mill_name] = monthly_average(mill["inv_prop"]) * mill["usage"]

    # Yard capacity of each mill: big enough for the manager policy's targets (assumption).
    capacity = {}
    for mill_name in C.MILLS:
        manager_total = sum(t[0] for (g, m), t in target[MANAGER].items() if m == mill_name)
        capacity[mill_name] = C.YARD_TIGHTNESS * manager_total

    return dict(pairs=pairs, demand=demand, avail=ships_available, target=target, cap=capacity)


def solve_plan(inp, policy, cap_scale=1.0, integer=True):
    """
    Build and solve the 12-month shipping plan for one stock policy.

    integer=True   ships are whole numbers  -> integer programme (branch and bound)
    integer=False  ships can be fractions   -> LP relaxation (gives shadow prices)
    cap_scale      multiply every yard capacity by this number (for the what-if analysis)
    """
    pairs = inp["pairs"]
    target = inp["target"][policy]
    model = pulp.LpProblem("shipping_plan", pulp.LpMinimize)

    # ---- 1. DECISION VARIABLES (g = brand group, m = mill, t = month)
    pair_months = [(g, m, t) for g, m in pairs for t in MONTHS]
    ship_type = "Integer" if integer else "Continuous"
    ships = pulp.LpVariable.dicts("n", pair_months, 0, cat=ship_type)   # ships unloaded
    stock = pulp.LpVariable.dicts("I", pair_months, 0)                  # kt in the yard at month end
    short = pulp.LpVariable.dicts("U", pair_months, 0)                  # kt below the safety target
    waiting = pulp.LpVariable.dicts("W", [(m, t) for m in C.MILLS for t in MONTHS], 0)   # kt waiting at sea
    deferred = pulp.LpVariable.dicts("R", [(g, t) for g in inp["avail"] for t in MONTHS], 0)  # ships held back

    # ---- 2. OBJECTIVE: minimise holding cost + demurrage + shortfall penalty
    model += (C.HOLDING_COST * pulp.lpSum(stock.values())
              + C.DEMURRAGE_COST * pulp.lpSum(waiting.values())
              + C.SHORTFALL_PENALTY * pulp.lpSum(short.values()))

    # ---- 3. CONSTRAINTS
    for g, m in pairs:
        for t in MONTHS:
            last_month = (t - 1) % 12        # January follows December: the year repeats

            # (a) stock balance:  stock now = stock before + ships x lot size - usage
            model += (stock[g, m, t] == stock[g, m, last_month]
                      + C.LOT_KT * ships[g, m, t] - inp["demand"][g, m][t]), \
                f"balance_{g}_{m}_{t}".replace(" ", "_")

            # (b) safety target from Part 1:  stock + shortfall >= target
            model += stock[g, m, t] + short[g, m, t] >= target[g, m][t]

    for g in inp["avail"]:
        mills_of_group = [m for gg, m in pairs if gg == g]
        for t in MONTHS:
            last_month = (t - 1) % 12
            # (c) seasonal supply: a ship can be taken later than it is available, never earlier
            #     held back now = held back before + newly available - ships taken this month
            model += (deferred[g, t] == deferred[g, last_month] + int(inp["avail"][g][t])
                      - pulp.lpSum(ships[g, m, t] for m in mills_of_group))

    for m in C.MILLS:
        for t in MONTHS:
            stock_at_mill = pulp.lpSum(stock[g, mm, t] for g, mm in pairs if mm == m)
            # (d) yard capacity: what does not fit in the yard has to wait at sea
            model += (stock_at_mill - waiting[m, t] <= cap_scale * inp["cap"][m]), \
                f"yard_{m}_{t}".replace(" ", "_")

    # ---- 4. SOLVE
    model.solve(make_solver())

    # ---- 5. READ THE ANSWER
    stock_result = {}
    ships_result = {}
    short_result = {}
    for g, m in pairs:
        stock_result[g, m] = np.array([value_of(stock[g, m, t]) for t in MONTHS])
        ships_result[g, m] = np.array([round(value_of(ships[g, m, t]), 6) for t in MONTHS])
        short_result[g, m] = np.array([value_of(short[g, m, t]) for t in MONTHS])

    waiting_result = {}
    stock_per_mill = {}
    capacity = {}
    for m in C.MILLS:
        waiting_result[m] = np.array([value_of(waiting[m, t]) for t in MONTHS])
        stock_per_mill[m] = sum(stock_result[g, mm] for g, mm in pairs if mm == m)
        capacity[m] = cap_scale * inp["cap"][m]

    holding_cost = C.HOLDING_COST * sum(v.sum() for v in stock_result.values())
    demurrage_cost = C.DEMURRAGE_COST * sum(v.sum() for v in waiting_result.values())
    penalty_cost = C.SHORTFALL_PENALTY * sum(v.sum() for v in short_result.values())

    result = dict(
        policy=policy, integer=integer, status=pulp.LpSolution[model.sol_status],
        ships=ships_result, inv=stock_result, short=short_result, wait=waiting_result,
        mill_inv=stock_per_mill, cap=capacity,
        holding=holding_cost, demurrage=demurrage_cost, penalty=penalty_cost,
        total=holding_cost + demurrage_cost + penalty_cost,
        avg_inv=sum(v.mean() for v in stock_result.values()))

    if not integer:
        # DUALITY: the shadow price of a yard constraint says how much the total cost
        # would fall if that yard were 1 kt bigger. We add it up over the 12 months.
        result["shadow"] = {}
        for m in C.MILLS:
            total_price = 0.0
            for t in MONTHS:
                constraint = model.constraints[f"yard_{m}_{t}".replace(" ", "_")]
                total_price += constraint.pi or 0.0
            result["shadow"][m] = -total_price
    return result


def run_plans(groups):
    """Solve the plan for both policies, then repeat it for smaller and larger yards."""
    inp = plan_inputs(groups)
    out = dict(inputs=inp, mip={}, lp={}, sensitivity=[])

    for policy in POLICIES:
        out["mip"][policy] = solve_plan(inp, policy, integer=True)     # whole ships
        out["lp"][policy] = solve_plan(inp, policy, integer=False)     # LP relaxation

    # SENSITIVITY ANALYSIS: what if the yards were 20 % smaller ... 10 % larger?
    for scale in (0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10):
        row = dict(scale=scale)
        for policy in POLICIES:
            r = solve_plan(inp, policy, cap_scale=scale, integer=False)
            row[policy] = r["total"]
            row[policy + " demurrage"] = r["demurrage"]
            row[policy + " shadow"] = sum(r["shadow"].values()) / len(r["shadow"])
        out["sensitivity"].append(row)
    return out


# =============================================================================
# MODEL B   Brand grouping
# =============================================================================
def stock_needed(brands, mill_index):
    """
    Stock in kt that one mill must hold if `brands` are managed together as ONE group.
    Same formula as Part 1: cycle stock (half a lot) + safety stock.
    Bigger groups get ships more often, so they need less safety stock per tonne.
    """
    ships_to_japan = sum(sum(C.FINE_BRANDS[b][1]) for b in brands)          # Q, per year
    ships_to_mill = sum(C.FINE_BRANDS[b][1][mill_index] for b in brands)    # q_i, per year
    n_i = ships_to_japan / ships_to_mill
    arrival_rate = ships_to_japan / C.P
    safety_days = (gamma_ppf(C.SERVICE_LEVEL, n_i) - gamma_ppf(0.5, n_i)) / arrival_rate
    usage = ships_to_mill * C.LOT_KT / C.P
    return 0.5 * C.LOT_KT + usage * safety_days


def run_grouping():
    """
    Which fine-ore brands should each mill manage as one group?

    This is a "set partitioning" problem: list every allowed group, then choose
    groups so that every brand is covered exactly once, at the lowest total stock.
    """
    model = pulp.LpProblem("brand_grouping", pulp.LpMinimize)

    # ---- 1. DECISION VARIABLES: use[group, mill] = 1 if that group is used at that mill
    use = {}
    stock_of = {}
    brands_at = {}
    for mill_index, mill in enumerate(C.MILLS):
        brands_at[mill] = [b for b, (_, ships) in C.FINE_BRANDS.items() if ships[mill_index] > 0]

        # list every allowed group: up to MAX_GROUP_SIZE brands with similar iron grade
        for size in range(1, C.MAX_GROUP_SIZE + 1):
            for group in combinations(brands_at[mill], size):
                grades = [C.FINE_BRANDS[b][0] for b in group]
                if max(grades) - min(grades) <= C.MAX_FE_SPREAD + 1e-9:
                    stock_of[group, mill] = stock_needed(group, mill_index)
                    use[group, mill] = pulp.LpVariable(f"x_{mill_index}_{len(use)}", cat="Binary")

    # ---- 2. OBJECTIVE: minimise the total stock of the groups we use
    model += pulp.lpSum(stock_of[key] * use[key] for key in use)

    # ---- 3. CONSTRAINT: every brand is in exactly one chosen group at each mill
    for mill, brands in brands_at.items():
        for brand in brands:
            model += pulp.lpSum(use[group, m] for (group, m) in use
                                if m == mill and brand in group) == 1

    # ---- 4. SOLVE
    model.solve(make_solver())

    # ---- 5. READ THE ANSWER and compare with the grouping the managers use today
    rows = []
    for mill_index, mill in enumerate(C.MILLS):
        original = []
        for group in C.ORIGINAL_GROUPS:
            present = [b for b in group if b in brands_at[mill]]
            if present:
                original.append(present)
        optimal = [list(group) for (group, m) in use if m == mill and use[group, m].value() > 0.5]

        rows.append(dict(
            mill=mill, original=original, optimal=optimal,
            cost_individual=sum(stock_needed([b], mill_index) for b in brands_at[mill]),
            cost_original=sum(stock_needed(g, mill_index) for g in original),
            cost_optimal=sum(stock_needed(g, mill_index) for g in optimal)))

    total_individual = sum(r["cost_individual"] for r in rows)
    total_original = sum(r["cost_original"] for r in rows)
    total_optimal = sum(r["cost_optimal"] for r in rows)
    return dict(rows=rows, n_candidates=len(use), status=pulp.LpStatus[model.status],
                total_individual=total_individual, total_original=total_original,
                total_optimal=total_optimal,
                reduction=100 * (1 - total_optimal / total_original))


# =============================================================================
# MODEL C   Service level versus stock (two objectives)
# =============================================================================
def run_pareto(groups):
    """
    Two goals that pull against each other:
        objective 1  total stock [kt]                      (want it small)
        objective 2  expected stock-outs per year          (want it small)

    Each (brand group, mill) chooses ONE service level from ALPHA_MENU.
    Method (epsilon-constraint): minimise stock subject to "stock-outs <= limit",
    then repeat for many limits. The answers form the Pareto frontier.
    """
    menu = C.ALPHA_MENU
    choices = range(len(menu))

    # For every (brand group, mill) and every service level: how much stock, how much risk?
    items = []
    for group_name, group in groups.items():
        for mill_name, mill in group["per_mill"].items():
            cycles_per_year = (C.BRAND_GROUPS[group_name]["ships_per_year"]
                               * C.BRAND_GROUPS[group_name]["shares"][mill_name])
            stock_options = []
            risk_options = []
            for alpha in menu:
                safety_days = (gamma_ppf(alpha, mill["n_i"]) - mill["k_md"]) / group["lam_bar"]
                stock_options.append(mill["usage"] * (mill["cycle"] + safety_days))
                risk_options.append(cycles_per_year * (1 - alpha))
            items.append(dict(key=(group_name, mill_name), inv=stock_options, risk=risk_options))
    item_ids = range(len(items))

    # Reference points: the same service level for everybody
    uniform = []
    for j in choices:
        uniform.append(dict(alpha=menu[j],
                            inv=sum(item["inv"][j] for item in items),
                            risk=sum(item["risk"][j] for item in items)))
    base = uniform[menu.index(C.SERVICE_LEVEL)]          # "95 % everywhere"

    def least_stock_for(risk_limit):
        model = pulp.LpProblem("service_level", pulp.LpMinimize)
        # 1. variables: pick[k, j] = 1 if item k uses service level j
        pick = pulp.LpVariable.dicts("y", [(k, j) for k in item_ids for j in choices], cat="Binary")
        # 2. objective: total stock
        model += pulp.lpSum(items[k]["inv"][j] * pick[k, j] for k, j in pick)
        # 3. constraints: exactly one level per item, and total risk within the limit
        for k in item_ids:
            model += pulp.lpSum(pick[k, j] for j in choices) == 1
        model += pulp.lpSum(items[k]["risk"][j] * pick[k, j] for k, j in pick) <= risk_limit
        # 4. solve
        model.solve(make_solver())
        # 5. read the answer
        chosen = [next(j for j in choices if pick[k, j].value() > 0.5) for k in item_ids]
        return dict(eps=risk_limit,
                    inv=sum(items[k]["inv"][chosen[k]] for k in item_ids),
                    risk=sum(items[k]["risk"][chosen[k]] for k in item_ids),
                    alphas={items[k]["key"]: menu[chosen[k]] for k in item_ids})

    lowest_risk = uniform[-1]["risk"]
    highest_risk = uniform[0]["risk"]
    frontier = [least_stock_for(limit) for limit in np.linspace(lowest_risk, highest_risk, 16)]

    same_risk = least_stock_for(base["risk"] + 1e-9)     # same risk as "95 % everywhere"
    return dict(frontier=frontier, uniform=uniform, base=base, same_risk=same_risk,
                saving=100 * (1 - same_risk["inv"] / base["inv"]))
