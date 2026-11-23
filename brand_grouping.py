"""
brand_grouping.py  -  STEP 7: what happens to stock if we group brands differently?
(slide "Results": revised brand grouping, paper Tables 4 to 7)

The idea: if several similar brands share one stockpile, ships for that pile
come more often, so the gap between ships is shorter and less safety stock is needed.

There is no optimiser here, just the paper's formula applied twice:
    1. stock needed with the ORIGINAL grouping
    2. stock needed with the REVISED grouping
and the difference in holding days, normalised so the original total = 100.
"""
import config as C
from stats_utils import gamma_ppf


def stock_needed(brands, mill_index):
    """
    Stock in kt that one mill needs if `brands` are managed as ONE group.
    Same formula as inventory_model.py:  cycle stock + safety stock.
    """
    ships_to_japan = sum(sum(C.FINE_BRANDS[b]) for b in brands)          # Q per year
    ships_to_mill = sum(C.FINE_BRANDS[b][mill_index] for b in brands)    # q_i per year
    if ships_to_mill == 0:
        return 0.0

    n_i = ships_to_japan / ships_to_mill            # every N-th ship comes to this mill
    arrival_rate = ships_to_japan / C.P             # ships per day arriving in Japan
    late_gap = gamma_ppf(C.SERVICE_LEVEL, n_i)      # D, for rate 1
    normal_gap = gamma_ppf(0.50, n_i)               # A, for rate 1
    safety_days = (late_gap - normal_gap) / arrival_rate

    usage = ships_to_mill * C.LOT_KT / C.P          # kt burned per day
    cycle_stock = 0.5 * C.LOT_KT                    # half a ship, on average
    return cycle_stock + usage * safety_days


def stock_for_grouping(groups, mill_index):
    """Total stock at one mill for a whole grouping (a list of groups)."""
    total = 0.0
    for group in groups:
        brands_here = [b for b in group if C.FINE_BRANDS[b][mill_index] > 0]
        if brands_here:
            total += stock_needed(brands_here, mill_index)
    return total


def run_grouping():
    rows = []
    for mill_index, mill in enumerate(C.MILLS):
        usage = sum(C.FINE_BRANDS[b][mill_index] for b in C.FINE_BRANDS) * C.LOT_KT / C.P
        original = stock_for_grouping(C.ORIGINAL_GROUPS, mill_index)
        revised = stock_for_grouping(C.REVISED_GROUPS, mill_index)
        rows.append(dict(mill=mill, usage=usage,
                         stock_original=original, stock_revised=revised,
                         days_original=original / usage, days_revised=revised / usage))

    total_usage = sum(r["usage"] for r in rows)
    total_original = sum(r["stock_original"] for r in rows)
    total_revised = sum(r["stock_revised"] for r in rows)

    # Like the paper: holding days, with the original all-mill total set to 100
    scale = 100.0 / (total_original / total_usage)
    for r in rows:
        r["index_original"] = r["days_original"] * scale
        r["index_revised"] = r["days_revised"] * scale

    return dict(rows=rows,
                total_original=total_original, total_revised=total_revised,
                index_original=100.0, index_revised=100.0 * total_revised / total_original,
                reduction=100.0 * (1 - total_revised / total_original))
