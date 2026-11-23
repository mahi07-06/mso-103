"""
report.py - turns the results into charts (PNG) and an Excel workbook.
Nothing here changes the results; it only draws them.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from openpyxl import Workbook
from openpyxl.chart import LineChart, Reference
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

import config as C
from scipy import stats

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
BLUE, RED, GREY = "#1f5fa8", "#c0392b", "#7f8c8d"
MGR, PROP, STAT = "Manager (flat)", "Proposed (seasonal)", "No seasonality"
COL = {MGR: RED, PROP: BLUE, STAT: GREY}
MONTH_START = np.concatenate([[0], np.cumsum(C.MONTH_DAYS)[:-1]])
MONTH_OF_DAY = np.repeat(np.arange(12), C.MONTH_DAYS)

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 130})


def _month_axis(ax):
    ax.set_xticks(MONTH_START + 15)
    ax.set_xticklabels(C.MONTHS)
    ax.set_xlim(0, C.P)


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def simulate_trace(group, mill_name, days=C.P):
    """
    Day-by-day stock at one mill for one test year (for the simulation chart).
    Each ship tops the yard up to the planned level; then stock falls 1 day per day.
    Returns the time grid and the stock left under each policy, in days of usage.
    """
    k = group["mills"].index(mill_name)
    d = group["per_mill"][mill_name]
    arrivals = group["test_times"][(group["test_dest"] == k) & (group["test_times"] < days + 60)]
    t = np.arange(arrivals[0], days, 0.25)
    last = arrivals[np.searchsorted(arrivals, t, side="right") - 1]      # last ship before t
    start_day = (last % C.P).astype(int)
    level = {PROP: d["k_hi"] / group["lam_now"][start_day],                 # D(t) for this season
             MGR: np.full(len(t), d["k_md"] / group["lam_bar"] + d["z_mgr"])}
    return t, {p: level[p] - (t - last) for p in level}


# ============================================================================ charts
def make_charts(groups, summary, grouping):
    files = []

    # 1. seasonal arrival intensity (paper Fig. 7)
    fig, axes = plt.subplots(1, len(groups), figsize=(4.6 * len(groups), 3.6))
    for ax, (g, G) in zip(axes, groups.items()):
        counts = np.bincount(MONTH_OF_DAY[(G["times"] % C.P).astype(int)], minlength=12)
        ax.bar(MONTH_START + 15, counts / (C.YEARS_HISTORY * np.array(C.MONTH_DAYS)), width=24,
               color="#d6e4f5", label="observed (10 yrs of ships)")
        ax.plot(G["lam_true"], color=GREY, ls="--", label="true rate (hidden)")
        ax.plot(G["fit"]["lam_day"], color=BLUE, lw=2, label=f"PoissonRegressor fit, {G['fit']['order']} harmonic(s)")
        ax.set_title(g)
        ax.set_ylabel("ships arriving in Japan per day")
        ax.set_ylim(0, None)
        _month_axis(ax)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Stage 1 (AAM): seasonal arrival rate λ(t) fitted by Poisson regression", fontweight="bold")
    files.append(_save(fig, "fig1_seasonal_arrival_rate.png"))

    # 2. gamma fit of intervals at each mill (paper Fig. 6 / 9)
    g0 = next(iter(groups))
    G = groups[g0]
    fig, axes = plt.subplots(1, len(G["mills"]), figsize=(3.6 * len(G["mills"]), 3.3))
    for ax, (mill, d) in zip(axes, G["per_mill"].items()):
        gaps = d["gaps_tau"] / G["lam_bar"]                 # season-adjusted intervals [days]
        x = np.linspace(0, gaps.max() * 1.1, 200)
        ax.hist(gaps, bins=12, density=True, color="#d6e4f5", edgecolor="white")
        ax.plot(x, stats.gamma.pdf(x * G["lam_bar"], d["n_i"]) * G["lam_bar"], color=BLUE, lw=2)
        ax.set_title(f"{mill}: Gamma(N={d['n_i']:.2f})\nKS p-value = {d['ks_p']:.2f}", fontsize=9)
        ax.set_xlabel("interval between ships [days]")
    fig.suptitle(f"Stage 2 (AEM): arrival intervals at each mill vs Gamma model - {g0}", fontweight="bold")
    files.append(_save(fig, "fig2_gamma_fit_per_mill.png"))

    # 3. seasonal optimal inventory vs manager policy (paper Fig. 8)
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.axhline(100, color=RED, ls="--", lw=2, label="Manager policy (flat)")
    ax.plot(summary["pct_daily"], color=BLUE, lw=2.5, label="Proposed seasonal model")
    ax.fill_between(np.arange(C.P), summary["pct_daily"], 100, color=BLUE, alpha=0.08)
    ax.set_ylabel("proper inventory, all mills [% of manager policy]")
    ax.set_ylim(min(70, summary["pct_daily"].min() - 5), 105)
    _month_axis(ax)
    ax.set_title(f"Optimal inventory for all mills: {summary['winter_reduction']:.0f}% lower in winter, "
                 f"{summary['summer_reduction']:.0f}% lower in summer", fontweight="bold", fontsize=10)
    ax.legend(loc="lower left")
    files.append(_save(fig, "fig3_seasonal_optimal_inventory.png"))

    # 4. stock-out rate by policy and season (paper Tables 2-3)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    pols = [MGR, STAT, PROP]
    x = np.arange(2)
    for k, pol in enumerate(pols):
        r = summary["stockout_rates"][pol]
        bars = ax.bar(x + (k - 1) * 0.26, [r["low"], r["high"]], 0.25, color=COL[pol], label=pol)
        ax.bar_label(bars, fmt="%.1f%%", fontsize=8)
    ax.axhline(100 * (1 - C.SERVICE_LEVEL), color="black", ls=":", label="target stock-out rate (5%)")
    ax.set_xticks(x)
    ax.set_xticklabels(["disruption season\n(fewer ships than average)", "calm season\n(more ships than average)"])
    ax.set_ylabel("replenishment cycles with a stock-out [%]")
    ax.set_title(f"Inventory simulation on {C.YEARS_TEST} fresh years: shortage ratio", fontweight="bold")
    ax.legend(fontsize=8)
    files.append(_save(fig, "fig4_stockout_validation.png"))

    # 5. day-by-day inventory simulation at one mill (paper Fig. 10)
    mill0 = G["mills"][0]
    t, stock = simulate_trace(G, mill0)
    fig, ax = plt.subplots(figsize=(9.5, 4))
    for pol in (MGR, PROP):
        ax.plot(t, stock[pol], color=COL[pol], lw=1.4, label=pol)
    ax.fill_between(t, stock[PROP], 0, where=stock[PROP] < 0, color=BLUE, alpha=0.35,
                    label="stock-out (proposed)")
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("stock left [days of usage]")
    _month_axis(ax)
    ax.set_title(f"Inventory simulation, {g0} at {mill0} (one test year): below 0 = stock-out",
                 fontweight="bold", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    files.append(_save(fig, "fig5_inventory_simulation.png"))

    # 6. brand grouping (paper Tables 6-7)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    rows = grouping["rows"]
    x = np.arange(len(rows) + 1)
    orig = [r["index_original"] for r in rows] + [grouping["index_original"]]
    rev = [r["index_revised"] for r in rows] + [grouping["index_revised"]]
    b1 = ax.bar(x - 0.2, orig, 0.38, color=RED, label="original brand groups")
    b2 = ax.bar(x + 0.2, rev, 0.38, color=BLUE, label="revised brand groups")
    ax.bar_label(b1, fmt="%.0f", fontsize=8)
    ax.bar_label(b2, fmt="%.0f", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([r["mill"] for r in rows] + ["All mills"])
    ax.set_ylabel("fine-ore holding days (all mills, original = 100)")
    ax.set_title(f"Revised brand grouping: {grouping['reduction']:.1f}% fewer holding days",
                 fontweight="bold")
    ax.legend(fontsize=8)
    files.append(_save(fig, "fig6_brand_grouping.png"))
    return files


# ============================================================================= Excel
HEAD_FILL = PatternFill("solid", fgColor="1F3A5F")
INPUT_FILL = PatternFill("solid", fgColor="FFF2CC")


def _table(ws, row, headers, rows, fmt="0.00"):
    for c, h in enumerate(headers, 1):
        cell = ws.cell(row=row, column=c, value=h)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = HEAD_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for r, values in enumerate(rows, row + 1):
        for c, v in enumerate(values, 1):
            if isinstance(v, (np.floating, np.integer)):
                v = float(v)
            cell = ws.cell(row=r, column=c, value=v)
            if isinstance(v, float):
                cell.number_format = fmt
    return row + len(rows) + 2


def _widths(ws, first=30, rest=13):
    ws.column_dimensions["A"].width = first
    for c in range(2, 30):
        ws.column_dimensions[get_column_letter(c)].width = rest


def make_workbook(groups, summary, grouping, chart_files):
    wb = Workbook()
    n_pairs = sum(len(G["per_mill"]) for G in groups.values())
    n_pass = sum(d["ks_p"] > 0.05 for G in groups.values() for d in G["per_mill"].values())
    so = summary["stockout_rates"][PROP]

    # ------------------------------------------------------------------ Summary
    ws = wb.active
    ws.title = "Summary"
    ws["A1"] = "Seasonal Inventory Management Model for Raw Materials in Steel Industry - working prototype"
    ws["A1"].font = Font(bold=True, size=14)
    ws["A2"] = ("Kawakami, Kobayashi & Nakata (2021), INFORMS J. Applied Analytics 51(4). "
                "All data are SYNTHETIC (see config.py); results show the method, not Nippon Steel's real figures.")
    row = _table(ws, 4, ["Result", "Prototype", "Paper"], [
        ["Inventory reduction in winter", f"{summary['winter_reduction']:.1f} %", "6 %"],
        ["Inventory reduction in summer", f"{summary['summer_reduction']:.1f} %", "14 %"],
        ["Inventory reduction, year average", f"{summary['avg_reduction']:.1f} %", "11 %"],
        ["Mill/brand-group pairs passing KS test (p > 0.05)", f"{n_pass} of {n_pairs}",
         "most (brand group 7 failed)"],
        ["Shortage ratio in simulation (target 5 %)", f"{so['low']:.1f} % / {so['high']:.1f} %",
         "within threshold except brand group 7"],
        ["Holding days saved by revised brand grouping", f"{grouping['reduction']:.1f} %", "18.1 %"],
    ])
    row = _table(ws, row, ["Step", "What it does", "Presentation slide"], [
        ["STEP 1", "Seasonal arrival rate λ(t) by Poisson regression (sklearn), harmonics by AIC", "Methodology: AAM, seasonality"],
        ["STEP 2", "Every N_i-th ship goes to mill i, gap ~ Gamma(N_i, λ)", "Methodology: AEM"],
        ["STEP 3", "Z = G⁻¹(α) − G⁻¹(50%),  optimal stock = cycle + Z", "Calculating Safety Stock (Z)"],
        ["STEP 4", "Kolmogorov-Smirnov test of the gamma model", "Validation: statistical"],
        ["STEP 5", "Inventory simulation on fresh years, shortage ratio", "Validation: simulation"],
        ["STEP 6", "All mills together, % of manager policy", "Results: 14 % summer / 6 % winter"],
        ["STEP 7", "Original vs revised brand grouping", "Results: revised brand grouping"],
    ])
    _table(ws, row, ["Key assumption", "Value"], [
        ["Service level α", C.SERVICE_LEVEL], ["Lot size (one ship) [kt]", C.LOT_KT],
        ["Manager policy: flat, worst season + margin", C.MANAGER_MARGIN],
        ["Demand", "constant (as in the paper)"],
        ["History used", f"{C.YEARS_HISTORY} yrs seasonality, {C.YEARS_RECENT} yrs mill ratio"],
        ["Scale", f"{len(C.MILLS)} mills, {len(C.BRAND_GROUPS)} brand groups (paper: 8 and 10)"],
    ])
    _widths(ws, 52, 40)

    # --------------------------------------------------------------- Calculator
    ws = wb.create_sheet("Calculator")
    g0 = next(iter(groups))
    G = groups[g0]
    m0 = G["mills"][0]
    d0 = G["per_mill"][m0]
    ws["A1"] = "Live safety-stock calculator (paper eq. 2-4, 8) - change the yellow cells"
    ws["A1"].font = Font(bold=True, size=13)
    ws["A2"] = f"Pre-filled with {g0} at {m0}.  Excel function GAMMAINV is the G^-1 of the paper."
    inputs = [("Ships per year arriving in Japan (Q)", C.BRAND_GROUPS[g0]["ships_per_year"]),
              ("Ships per year allocated to this mill (q_i)", round(C.BRAND_GROUPS[g0]["ships_per_year"] / d0["n_i"], 1)),
              ("Service level α", C.SERVICE_LEVEL), ("Lot size [kt]", C.LOT_KT),
              ("Manager margin over worst season", C.MANAGER_MARGIN)]
    for k, (lab, v) in enumerate(inputs):
        ws.cell(row=4 + k, column=1, value=lab)
        ws.cell(row=4 + k, column=2, value=v).fill = INPUT_FILL
    derived = [("N_i = Q / q_i  (every N-th ship comes here)", "=B4/B5"),
               ("Daily usage [kt/day]", "=B5*B7/365"),
               ("Cycle inventory = half a lot [days of usage]", "=365/B5/2")]
    for k, (lab, f) in enumerate(derived):
        ws.cell(row=10 + k, column=1, value=lab)
        ws.cell(row=10 + k, column=2, value=f).number_format = "0.00"
    heads = ["Month", "Seasonal factor λ(t)/λ̄ (fitted)", "λ(t) [ships/day]", "A = G⁻¹(50%) [days]",
             "D = G⁻¹(α) [days]", "Safety stock Z = D − A [days]", "Optimal stock I = cycle + Z [days]",
             "Optimal stock [kt]", "Manager policy [days]", "Reduction", "Python check: I [days]"]
    factor = [float((G["lam_now"] / G["lam_bar"])[MONTH_OF_DAY == k].mean()) for k in range(12)]
    _table(ws, 14, heads, [])
    for k in range(12):
        r = 15 + k
        ws.cell(row=r, column=1, value=C.MONTHS[k])
        ws.cell(row=r, column=2, value=round(factor[k], 4)).fill = INPUT_FILL
        ws.cell(row=r, column=3, value=f"=$B$4/365*B{r}")
        ws.cell(row=r, column=4, value=f"=GAMMAINV(0.5,$B$10,1/C{r})")
        ws.cell(row=r, column=5, value=f"=GAMMAINV($B$6,$B$10,1/C{r})")
        ws.cell(row=r, column=6, value=f"=E{r}-D{r}")
        ws.cell(row=r, column=7, value=f"=$B$12+F{r}")
        ws.cell(row=r, column=8, value=f"=G{r}*$B$11")
        ws.cell(row=r, column=9, value="=$B$12+(1+$B$8)*MAX($F$15:$F$26)")
        ws.cell(row=r, column=10, value=f"=1-G{r}/I{r}")
        py = d0["cycle"] + (d0["k_hi"] - d0["k_md"]) / (G["lam_bar"] * factor[k])
        ws.cell(row=r, column=11, value=float(py))
        for c in range(3, 12):
            ws.cell(row=r, column=c).number_format = "0.0%" if c == 10 else "0.00"
    ws.row_dimensions[14].height = 48
    chart = LineChart()
    chart.title = "Optimal stock through the year [days of usage]"
    chart.add_data(Reference(ws, min_col=7, min_row=14, max_row=26), titles_from_data=True)
    chart.add_data(Reference(ws, min_col=9, min_row=14, max_row=26), titles_from_data=True)
    chart.set_categories(Reference(ws, min_col=1, min_row=15, max_row=26))
    chart.height, chart.width = 8, 18
    ws.add_chart(chart, "A29")
    _widths(ws, 46, 17)

    # ------------------------------------------------------------- Safety stock
    ws = wb.create_sheet("Safety stock")
    rows = []
    for g, G in groups.items():
        for m, d in G["per_mill"].items():
            monthly = [d["inv_prop"][MONTH_OF_DAY == k].mean() for k in range(12)]
            rows.append([g, m, d["q_total"], d["q_i"], d["n_i"], d["usage"], d["cycle"], d["z_t"].min(),
                         d["z_t"].max(), d["inv_stat"], d["inv_mgr"]] + monthly)
    _table(ws, 1, ["Brand group", "Mill", "Q (2 yrs)", "q_i (2 yrs)", "N_i", "Usage [kt/day]", "Cycle [days]",
                   "Z min [days]", "Z max [days]", "Stock, no seasonality [days]", "Stock, manager [days]"]
           + [f"Proposed {m} [days]" for m in C.MONTHS], rows)
    r0 = len(rows) + 4
    _table(ws, r0, ["All mills, % of manager policy"] + C.MONTHS, [["Proposed model"] + list(summary["pct_monthly"])], "0.0")
    ws.row_dimensions[1].height = 45
    _widths(ws, 30, 12)

    # --------------------------------------------------------------- Validation
    ws = wb.create_sheet("Validation")
    rows = [[g, m, len(d["gaps_tau"]), d["ks_d"], d["ks_p"], "pass" if d["ks_p"] > 0.05 else "REJECTED"]
            for g, G in groups.items() for m, d in G["per_mill"].items()]
    row = _table(ws, 1, ["Brand group", "Mill", "Intervals tested", "KS statistic D", "KS p-value", "Gamma model"], rows, "0.000")
    rows = [[pol, r["low"], r["high"]] for pol, r in summary["stockout_rates"].items()]
    row = _table(ws, row, ["Policy (shortage ratio %, simulation)", "Disruption season", "Calm season"], rows)
    rows = []
    for g, G in groups.items():
        t = G["fit"]["aic_table"]
        rows += [[g, a["order"], a["n_params"], a["loglik"], a["aic"], a["iterations"],
                  "selected" if a["order"] == G["fit"]["order"] else ""] for a in t]
    _table(ws, row, ["NHPP fit: brand group", "Harmonics", "Parameters", "Log-likelihood", "AIC", "Solver iterations", ""], rows)
    _widths(ws, 44, 16)

    # ----------------------------------------------------------- Brand grouping
    ws = wb.create_sheet("Brand grouping")
    fmt = lambda grp: "  |  ".join("+".join(b.replace("Brand ", "B") for b in g) for g in grp)
    row = _table(ws, 1, ["Grouping", "Groups"], [["Original (paper Table 4)", fmt(C.ORIGINAL_GROUPS)],
                                                  ["Revised (paper Table 5)", fmt(C.REVISED_GROUPS)]])
    row = _table(ws, row, ["Mill", "Stock original [kt]", "Stock revised [kt]",
                           "Holding days original (index)", "Holding days revised (index)"],
                 [[r["mill"], r["stock_original"], r["stock_revised"], r["index_original"], r["index_revised"]]
                  for r in grouping["rows"]]
                 + [["All mills", grouping["total_original"], grouping["total_revised"],
                     grouping["index_original"], grouping["index_revised"]]], "#,##0.0")
    _table(ws, row, ["Brand"] + [f"Ships/yr {m}" for m in C.MILLS],
           [[b] + q for b, q in C.FINE_BRANDS.items()], "0")
    _widths(ws, 26, 22)
    ws.column_dimensions["B"].width = 50

    # ------------------------------------------------------------------- Charts
    ws = wb.create_sheet("Charts")
    row = 1
    for f in chart_files:
        img = XLImage(f)
        scale = 720 / img.width
        img.width, img.height = 720, int(img.height * scale)
        ws.add_image(img, f"A{row}")
        row += int(img.height / 20) + 2

    path = os.path.join(OUT, "Steel_Inventory_Prototype.xlsx")
    wb.save(path)
    return path
