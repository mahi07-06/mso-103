"""
report.py - turns the results into charts (PNG) and an Excel workbook.
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
from stats_utils import gamma_pdf

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
BLUE, RED, GREY, GREEN = "#1f5fa8", "#c0392b", "#7f8c8d", "#2e8b57"
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


# ============================================================================ charts
def make_charts(groups, summary, plans, grouping, pareto):
    files = []

    # 1. seasonal arrival intensity (paper Fig. 7)
    fig, axes = plt.subplots(1, len(groups), figsize=(4.6 * len(groups), 3.6))
    for ax, (g, G) in zip(axes, groups.items()):
        counts = np.bincount(MONTH_OF_DAY[(G["times"] % C.P).astype(int)], minlength=12)
        ax.bar(MONTH_START + 15, counts / (C.YEARS_HISTORY * np.array(C.MONTH_DAYS)), width=24,
               color="#d6e4f5", label="observed (10 yrs of ships)")
        ax.plot(G["lam_true"], color=GREY, ls="--", label="true rate (hidden)")
        ax.plot(G["fit"]["lam_day"], color=BLUE, lw=2, label=f"MLE fit, {G['fit']['order']} harmonic(s)")
        ax.set_title(g)
        ax.set_ylabel("ships arriving in Japan per day")
        _month_axis(ax)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Stage 1 (AAM): seasonal arrival rate λ(t) fitted by maximum likelihood", fontweight="bold")
    files.append(_save(fig, "fig1_seasonal_arrival_rate.png"))

    # 2. gamma fit of intervals at each mill (paper Fig. 6 / 9)
    g0 = next(iter(groups))
    G = groups[g0]
    fig, axes = plt.subplots(1, len(G["mills"]), figsize=(3.6 * len(G["mills"]), 3.3))
    for ax, (mill, d) in zip(axes, G["per_mill"].items()):
        gaps = d["gaps_tau"] / G["lam_bar"]                 # season-adjusted intervals [days]
        x = np.linspace(0, gaps.max() * 1.1, 200)
        ax.hist(gaps, bins=12, density=True, color="#d6e4f5", edgecolor="white")
        ax.plot(x, gamma_pdf(x * G["lam_bar"], d["n_i"]) * G["lam_bar"], color=BLUE, lw=2)
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
    ax.set_ylim(min(60, summary["pct_daily"].min() - 5), 105)
    _month_axis(ax)
    ax.set_title(f"Optimal inventory for all mills: {summary['min_reduction']:.0f}% lower in the disruption season, "
                 f"{summary['max_reduction']:.0f}% lower in the calm season", fontweight="bold", fontsize=10)
    ax.legend(loc="lower left")
    files.append(_save(fig, "fig3_seasonal_optimal_inventory.png"))

    # 4. out-of-sample stock-out validation
    fig, ax = plt.subplots(figsize=(7.5, 4))
    pols = [MGR, STAT, PROP]
    x = np.arange(2)
    for k, pol in enumerate(pols):
        r = summary["stockout_rates"][pol]
        bars = ax.bar(x + (k - 1) * 0.26, [r["low"], r["high"]], 0.25, color=COL[pol], label=pol)
        ax.bar_label(bars, fmt="%.1f%%", fontsize=8)
    ax.axhline(100 * (1 - C.SERVICE_LEVEL), color="black", ls=":", label="target stock-out rate")
    ax.set_xticks(x)
    ax.set_xticklabels(["disruption season\n(arrival rate below average)", "calm season\n(arrival rate above average)"])
    ax.set_ylabel("replenishment cycles with a stock-out [%]")
    ax.set_title(f"Validation on {C.YEARS_TEST} fresh simulated years", fontweight="bold")
    ax.legend(fontsize=8)
    files.append(_save(fig, "fig4_stockout_validation.png"))

    # 5. shipping plan: yard stock vs capacity
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    for ax, mill in zip(axes.ravel(), C.MILLS):
        for pol in (MGR, PROP):
            ax.plot(range(12), plans["mip"][pol]["mill_inv"][mill], marker="o", ms=3, color=COL[pol], label=pol)
        ax.axhline(plans["inputs"]["cap"][mill], color="black", ls="--", lw=1, label="yard capacity")
        ax.set_title(mill)
        ax.set_xticks(range(12))
        ax.set_xticklabels(C.MONTHS, fontsize=8)
        ax.set_ylabel("stock at month end [kt]")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Integer-programming shipping plan: stock above the yard capacity waits at sea (demurrage)",
                 fontweight="bold")
    files.append(_save(fig, "fig5_shipping_plan_yard.png"))

    # 6. cost comparison + yard-capacity sensitivity
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    bottom = np.zeros(2)
    for key, colr in (("holding", "#9db8d9"), ("demurrage", RED), ("penalty", "#f0b27a")):
        vals = np.array([plans["mip"][p][key] for p in (MGR, PROP)]) / 1000
        a1.bar([0, 1], vals, 0.5, bottom=bottom, color=colr, label=key)
        bottom += vals
    for k in range(2):
        a1.text(k, bottom[k], f"${bottom[k]:.1f}M", ha="center", va="bottom", fontweight="bold")
    a1.set_xticks([0, 1])
    a1.set_xticklabels([MGR, PROP])
    a1.set_ylabel("annual cost [million US$]")
    a1.set_title("Cost of the best integer plan found", fontweight="bold")
    a1.legend(fontsize=8)
    sc = [100 * r["scale"] for r in plans["sensitivity"]]
    for pol in (MGR, PROP):
        a2.plot(sc, [r[pol] / 1000 for r in plans["sensitivity"]], marker="o", color=COL[pol], label=pol)
    a2.set_xlabel("yard capacity [% of base case]")
    a2.set_ylabel("annual cost, LP relaxation [million US$]")
    a2.set_title("Sensitivity: slope of each line = shadow price of yard space", fontweight="bold", fontsize=10)
    a2.legend(fontsize=8)
    files.append(_save(fig, "fig6_cost_and_yard_sensitivity.png"))

    # 7. brand grouping
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(grouping["rows"]))
    for k, (key, lab, colr) in enumerate((("cost_individual", "every brand on its own", "#cfd8dc"),
                                          ("cost_original", "original grouping", RED),
                                          ("cost_optimal", "BIP-optimal grouping", BLUE))):
        ax.bar(x + (k - 1) * 0.27, [r[key] for r in grouping["rows"]], 0.26, color=colr, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([r["mill"] for r in grouping["rows"]])
    ax.set_ylabel("fine-ore stock required [kt]")
    ax.set_title(f"Brand grouping by binary programming: {grouping['reduction']:.0f}% less stock than the original groups",
                 fontweight="bold", fontsize=10)
    ax.legend(fontsize=8)
    files.append(_save(fig, "fig7_brand_grouping.png"))

    # 8. Pareto frontier
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot([f["risk"] for f in pareto["frontier"]], [f["inv"] for f in pareto["frontier"]],
            marker="o", color=BLUE, label="Pareto frontier (service level chosen per mill & group)")
    ax.scatter([u["risk"] for u in pareto["uniform"]], [u["inv"] for u in pareto["uniform"]],
               color=RED, marker="x", zorder=3, label="same service level everywhere")
    for u in pareto["uniform"]:
        ax.annotate(f"{100 * u['alpha']:g}%", (u["risk"], u["inv"]), textcoords="offset points",
                    xytext=(5, 5), fontsize=7, color=RED)
    ax.set_xlabel("expected stock-out cycles per year (all mills)")
    ax.set_ylabel("total stock [kt]")
    ax.set_title("Two objectives: less inventory vs fewer stock-outs", fontweight="bold")
    ax.legend(fontsize=8)
    files.append(_save(fig, "fig8_pareto_service_level.png"))
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


def make_workbook(groups, summary, plans, grouping, pareto, chart_files):
    wb = Workbook()

    # ------------------------------------------------------------------ Summary
    ws = wb.active
    ws.title = "Summary"
    ws["A1"] = "Seasonal Inventory Management Model for Raw Materials in Steel Industry - working prototype"
    ws["A1"].font = Font(bold=True, size=14)
    ws["A2"] = ("Kawakami, Kobayashi & Nakata (2021), INFORMS J. Applied Analytics 51(4). "
                "All data are SYNTHETIC (see config.py); results show the method, not Nippon Steel's real figures.")
    mip = plans["mip"]
    row = _table(ws, 4, ["Result", "Prototype", "Paper"], [
        ["Inventory reduction, disruption season", f"{summary['min_reduction']:.1f} %", "6 % (winter)"],
        ["Inventory reduction, calm season", f"{summary['max_reduction']:.1f} %", "14 % (summer)"],
        ["Inventory reduction, year average", f"{summary['avg_reduction']:.1f} %", "11 %"],
        ["Mill/brand-group pairs passing KS test (p > 0.05)",
         f"{sum(d['ks_p'] > 0.05 for G in groups.values() for d in G['per_mill'].values())} of "
         f"{sum(len(G['per_mill']) for G in groups.values())}", "most (1 brand group failed)"],
        ["Annual cost, manager policy (integer plan)", f"US$ {mip[MGR]['total'] / 1000:.1f} M", "-"],
        ["Annual cost, proposed policy (integer plan)", f"US$ {mip[PROP]['total'] / 1000:.1f} M", "-"],
        ["Demurrage saved per year", f"US$ {(mip[MGR]['demurrage'] - mip[PROP]['demurrage']) / 1000:.1f} M",
         "at least US$ 4.5 M"],
        ["Stock saved by optimal brand grouping (BIP)", f"{grouping['reduction']:.1f} %", "18.1 % (manual regrouping)"],
        ["Stock saved by optimising service levels at equal risk", f"{pareto['saving']:.1f} %", "not done (limitation)"],
    ])
    row = _table(ws, row, ["Model", "Course topic", "Where"], [
        ["Seasonal arrival rate λ(t), MLE by Newton's method", "Nonlinear programming", "inventory_model.py"],
        ["Shipping & yard plan with integer ships", "Integer programming, branch & bound", "optimisation.py (A)"],
        ["LP relaxation, yard shadow prices, what-if on capacity", "LP, duality, sensitivity analysis", "optimisation.py (A)"],
        ["Brand grouping as set partitioning", "Binary programming", "optimisation.py (B)"],
        ["Inventory vs stock-out risk, ε-constraint", "Multi-objective optimisation", "optimisation.py (C)"],
    ])
    _table(ws, row, ["Key assumption", "Value"], [
        ["Service level α", C.SERVICE_LEVEL], ["Lot size (one ship) [kt]", C.LOT_KT],
        ["Manager policy: flat, worst season + margin", C.MANAGER_MARGIN],
        ["Holding cost [US$/t-month]", C.HOLDING_COST], ["Demurrage cost [US$/t-month]", C.DEMURRAGE_COST],
        ["Shortfall penalty [US$/t-month]", C.SHORTFALL_PENALTY],
        ["Yard capacity / manager-policy stock", C.YARD_TIGHTNESS],
        ["Demand", "constant (as in the paper)"], ["History used", f"{C.YEARS_HISTORY} yrs seasonality, {C.YEARS_RECENT} yrs mill ratio"],
    ])
    _widths(ws, 52, 34)

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
    row = _table(ws, row, ["Policy (stock-out % of cycles, out of sample)", "Disruption season", "Calm season"], rows)
    rows = []
    for g, G in groups.items():
        t = G["fit"]["aic_table"]
        rows += [[g, a["order"], a["n_params"], a["loglik"], a["aic"], a["newton_iters"],
                  "selected" if a["order"] == G["fit"]["order"] else ""] for a in t]
    _table(ws, row, ["NHPP fit: brand group", "Harmonics", "Parameters", "Log-likelihood", "AIC", "Newton iterations", ""], rows)
    _widths(ws, 44, 16)

    # ------------------------------------------------------------ Shipping plan
    ws = wb.create_sheet("Shipping plan")
    rows = [[pol, r["status"], r["holding"], r["demurrage"], r["penalty"], r["total"], r["avg_inv"],
             plans["lp"][pol]["total"]] for pol, r in mip.items()]
    row = _table(ws, 1, ["Policy", "Solver status", "Holding [k$]", "Demurrage [k$]", "Shortfall penalty [k$]",
                         "Total cost [k$]", "Average stock [kt]", "LP relaxation bound [k$]"], rows, "#,##0")
    for pol, r in mip.items():
        ws.cell(row=row, column=1, value=f"{pol}: ships unloaded per month").font = Font(bold=True)
        row = _table(ws, row + 1, ["Brand group / mill"] + C.MONTHS + ["Year"],
                     [[f"{g} - {m}"] + [int(round(v)) for v in s] + [int(round(s.sum()))] for (g, m), s in r["ships"].items()])
        row = _table(ws, row, ["Yard stock at month end [kt]"] + C.MONTHS + ["Capacity"],
                     [[m] + list(r["mill_inv"][m]) + [r["cap"][m]] for m in C.MILLS], "#,##0")
        row = _table(ws, row, ["Cargo waiting at sea [kt]"] + C.MONTHS,
                     [[m] + list(r["wait"][m]) for m in C.MILLS], "#,##0")
    _table(ws, row, ["Yard capacity (x base)", "Manager cost [k$]", "Proposed cost [k$]", "Manager demurrage [k$]",
                     "Proposed demurrage [k$]", "Manager shadow price [k$/kt-yr]", "Proposed shadow price [k$/kt-yr]"],
           [[s["scale"], s[MGR], s[PROP], s[MGR + " demurrage"], s[PROP + " demurrage"], s[MGR + " shadow"],
             s[PROP + " shadow"]] for s in plans["sensitivity"]], "#,##0.00")
    _widths(ws, 40, 13)

    # ----------------------------------------------------------- Brand grouping
    ws = wb.create_sheet("Brand grouping")
    fmt = lambda grp: "  |  ".join("+".join(b.replace("Brand ", "B") for b in g) for g in grp)
    row = _table(ws, 1, ["Mill", "Original groups", "BIP-optimal groups", "Stock individual [kt]",
                         "Stock original [kt]", "Stock optimal [kt]", "Reduction"],
                 [[r["mill"], fmt(r["original"]), fmt(r["optimal"]), r["cost_individual"], r["cost_original"],
                   r["cost_optimal"], f"{100 * (1 - r['cost_optimal'] / r['cost_original']):.1f} %"] for r in grouping["rows"]]
                 + [["All mills", "", "", grouping["total_individual"], grouping["total_original"],
                     grouping["total_optimal"], f"{grouping['reduction']:.1f} %"]], "#,##0")
    _table(ws, row, ["Brand", "Fe grade [%]"] + [f"Ships/yr {m}" for m in C.MILLS],
           [[b, fe] + q for b, (fe, q) in C.FINE_BRANDS.items()], "0.0")
    _widths(ws, 14, 16)
    ws.column_dimensions["B"].width = ws.column_dimensions["C"].width = 44

    # ------------------------------------------------------------------- Pareto
    ws = wb.create_sheet("Pareto")
    row = _table(ws, 1, ["Risk limit ε [stock-outs/yr]", "Stock-outs/yr achieved", "Total stock [kt]"],
                 [[f["eps"], f["risk"], f["inv"]] for f in pareto["frontier"]], "#,##0.0")
    row = _table(ws, row, ["Same service level everywhere", "Stock-outs/yr", "Total stock [kt]"],
                 [[u["alpha"], u["risk"], u["inv"]] for u in pareto["uniform"]], "#,##0.000")
    _table(ws, row, [f"Optimal service levels at the risk of 'α = {C.SERVICE_LEVEL} everywhere'", "Mill", "α chosen"],
           [[g, m, a] for (g, m), a in pareto["same_risk"]["alphas"].items()], "0.000")
    _widths(ws, 44, 22)

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
