"""
main.py - run the whole prototype:   python main.py

Working prototype of
  Kawakami, Kobayashi & Nakata (2021), "Seasonal Inventory Management Model for
  Raw Materials in Steel Industry", INFORMS Journal on Applied Analytics 51(4).

PART 1 reproduces the paper's statistical inventory model on synthetic data.
PART 2 adds optimisation models (IP, LP sensitivity, BIP, multi-objective).
Results go to the console, to outputs/*.png and to outputs/Steel_Inventory_Prototype.xlsx
"""
import time

import config as C
import inventory_model
import optimisation
import report

MGR, PROP = "Manager (flat)", "Proposed (seasonal)"


def line(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def main():
    t0 = time.time()

    line("PART 1  Paper replication: two-stage arrival model and seasonal safety stock")
    groups, summary = inventory_model.run_all()
    for g, G in groups.items():
        aic = ", ".join(f"{a['order']} harm: {a['aic']:.1f}" for a in G["fit"]["aic_table"])
        print(f"\n{g}: {len(G['times'])} ships in {C.YEARS_HISTORY} years")
        print(f"  Stage 1 (NHPP, MLE by Newton): AIC [{aic}] -> {G['fit']['order']} harmonic(s) selected")
        print(f"  {'mill':8s}{'N_i':>6s}{'cycle':>8s}{'Z low':>8s}{'Z high':>8s}{'manager Z':>11s}{'KS p':>7s}   (days of usage)")
        for m, d in G["per_mill"].items():
            print(f"  {m:8s}{d['n_i']:6.2f}{d['cycle']:8.1f}{d['z_t'].min():8.1f}{d['z_t'].max():8.1f}"
                  f"{d['z_mgr']:11.1f}{d['ks_p']:7.2f}")
    print(f"\nAll mills, proposed inventory as % of manager policy, by month:")
    print("  " + "  ".join(f"{m} {v:.0f}" for m, v in zip(C.MONTHS, summary["pct_monthly"])))
    print(f"  -> reduction {summary['min_reduction']:.1f}% (disruption season) to {summary['max_reduction']:.1f}% "
          f"(calm season), {summary['avg_reduction']:.1f}% on average   [paper: 6% / 14% / 11%]")
    print(f"\nOut-of-sample stock-out rate per replenishment cycle (target {100 * (1 - C.SERVICE_LEVEL):.0f}%):")
    for pol, r in summary["stockout_rates"].items():
        print(f"  {pol:22s} disruption season {r['low']:5.1f}%   calm season {r['high']:5.1f}%")

    line("PART 2A  Shipping & yard plan  (integer programming + LP sensitivity)")
    print(f"solving 2 integer models (up to {C.MIP_TIME_LIMIT}s each) and 16 LPs ...")
    plans = optimisation.run_plans(groups)
    print(f"  {'policy':22s}{'holding':>10s}{'demurrage':>11s}{'penalty':>9s}{'TOTAL':>9s}{'avg stock':>11s}   [k$ / kt]")
    for pol, r in plans["mip"].items():
        print(f"  {pol:22s}{r['holding']:10.0f}{r['demurrage']:11.0f}{r['penalty']:9.0f}{r['total']:9.0f}"
              f"{r['avg_inv']:11.0f}   {r['status']}")
    a, b = plans["mip"][MGR], plans["mip"][PROP]
    print(f"  -> proposed policy saves US$ {(a['total'] - b['total']) / 1000:.1f} M per year, of which "
          f"US$ {(a['demurrage'] - b['demurrage']) / 1000:.1f} M is demurrage")
    print("  yard-capacity what-if (LP relaxation):  cost [k$]  and shadow price [k$ per kt of yard per year]")
    for s in plans["sensitivity"]:
        print(f"    yard x{s['scale']:.2f}:  manager {s[MGR]:7.0f} (shadow {s[MGR + ' shadow']:5.1f})"
              f"   proposed {s[PROP]:7.0f} (shadow {s[PROP + ' shadow']:5.1f})")

    line("PART 2B  Brand grouping  (binary programming, set partitioning)")
    grouping = optimisation.run_grouping()
    print(f"{grouping['n_candidates']} candidate groups, solver: {grouping['status']}")
    for r in grouping["rows"]:
        best = " | ".join("+".join(b.replace("Brand ", "B") for b in g) for g in r["optimal"])
        print(f"  {r['mill']}: {r['cost_original']:6.0f} kt -> {r['cost_optimal']:6.0f} kt   groups: {best}")
    print(f"  -> {grouping['reduction']:.1f}% less fine-ore stock than the original grouping   [paper: 18.1%]")

    line("PART 2C  Service level vs inventory  (multi-objective, epsilon-constraint)")
    pareto = optimisation.run_pareto(groups)
    print("  stock-outs/yr -> minimum total stock [kt]:")
    print("  " + "   ".join(f"{f['risk']:.0f}: {f['inv']:.0f}" for f in pareto["frontier"][::3]))
    print(f"  -> same risk as '{100 * C.SERVICE_LEVEL:g}% everywhere' with {pareto['saving']:.1f}% less stock "
          f"by choosing the service level per mill and brand group")

    line("Writing charts and Excel workbook")
    charts = report.make_charts(groups, summary, plans, grouping, pareto)
    book = report.make_workbook(groups, summary, plans, grouping, pareto, charts)
    for f in charts + [book]:
        print("  " + f)
    print(f"\nDone in {time.time() - t0:.0f} s.")


if __name__ == "__main__":
    main()
