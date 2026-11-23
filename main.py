"""
main.py - run the whole prototype:   python main.py

Working prototype of
  Kawakami, Kobayashi & Nakata (2021), "Seasonal Inventory Management Model for
  Raw Materials in Steel Industry", INFORMS Journal on Applied Analytics 51(4).

It follows our presentation slide by slide, on synthetic data:
  STEP 1-2  two-stage model (AAM seasonal Poisson, AEM gamma)
  STEP 3    safety stock and optimal stock
  STEP 4-5  validation (KS test, inventory simulation)
  STEP 6    results: seasonal inventory vs manager policy
  STEP 7    results: revised brand grouping
Results go to the console, to outputs/*.png and to outputs/Steel_Inventory_Prototype.xlsx
"""
import time

import config as C
import inventory_model
import brand_grouping
import report

MGR, PROP = "Manager (flat)", "Proposed (seasonal)"


def heading(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def main():
    t0 = time.time()
    groups, summary = inventory_model.run_all()

    heading("STEP 1-3  Two-stage arrival model and safety stock (slides: Methodology, Safety Stock)")
    for g, G in groups.items():
        aic = ", ".join(f"{a['order']}: {a['aic']:.1f}" for a in G["fit"]["aic_table"])
        print(f"\n{g}: {len(G['times'])} ships in {C.YEARS_HISTORY} years")
        print(f"  Stage 1 (AAM): AIC by number of harmonics [{aic}] -> {G['fit']['order']} selected")
        print(f"  Stage 2 (AEM) and safety stock, in days of usage:")
        print(f"  {'mill':8s}{'N_i':>6s}{'cycle':>8s}{'Z low':>8s}{'Z high':>8s}{'manager Z':>11s}")
        for m, d in G["per_mill"].items():
            print(f"  {m:8s}{d['n_i']:6.2f}{d['cycle']:8.1f}{d['z_t'].min():8.1f}{d['z_t'].max():8.1f}"
                  f"{d['z_mgr']:11.1f}")

    heading("STEP 4  Statistical validation: Kolmogorov-Smirnov test (slide: Validation)")
    n_pass = 0
    for g, G in groups.items():
        for m, d in G["per_mill"].items():
            ok = d["ks_p"] > 0.05
            n_pass += ok
            print(f"  {g:22s} {m:8s} p-value {d['ks_p']:.2f}  {'pass' if ok else 'REJECTED'}")
    n_pairs = sum(len(G["per_mill"]) for G in groups.values())
    print(f"  -> {n_pass} of {n_pairs} pass   [paper: most pass, brand group 7 fails]")

    heading("STEP 5  Simulation validation: shortage ratio on fresh data (slide: Validation)")
    print(f"  share of replenishment cycles with a stock-out (target {100 * (1 - C.SERVICE_LEVEL):.0f}%):")
    for pol, r in summary["stockout_rates"].items():
        print(f"  {pol:22s} disruption season {r['low']:5.1f}%   calm season {r['high']:5.1f}%")

    heading("STEP 6  Results: seasonal inventory vs manager policy (slide: Results)")
    print("  all mills, proposed inventory as % of manager policy, by month:")
    print("  " + "  ".join(f"{m} {v:.0f}" for m, v in zip(C.MONTHS, summary["pct_monthly"])))
    print(f"  -> {summary['winter_reduction']:.1f}% less in winter, {summary['summer_reduction']:.1f}% less in summer, "
          f"{summary['avg_reduction']:.1f}% on average   [paper: 6% / 14% / 11%]")

    heading("STEP 7  Results: revised brand grouping (slide: Results)")
    grouping = brand_grouping.run_grouping()
    print(f"  {'mill':8s}{'original':>10s}{'revised':>10s}   (fine-ore holding days, all mills original = 100)")
    for r in grouping["rows"]:
        print(f"  {r['mill']:8s}{r['index_original']:10.1f}{r['index_revised']:10.1f}")
    print(f"  {'All':8s}{grouping['index_original']:10.1f}{grouping['index_revised']:10.1f}")
    print(f"  -> {grouping['reduction']:.1f}% fewer holding days   [paper: 18.1%]")

    heading("Writing charts and Excel workbook")
    charts = report.make_charts(groups, summary, grouping)
    book = report.make_workbook(groups, summary, grouping, charts)
    for f in charts + [book]:
        print("  " + f)
    print(f"\nDone in {time.time() - t0:.0f} s.")


if __name__ == "__main__":
    main()
