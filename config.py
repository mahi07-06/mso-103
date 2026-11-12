"""
config.py  -  every assumption of the prototype lives here.

The real Nippon Steel shipping records are confidential, so the prototype
works on SYNTHETIC data generated from the numbers below.  Change anything
here and re-run main.py: all results, charts and the Excel workbook update.
"""

SEED = 42
P = 365                      # length of the seasonal cycle [days]
YEARS_HISTORY = 10           # paper: 10 years of data for seasonality (AAM)
YEARS_RECENT = 2             # paper: 2 years of data for the mill ratio N_i (AEM)
YEARS_TEST = 30              # fresh out-of-sample years used to validate policies

SERVICE_LEVEL = 0.95         # alpha: P(no stock-out in a replenishment cycle)

# "Manager policy" (the conventional rule the paper compares against).
# ASSUMPTION: one flat level all year, sized for the worst season,
# plus a judgement margin.  The paper does not publish the real rule.
MANAGER_MARGIN = 0.05

LOT_KT = 170.0               # one Capesize cargo [thousand tonnes], fixed lot size

MILLS = ["Mill A", "Mill B", "Mill C", "Mill D"]

# Brand groups.  ships_per_year = this year's contracted cargoes to Japan.
# Seasonality (true process, unknown to the model - it must be estimated):
#   log lambda(t) = c - amp1*cos(2*pi*(t-dip_day)/P) - amp2*cos(4*pi*(t-dip_day)/P)
# dip_day = day of year when arrivals in Japan are at their lowest.
# shares  = fraction of the group's ships allocated to each mill.
BRAND_GROUPS = {
    "Fine ore (Australia)": dict(
        ships_per_year=120, amp1=0.16, amp2=0.00, dip_day=75,    # cyclones Feb-Mar
        shares={"Mill A": 0.35, "Mill B": 0.30, "Mill C": 0.20, "Mill D": 0.15}),
    "Lump ore (Australia)": dict(
        ships_per_year=60, amp1=0.18, amp2=0.00, dip_day=70,
        shares={"Mill A": 0.40, "Mill B": 0.35, "Mill C": 0.25}),
    "Pellet (Brazil)": dict(
        ships_per_year=40, amp1=0.22, amp2=0.06, dip_day=110,    # rainy season + 40-day voyage
        shares={"Mill B": 0.50, "Mill C": 0.30, "Mill D": 0.20}),
}

# Probability that the operator overrides the regular "every N-th ship" rotation
# (human intervention in ship allocation - the reason one brand group failed in the paper).
ALLOCATION_NOISE = 0.05

# ---------------------------------------------------------------- optimisation
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# Costs, in thousand US$ per thousand tonnes per month  (= US$ per tonne-month)
HOLDING_COST = 0.7           # ~8 %/yr capital cost on ~US$100/t ore
DEMURRAGE_COST = 3.5         # ~US$20,000/day for a 170 kt ship waiting offshore
SHORTFALL_PENALTY = 25.0     # stock below the safety target (blast-furnace risk)

# Yard capacity of each mill [kt].  ASSUMPTION: the yard was sized for the stock the
# conventional policy wants to hold, with no spare room (the premise of the paper).
# main.py also re-solves the plan for yards 15 % smaller ... 20 % larger.
YARD_TIGHTNESS = 1.00        # capacity = this x (sum of the manager-policy targets)

MIP_TIME_LIMIT = 30          # seconds of branch & bound allowed per integer model

# ---------------------------------------------------- brand-grouping BIP (fine ore)
# Individual fine-ore brands: Fe grade [%] and ships per year to each mill.
FINE_BRANDS = {
    #  brand      Fe     Mill A  Mill B  Mill C  Mill D
    "Brand 1": (62.0, [10,      8,      0,      4]),
    "Brand 2": (61.5, [8,       6,      6,      0]),
    "Brand 3": (61.0, [6,       0,      4,      4]),
    "Brand 4": (60.2, [6,       6,      4,      0]),
    "Brand 5": (58.5, [4,       6,      0,      4]),
    "Brand 6": (58.0, [4,       4,      4,      0]),
    "Brand 7": (57.4, [0,       4,      4,      4]),
    "Brand 8": (56.8, [4,       2,      2,      2]),
}
MAX_GROUP_SIZE = 4           # brands that can share one yard pile / blend
MAX_FE_SPREAD = 1.5          # max Fe-grade difference inside a group [%-points]
# Grouping currently used by the managers (the "before" case).
ORIGINAL_GROUPS = [["Brand 1", "Brand 2"], ["Brand 3"], ["Brand 4"],
                   ["Brand 5", "Brand 6"], ["Brand 7"], ["Brand 8"]]

# ------------------------------------------------- multi-objective (service level)
ALPHA_MENU = [0.80, 0.85, 0.90, 0.925, 0.95, 0.965, 0.98, 0.99, 0.995]
