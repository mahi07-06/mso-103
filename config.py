"""
config.py  -  every assumption of the prototype lives here.

The real Nippon Steel shipping records are confidential, so the prototype
works on SYNTHETIC data generated from the numbers below. Change anything
here and re-run main.py: all results, charts and the Excel workbook update.
"""

SEED = 42                    # fixes the random numbers, so every run gives the same result
P = 365                      # length of the seasonal cycle [days]
YEARS_HISTORY = 10           # paper: 10 years of data to find the seasonality (AAM)
YEARS_RECENT = 2             # paper: 2 years of data to find each mill's share (AEM)
YEARS_TEST = 30              # fresh simulated years used only to test the stock levels

SERVICE_LEVEL = 0.95         # alpha: chance of NO stock-out in one replenishment cycle (paper: 5 % stock-out)

LOT_KT = 170.0               # one ship = 170 thousand tonnes (fixed lot size, as in the paper)

# "Manager policy" = the old way the paper compares against.
# ASSUMPTION (the paper does not publish the real rule): one flat stock level
# for the whole year, big enough for the worst season, plus a 5 % judgement margin.
MANAGER_MARGIN = 0.05

MILLS = ["Mill A", "Mill B", "Mill C", "Mill D"]

# Brand groups.
#   ships_per_year : cargoes of this group arriving in Japan this year
#   dip_day        : day of the year when arrivals are lowest
#                    Australia: cyclones in Feb-Mar.  Brazil: rainy season Dec-May + 40-day voyage.
#   amp1, amp2     : how strong the seasonal dip is (hidden from the model; it must find it)
#   shares         : share of the group's ships that each mill receives
BRAND_GROUPS = {
    "Fine ore (Australia)": dict(
        ships_per_year=120, amp1=0.10, amp2=0.00, dip_day=40,
        shares={"Mill A": 0.35, "Mill B": 0.30, "Mill C": 0.20, "Mill D": 0.15}),
    "Lump ore (Australia)": dict(
        ships_per_year=60, amp1=0.10, amp2=0.00, dip_day=35,
        shares={"Mill A": 0.40, "Mill B": 0.35, "Mill C": 0.25}),
    "Pellet (Brazil)": dict(
        ships_per_year=40, amp1=0.14, amp2=0.03, dip_day=70,
        shares={"Mill B": 0.50, "Mill C": 0.30, "Mill D": 0.20}),
}

# Chance that the operator breaks the regular "every N-th ship" rotation
# (human intervention in ship allocation, as described in the paper).
ALLOCATION_NOISE = 0.05

# Calendar used for monthly averages and the seasons the paper reports.
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
WINTER = ["Dec", "Jan", "Feb"]
SUMMER = ["Jun", "Jul", "Aug"]

# ---------------------------------------------------- brand grouping (paper Tables 4-7)
# Fine-ore brands and how many ships per year each mill receives.
FINE_BRANDS = {
    #  brand      Mill A  Mill B  Mill C  Mill D
    "Brand 1": [10,      8,      0,      4],
    "Brand 2": [8,       6,      6,      0],
    "Brand 3": [6,       0,      4,      4],
    "Brand 4": [6,       6,      4,      0],
    "Brand 5": [4,       6,      0,      4],
    "Brand 6": [4,       4,      4,      0],
    "Brand 7": [0,       4,      4,      4],
    "Brand 8": [4,       2,      2,      2],
}
# How the managers group the brands today (paper Table 4).
ORIGINAL_GROUPS = [["Brand 1", "Brand 2"], ["Brand 3"], ["Brand 4"],
                   ["Brand 5", "Brand 6"], ["Brand 7"], ["Brand 8"]]
# Revised grouping (paper Table 5 merges small fine-ore brands into an existing group).
# Here the small Brand 8 joins the Brand 5 + 6 stockpile; everything else stays as it was.
REVISED_GROUPS = [["Brand 1", "Brand 2"], ["Brand 3"], ["Brand 4"],
                  ["Brand 5", "Brand 6", "Brand 8"], ["Brand 7"]]
