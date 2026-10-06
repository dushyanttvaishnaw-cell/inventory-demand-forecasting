"""
ABC/XYZ, safety stock, reorder points, and a simple inventory simulation.

The dataset has sales but no on hand inventory, no lead times, no PO data.
So the inventory side runs on stated assumptions (see ASSUMPTIONS below),
and the policy comparison is a simulation replaying real demand, not a
record of what this company actually had on the shelf.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

ASSUMPTIONS = {
    "lead_time_weeks": 2,      # supplier lead time, review weekly
    "service_level": 0.95,     # cycle service level target
    "order_cover_weeks": 4,    # each PO covers ~4 weeks of demand
    "start_cover_weeks": 3,    # starting stock for both policies in the sim
}


def abc_xyz(df_clean, wk):
    """ABC on revenue (80/15/5), XYZ on weekly demand CV (<0.5, <1.0, rest)."""
    sales = df_clean[~df_clean["is_cancel"]]
    rev = sales.groupby("StockCode")["revenue"].sum().sort_values(ascending=False)
    share = rev.cumsum() / rev.sum()
    abc = pd.cut(share, [0, 0.80, 0.95, 1.0001], labels=["A", "B", "C"])

    cv = (wk.std(axis=1) / wk.mean(axis=1)).replace([np.inf], np.nan)
    xyz = pd.cut(cv, [0, 0.5, 1.0, np.inf], labels=["X", "Y", "Z"])

    out = pd.DataFrame({"revenue": rev, "abc": abc})
    out = out.join(pd.DataFrame({"cv": cv, "xyz": xyz}), how="left")
    return out


def safety_stock(sigma_weekly, lead_time, service_level):
    z = norm.ppf(service_level)
    return z * sigma_weekly * np.sqrt(lead_time)


def reorder_point(fc_weekly, sigma_weekly, lead_time, service_level):
    ss = safety_stock(sigma_weekly, lead_time, service_level)
    return fc_weekly * lead_time + ss, ss


def simulate(demand, rop_by_week, qty_by_week, start_on_hand, lead_time):
    """
    Weekly periodic review, lost sales (wholesale customers just buy
    somewhere else). Order arrives `lead_time` weeks after it's placed.

    demand, rop_by_week, qty_by_week: arrays for the same weeks.
    """
    n = len(demand)
    on_hand = start_on_hand
    pipeline = [0.0] * (n + lead_time + 1)
    served = short = 0.0
    stockout_weeks = 0
    end_inv = []
    for t in range(n):
        on_hand += pipeline[t]                     # receive
        d = demand[t]
        sold = min(on_hand, d)
        served += sold
        short += d - sold
        if d > on_hand:
            stockout_weeks += 1
        on_hand -= sold
        on_order = sum(pipeline[t + 1:])
        if on_hand + on_order <= rop_by_week[t]:   # review + reorder
            pipeline[t + lead_time] += qty_by_week[t]
        end_inv.append(on_hand)
    total = served + short
    return {
        "fill_rate": served / total if total else 1.0,
        "units_short": short,
        "stockout_weeks": stockout_weeks,
        "avg_on_hand": float(np.mean(end_inv)),
    }
