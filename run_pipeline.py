"""
Runs everything end to end and writes results to outputs/.

    python get_data.py
    python run_pipeline.py

Takes about a minute (the rolling backtest refits a few thousand small models).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import prep, forecast as fc, inventory as inv

A = inv.ASSUMPTIONS
L = A["lead_time_weeks"]
SL = A["service_level"]
TEST_WEEKS = 12      # last 12 full weeks (mid Sep to early Dec 2011), the Q4 ramp
VAL_WEEKS = 12       # 12 weeks before that, used to pick a model + measure error
FORWARD_MODEL = "ses"

os.makedirs("outputs/charts", exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})


def main():
    raw = prep.load_raw()
    df = prep.clean(raw)
    wk_all = prep.weekly_demand(df)
    skus = prep.pick_skus(df, wk_all, n=50)
    names = prep.descriptions(df)
    price = prep.unit_price(df)
    wk = wk_all.loc[skus]
    wk.T.rename_axis("week_start").to_csv("data/weekly_demand_top50.csv")
    weeks = wk.columns
    print(f"{len(weeks)} full weeks, {weeks[0].date()} to {weeks[-1].date()}, {len(skus)} SKUs")

    # ---------- ABC / XYZ on the whole catalog ----------
    seg = inv.abc_xyz(df, wk_all)
    seg.to_csv("outputs/abc_xyz_all_skus.csv")
    seg_counts = pd.crosstab(seg["abc"], seg["xyz"])
    print(seg_counts)

    n = len(weeks)
    val_end = n - TEST_WEEKS          # validation = [val_end-VAL, val_end)
    rows_acc, rows_rop, rows_sim, bt_store = [], [], [], {}

    for i, sku in enumerate(skus):
        y = wk.loc[sku]

        # 1) pick model on validation window (no peeking at the test weeks)
        bt_val = fc.backtest(y.iloc[:val_end], holdout=VAL_WEEKS)
        sc_val = fc.score(bt_val)
        best = min(sc_val, key=lambda m: sc_val[m]["wape"])
        sigma = sc_val[best]["rmse"]

        # 2) run every method through the test window too, for the accuracy table
        bt_test = fc.backtest(y, holdout=TEST_WEEKS)
        sc_test = fc.score(bt_test)
        bt_store[sku] = (bt_test, best)
        for m in fc.METHODS:
            rows_acc.append({"sku": sku, "method": m, "picked": m == best,
                             **{f"test_{k}": v for k, v in sc_test[m].items()}})

        # 3) simulate two policies over the test weeks on actual demand
        train_hist = y.iloc[:val_end]
        avg_recent = train_hist.iloc[-13:].mean()       # trailing quarter average
        demand = bt_test["actual"].values
        start = A["start_cover_weeks"] * avg_recent

        # flat rule: reorder at 2 weeks of the trailing average, no safety stock
        flat_rop = np.full(TEST_WEEKS, avg_recent * L)
        flat_q = np.full(TEST_WEEKS, avg_recent * A["order_cover_weeks"])
        r_flat = inv.simulate(demand, flat_rop, flat_q, start, L)

        # forecast policy: ROP moves with the weekly forecast + safety stock
        f = bt_test[best].values
        fc_rop, _ = inv.reorder_point(f, sigma, L, SL)
        fc_q = np.maximum(f * A["order_cover_weeks"], 1)
        r_fc = inv.simulate(demand, fc_rop, fc_q, start, L)

        for pol, r in [("flat_rule", r_flat), ("forecast_ss", r_fc)]:
            r["avg_inventory_value"] = r["avg_on_hand"] * price.get(sku, 0)
            rows_sim.append({"sku": sku, "policy": pol, **r})

        # 4) reorder point for the next cycle, using all history.
        # Picking a model per SKU didn't beat plain SES on the test weeks
        # (see summary.txt), so going forward everything runs on SES.
        next_fc = fc.one_step(y.values, FORWARD_MODEL)
        sigma_now = sc_test[FORWARD_MODEL]["rmse"]
        rop_now, ss_now = inv.reorder_point(next_fc, sigma_now, L, SL)
        # stockout risk if they kept the flat rule: P(lead time demand > flat ROP)
        flat_now = y.iloc[-13:].mean() * L
        mu_L, sd_L = next_fc * L, sigma_now * np.sqrt(L)
        risk = 1 - inv.norm.cdf(flat_now, mu_L, sd_L) if sd_L > 0 else float(mu_L > flat_now)

        rows_rop.append({
            "sku": sku, "description": names.get(sku, ""),
            "abc": seg.loc[sku, "abc"], "xyz": seg.loc[sku, "xyz"],
            "unit_price": round(price.get(sku, np.nan), 2),
            "model": FORWARD_MODEL,
            "weekly_forecast": round(next_fc, 1),
            "forecast_rmse": round(sigma_now, 1),
            "safety_stock": int(np.ceil(ss_now)),
            "reorder_point": int(np.ceil(rop_now)),
            "flat_rule_rop": int(np.ceil(flat_now)),
            "stockout_risk_flat_rule": round(risk, 3),
            "order_qty_4wk": int(np.ceil(next_fc * A["order_cover_weeks"])),
        })
        print(f"  {i+1:>2}/{len(skus)} {sku:<7} best={best}")

    acc = pd.DataFrame(rows_acc)
    acc.to_csv("outputs/forecast_accuracy.csv", index=False)
    sim = pd.DataFrame(rows_sim)
    sim.to_csv("outputs/policy_simulation.csv", index=False)
    rop = (pd.DataFrame(rows_rop)
             .sort_values("stockout_risk_flat_rule", ascending=False))
    rop.to_csv("outputs/reorder_points.csv", index=False)

    summarize(acc, sim, rop, wk, weeks, bt_store, names, seg_counts)


def summarize(acc, sim, rop, wk, weeks, bt_store, names, seg_counts):
    lines = []

    # accuracy: the per SKU picked model vs each single method used everywhere
    picked = acc[acc["picked"]]
    lines.append("Test window WAPE (lower is better), averaged over 50 SKUs")
    for m in fc.METHODS:
        lines.append(f"  {m:<6} {acc[acc.method == m]['test_wape'].mean():.3f}")
    lines.append(f"  picked {picked['test_wape'].mean():.3f}   (model chosen per SKU on validation weeks)")
    lines.append(f"  picked bias {picked['test_bias'].mean():+.3f}")
    lines.append("")

    g = sim.groupby("policy").agg(
        fill_rate=("fill_rate", "mean"),
        units_short=("units_short", "sum"),
        stockout_weeks=("stockout_weeks", "sum"),
        skus_with_stockout=("stockout_weeks", lambda s: int((s > 0).sum())),
        avg_on_hand=("avg_on_hand", "sum"),
        avg_inventory_value=("avg_inventory_value", "sum"))
    lines.append("Policy simulation over the 12 test weeks")
    lines.append(g.round(3).to_string())
    lines.append("")
    lines.append("Model picked per SKU:")
    lines.append(picked["method"].value_counts().to_string())
    lines.append("")
    lines.append("ABC x XYZ (whole catalog):")
    lines.append(seg_counts.to_string())
    lines.append("")
    lines.append("Top 10 stockout risk if the flat rule stays:")
    lines.append(rop.head(10)[["sku", "description", "abc", "weekly_forecast",
                               "flat_rule_rop", "reorder_point",
                               "stockout_risk_flat_rule"]].to_string(index=False))
    txt = "\n".join(lines)
    open("outputs/summary.txt", "w").write(txt + "\n")
    print(txt)

    charts(sim, rop, wk, weeks, bt_store, names, g)


def charts(sim, rop, wk, weeks, bt_store, names, g):
    navy, teal, brass, rust, grey = "#14192b", "#3c6e68", "#c08a3e", "#a84a3a", "#9aa0b4"

    # 1. total weekly demand, top 50
    fig, ax = plt.subplots(figsize=(9, 3.6))
    tot = wk.sum()
    ax.plot(tot.index, tot.values, color=navy, lw=1.8)
    ax.axvspan(weeks[-TEST_WEEKS], weeks[-1], color=brass, alpha=0.12)
    ax.text(weeks[-TEST_WEEKS], tot.max() * 0.97, "  test window", color=brass, fontsize=9, va="top")
    ax.set_title("Weekly units sold, top 50 SKUs combined", loc="left")
    ax.set_ylabel("units")
    fig.tight_layout(); fig.savefig("outputs/charts/01_weekly_demand.png"); plt.close(fig)

    # 2. forecast vs actual for the highest volume SKU
    sku = wk.sum(axis=1).idxmax()
    bt, best = bt_store[sku]
    hist = wk.loc[sku]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(hist.index, hist.values, color=grey, lw=1.2, label="actual")
    ax.plot(bt["week"], bt[best], color=teal, lw=2, marker="o", ms=3,
            label=f"forecast ({best}, refit weekly)")
    ax.plot(bt["week"], bt["ma4"] if best != "ma4" else bt["naive"], color=rust, lw=1.2,
            ls="--", label="ma4" if best != "ma4" else "naive")
    ax.set_title(f"{sku} {names[sku]}: forecast vs actual", loc="left")
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout(); fig.savefig("outputs/charts/02_forecast_vs_actual.png"); plt.close(fig)

    # 3. policy comparison
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))
    labels = ["Flat rule", "Forecast + SS"]
    order = ["flat_rule", "forecast_ss"]
    vals = [g.loc[order, "fill_rate"] * 100, g.loc[order, "skus_with_stockout"],
            g.loc[order, "avg_inventory_value"] / 1000]
    titles = ["Avg fill rate (%)", "SKUs that stocked out", "Avg stock value (GBP k)"]
    for ax, v, t in zip(axes, vals, titles):
        bars = ax.bar(labels, v.values, color=[grey, teal], width=0.6)
        ax.set_title(t, loc="left", fontsize=10)
        for b, x in zip(bars, v.values):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{x:,.1f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    fig.suptitle("12 week simulation on real Q4 2011 demand", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(); fig.savefig("outputs/charts/03_policy_comparison.png"); plt.close(fig)

    # 4. stockout risk, top 15
    top = rop.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    lab = (top["sku"] + "  " + top["description"].str.slice(0, 32))
    colors = [rust if r >= 0.5 else brass if r >= 0.25 else teal for r in top["stockout_risk_flat_rule"]]
    ax.barh(lab, top["stockout_risk_flat_rule"] * 100, color=colors)
    ax.set_xlabel("chance of running out in one replenishment cycle (%)")
    ax.set_title("Stockout risk if the flat 2 week rule stays in place", loc="left")
    fig.tight_layout(); fig.savefig("outputs/charts/04_stockout_risk.png"); plt.close(fig)


if __name__ == "__main__":
    main()
