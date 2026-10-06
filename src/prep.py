"""
Cleaning + weekly demand build.

Things I ran into with this dataset:
- Invoices starting with "C" are cancellations. A couple are huge (a single
  80,995 unit order of PAPER CRAFT, LITTLE BIRDIE was cancelled the same day).
  If you only filter out the C rows, those giant orders stay in and wreck
  the forecast for that SKU. So I net cancellations back against demand.
- Negative quantities with a 0 price are warehouse adjustments (damaged,
  lost, "found" etc), not customer demand. Dropped.
- Non product codes like POST, DOT, M, BANK CHARGES, gift vouchers. Dropped.
- First and last weeks are partial (data starts on a Wednesday and ends on a
  Friday), so they get trimmed.
"""
import pandas as pd

NON_PRODUCT = {"POST", "DOT", "M", "C2", "BANK CHARGES", "PADS", "AMAZONFEE",
               "CRUK", "D", "S", "B"}


def load_raw(path="data/online_retail.csv"):
    df = pd.read_csv(path, encoding="latin-1",
                     dtype={"InvoiceNo": str, "StockCode": str})
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="%m/%d/%Y %H:%M")
    df["StockCode"] = df["StockCode"].str.upper().str.strip()
    return df


def clean(df):
    df = df.copy()
    df["is_cancel"] = df["InvoiceNo"].str.startswith("C")

    # product codes only (5 digits, sometimes with a letter suffix)
    df = df[df["StockCode"].str.match(r"^\d{5}[A-Z]{0,2}$")]
    df = df[~df["StockCode"].isin(NON_PRODUCT)]

    # warehouse adjustments: not a cancellation, but negative qty or zero price
    adj = (~df["is_cancel"]) & ((df["Quantity"] <= 0) | (df["UnitPrice"] <= 0))
    df = df[~adj]

    df["revenue"] = df["Quantity"] * df["UnitPrice"]
    return df


def weekly_demand(df):
    """Net units per SKU per week (weeks end Sunday). Cancels subtract."""
    df = df.copy()
    df["week"] = df["InvoiceDate"].dt.to_period("W-SUN").dt.start_time
    wk = (df.groupby(["StockCode", "week"])["Quantity"].sum()
            .clip(lower=0)              # a cancel in a later week can't make demand negative
            .unstack(fill_value=0))

    # trim partial first and last weeks
    wk = wk.iloc[:, 1:-1]
    # make sure the calendar is continuous (store was closed over xmas 2010)
    full = pd.date_range(wk.columns.min(), wk.columns.max(), freq="7D")
    wk = wk.reindex(columns=full, fill_value=0)
    return wk


def pick_skus(df, wk, n=50, min_weeks_sold=40):
    """
    Top n SKUs by number of orders, as long as they sold in most weeks.
    I care about the steady movers here. Those are the ones a buyer actually
    sets reorder points for. One off items and lumpy stuff need a
    different approach (see README).
    """
    sales = df[~df["is_cancel"]]
    orders = sales.groupby("StockCode")["InvoiceNo"].nunique()
    weeks_sold = (wk > 0).sum(axis=1)
    ok = weeks_sold[weeks_sold >= min_weeks_sold].index
    top = orders.loc[orders.index.intersection(ok)].sort_values(ascending=False).head(n)
    return list(top.index)


def descriptions(df):
    return (df[~df["is_cancel"]].groupby("StockCode")["Description"]
              .agg(lambda s: s.mode().iat[0] if len(s.mode()) else s.iat[0])
              .str.strip().str.title())


def unit_price(df):
    s = df[~df["is_cancel"]]
    return s.groupby("StockCode")["UnitPrice"].median()
