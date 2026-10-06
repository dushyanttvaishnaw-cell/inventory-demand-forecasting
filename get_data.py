"""
Downloads the UCI Online Retail dataset (Dec 2010 to Dec 2011, UK based gift wholesaler).

Original source: https://archive.ics.uci.edu/dataset/352/online+retail
The UCI site is sometimes slow, so this pulls the same file from the copy
Databricks keeps in their Spark book repo. Same 541,909 rows.
"""
import os
import urllib.request

URL = ("https://raw.githubusercontent.com/databricks/Spark-The-Definitive-Guide/"
       "master/data/retail-data/all/online-retail-dataset.csv")
OUT = os.path.join("data", "online_retail.csv")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    if os.path.exists(OUT):
        print(f"{OUT} already there, skipping")
    else:
        print("downloading (about 45 MB)...")
        urllib.request.urlretrieve(URL, OUT)
        print("saved to", OUT)
