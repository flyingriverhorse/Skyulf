# Online Retail (UCI)

- **Original dataset**: https://archive.ics.uci.edu/dataset/352/online+retail
  (a classic, widely-used real transactional dataset for RFM-based customer
  segmentation).
- **Source used**: fetched directly from UCI's own hosting
  (`archive.ics.uci.edu/static/public/352/online+retail.zip`), which unpacks
  to `Online Retail.xlsx`. Verified authentic: full file is 541,909
  transaction rows, 4,372 unique non-null `CustomerID`s, date range
  2010-12-01 to 2011-12-09, 10,624 negative-`Quantity` rows (real
  returns/cancellations) — all matching well-known public statistics for
  this dataset.
- **Bundled file**: `online_retail_sample.csv` — a **stratified-by-customer
  subsample**, NOT the full file.
  - Method: rows with a null `CustomerID` (guest checkouts, ~135k rows) are
    dropped first, since per-customer segmentation requires a known
    customer. Then 1,800 of the 4,372 real customers are randomly sampled
    (`seed=42`, `numpy.random.default_rng`), and **all** of their real
    transactions are kept (not a random row sample) — this preserves each
    sampled customer's full, authentic purchase history, which matters for
    RFM (Recency/Frequency/Monetary) feature computation.
  - Result: 153,150 real transaction rows across 1,800 real customers
    (~14MB vs. ~45MB/542k rows for the full file).
  - Reason: the full file was judged too large to bundle raw in this repo;
    this subsample keeps full per-customer transaction integrity, unlike a
    plain random row sample which would truncate customers' histories and
    bias RFM features.
- **This is a real-data subsample, not synthetic data** — every row is an
  authentic row from the original UCI dataset.
