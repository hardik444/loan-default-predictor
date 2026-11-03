"""FICO bucketing with dynamic programming (the J.P. Morgan task idea).

Goal: split sorted FICO scores into k contiguous buckets so that the total
log-likelihood of defaults is maximised, i.e. each bucket groups borrowers with
similar default rates.

For a bucket with n borrowers and d defaults, p = d/n and
    LL = d*ln(p) + (n-d)*ln(1-p)
dp[j][i] = best LL using j buckets on the first i FICO values.
"""
import numpy as np
import pandas as pd


def _ll(n: int, d: int) -> float:
    if n == 0 or d == 0 or d == n:
        return 0.0
    p = d / n
    return d * np.log(p) + (n - d) * np.log(1 - p)


def optimal_fico_buckets(fico, default, n_buckets: int = 6, bin_width: int = 10):
    """Return a list of bucket upper edges (inclusive) found by DP."""
    df = pd.DataFrame({"f": np.floor(np.asarray(fico) / bin_width) * bin_width,
                       "d": np.asarray(default)})
    g = df.groupby("f")["d"].agg(["count", "sum"]).sort_index()
    scores = g.index.to_numpy()
    cnt = g["count"].to_numpy()
    dfl = g["sum"].to_numpy()
    m = len(scores)
    c_cnt = np.concatenate([[0], np.cumsum(cnt)])
    c_def = np.concatenate([[0], np.cumsum(dfl)])

    def cost(a, b):  # buckets covering scores[a:b]
        return _ll(c_cnt[b] - c_cnt[a], c_def[b] - c_def[a])

    k = min(n_buckets, m)
    dp = np.full((k + 1, m + 1), -np.inf)
    back = np.zeros((k + 1, m + 1), dtype=int)
    dp[0][0] = 0
    for j in range(1, k + 1):
        for i in range(j, m + 1):
            for s in range(j - 1, i):
                val = dp[j - 1][s] + cost(s, i)
                if val > dp[j][i]:
                    dp[j][i], back[j][i] = val, s
    edges, i = [], m
    for j in range(k, 0, -1):
        edges.append(scores[i - 1] + bin_width - 1)
        i = back[j][i]
    return sorted(edges)


def apply_buckets(fico, edges):
    """Map FICO scores to bucket ids: 0 = worst (lowest scores)."""
    return np.searchsorted(np.array(edges[:-1]), np.asarray(fico), side="left").clip(0, len(edges) - 1)
