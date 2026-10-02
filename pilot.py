"""Pilot: does split-conformal coverage hold per group on HE-Austenite v0.1?

Target: relative_ratio (property in H2 / property in reference env).
Base model: GradientBoostingRegressor. alpha = 0.10 (nominal 90%).
Repeated random splits: train 40% / calibration 35% / test 25%.
Mondrian groups calibrate only when the group has >= 10 calibration points;
otherwise that split is skipped for that group (never quoted).
"""
import math, sys
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

CSV = "D:/research/pilots/he/Hydrogen Embrittlement in Stainless Steels/data/HE-AUSTENITE.csv"
ALPHA, NSPLIT, MIN_CAL = 0.10, 200, 10

d = pd.read_csv(CSV)
stab = {
    "300-series (304 type)": "metastable", "300-series (316/317 type)": "metastable",
    "300-series stabilised (321/347)": "metastable",
    "N-strengthened (21-6-9)": "stable", "N-strengthened (22-13-5)": "stable",
    "high-Ni austenitic (310)": "stable",
}
d = d[d.material_family.isin(stab)].reset_index(drop=True)   # drop duplex, A286 (n=4)
d["stability"] = d.material_family.map(stab)
d["fam304"] = np.where(d.material_family == "300-series (304 type)", "304-type", "other")
y = d.relative_ratio.values

X = pd.get_dummies(d[["material_family", "zone", "hydrogen_exposure", "property_name"]]).astype(float)
X["p"] = d.h2_pressure_MPa.fillna(d.h2_pressure_MPa.median())
X["p_missing"] = d.h2_pressure_MPa.isna().astype(float)
X["T"] = d.test_temperature_K
X = X.values

def model():
    return GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3,
                                     min_samples_leaf=3, random_state=0)

def qhat(scores):
    n = len(scores); k = math.ceil((n + 1) * (1 - ALPHA))
    return np.inf if k > n else np.sort(scores)[k - 1]

def run(grouping, rng_seed=0):
    g = d[grouping].values
    rows = []
    for s in range(NSPLIT):
        r = np.random.default_rng(s + rng_seed).permutation(len(d))
        ntr, nca = int(0.40 * len(d)), int(0.35 * len(d))
        tr, ca, te = r[:ntr], r[ntr:ntr + nca], r[ntr + nca:]
        m = model().fit(X[tr], y[tr])
        sc = np.abs(y[ca] - m.predict(X[ca])); res = np.abs(y[te] - m.predict(X[te]))
        q = qhat(sc)
        rows.append(dict(split=s, group="ALL", method="SCP", n_cal=len(ca), n_test=len(te),
                         cov=np.mean(res <= q), width=2 * q))
        for grp in np.unique(g):
            tm = g[te] == grp; cm = g[ca] == grp
            if tm.sum() == 0: continue
            rows.append(dict(split=s, group=grp, method="SCP", n_cal=cm.sum(), n_test=tm.sum(),
                             cov=np.mean(res[tm] <= q), width=2 * q))
            if cm.sum() >= MIN_CAL:
                qg = qhat(sc[cm])
                rows.append(dict(split=s, group=grp, method="Mondrian", n_cal=cm.sum(), n_test=tm.sum(),
                                 cov=np.mean(res[tm] <= qg), width=2 * qg))
    return pd.DataFrame(rows)

def summarise(df, n_total):
    out = []
    for (grp, meth), t in df.groupby(["group", "method"]):
        pooled = np.average(t["cov"], weights=t.n_test)
        out.append(dict(group=grp, method=meth, n_in_data=n_total.get(grp, len(d)),
                        splits_used=len(t), mean_n_cal=round(t.n_cal.mean(), 1),
                        mean_n_test=round(t.n_test.mean(), 1),
                        coverage_pooled=round(pooled, 3), coverage_mean=round(t["cov"].mean(), 3),
                        coverage_sd=round(t["cov"].std(), 3),
                        p10=round(t["cov"].quantile(.10), 2), p90=round(t["cov"].quantile(.90), 2),
                        mean_width=round(t.width.replace(np.inf, np.nan).mean(), 3)))
    return pd.DataFrame(out)

pd.set_option("display.width", 250)
print(f"n used = {len(d)} (dropped duplex + A286); alpha = {ALPHA}; splits = {NSPLIT}\n")
for grouping in ["stability", "zone", "fam304"]:
    res = run(grouping)
    n_total = d[grouping].value_counts().to_dict()
    print(f"=== grouping: {grouping} ===")
    print(summarise(res, n_total).to_string(index=False), "\n")
    res.to_csv(f"raw_{grouping}.csv", index=False)

# Leave-one-source-out: fit + calibrate on other sources, test on held-out source.
print("=== leave-one-source-out (SCP, 50 random train/cal splits of the other sources) ===")
src = d.source_short.values
for held in [s for s, c in d.source_short.value_counts().items() if c >= 9]:
    te = np.where(src == held)[0]; pool = np.where(src != held)[0]; covs = []; widths = []
    for s in range(50):
        r = np.random.default_rng(1000 + s).permutation(pool); ntr = int(0.55 * len(pool))
        tr, ca = r[:ntr], r[ntr:]
        m = model().fit(X[tr], y[tr]); q = qhat(np.abs(y[ca] - m.predict(X[ca])))
        covs.append(np.mean(np.abs(y[te] - m.predict(X[te])) <= q)); widths.append(2 * q)
    print(f"{held:16s} n_test={len(te):3d}  coverage mean={np.mean(covs):.3f} sd={np.std(covs):.3f}  width={np.mean(widths):.3f}")

