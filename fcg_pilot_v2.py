"""Crack-growth pilot v2: conformal intervals on log10(da/dN) in gaseous H2.

Run:  python fcg_pilot_v2.py                      (uses D:/research/pilots/fcg/dataset)
      python fcg_pilot_v2.py "path/to/fcg/dataset"

Data: tRosJan/machine-learning-hydrogen-assisted-fatigue (MIT), raw per-pressure
curve files. R = 0.5, 1 Hz. dK in MPa m^0.5, da/dN in mm/cycle.

Changes from v1
  * ASME baseline = CC220 Table 2 (carbon steels) as given in San Marchi et al.,
    PVP2024-122529 (technical basis paper, NOT the code text): g(P) = 0.071 P^0.51,
    dKc, dKa switching, air branch below dKa. PH = H2 partial pressure (MPa).
  * ASME scope: PH <= 21 MPa (CC220 stops at 20.7; the 21 MPa curve is a footnote).
    X70@34MPa is excluded from every ASME number; it stays in the ML models.
  * X70@5.5MPa: rows after the dK peak (falling-dK segment) are dropped.
  * Data folder is a command-line argument; works on Windows and Linux.
Splits: point (leaky, 200 reps), leave-one-curve-out (2 calibration curves, all
combos), leave-one-grade-out. Nominal 90%. With <= 7 calibration curves no
curve-level finite-sample guarantee is possible, so curve/grade rows are empirical
checks on 8 curves, not guarantees.
"""
import glob, itertools, math, os, sys
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

ROOT = sys.argv[1] if len(sys.argv) > 1 else "D:/research/pilots/fcg/dataset"
ALPHA, R_RATIO, P_SCOPE = 0.10, 0.5, 21.0

rows = []
for f in sorted(glob.glob(os.path.join(ROOT, "X*", "*.csv"))):
    grade, name = os.path.basename(os.path.dirname(f)), os.path.basename(f)[:-4]
    if "poly" in name or name == "Air": continue
    t = pd.read_csv(f, header=None, names=["K", "dadN"])
    if grade == "X70" and name == "5_5":                       # drop falling-dK segment
        t = t.iloc[: int(t.K.values.argmax()) + 1]
    t["P"] = float(name.replace("_", ".")); t["grade"] = grade; t["curve"] = f"{grade}@{t.P[0]:g}MPa"
    rows.append(t)
d = pd.concat(rows, ignore_index=True)
d["y"] = np.log10(d.dadN)
d["regime"] = pd.cut(d.K, [0, 10, 16, 99], labels=["low dK <10", "mid dK 10-16", "high dK >16"])

def asme(dK, R, PH):
    """da/dN in m/cycle."""
    dK = np.asarray(dK, float)
    g   = 0.071 * PH ** 0.51
    dKc = (21.66 + 10 * R - 3.7 * R ** 2) * PH ** -0.18
    dKa = (8.6 - 3.0 * R + 7.9 * R ** 2 - 9.4 * R ** 3) * PH ** -0.15
    air  = 3.8e-12 * (2.88 / (2.88 - R)) ** 3.07 * dK ** 3.07
    low  = 3.5e-14 * (1 + 0.43 * R) / (1 - R) * dK ** 6.5 * g
    high = 1.5e-11 * (1 + 2.0 * R) / (1 - R) * dK ** 3.66
    br = np.where(dK < dKa, "air", np.where(dK < dKc, "low", "high"))
    return np.where(br == "air", air, np.where(br == "low", low, high)), br

c, d["branch"] = asme(d.K.values, R_RATIO, d.P.values)
d["y_asme"] = np.log10(c * 1e3)                                   # mm/cycle
d["in_scope"] = d.P <= P_SCOPE

Xall = pd.get_dummies(d.grade).astype(float)
Xall["logK"] = np.log10(d.K); Xall["logP"] = np.log10(d.P)
X, y = Xall.values, d.y.values

def fit_predict(kind, tr, idx):
    if kind == "ASME": return d.y_asme.values[idx]
    m = LinearRegression() if kind == "Linear" else GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=3, min_samples_leaf=5, random_state=0)
    return m.fit(X[tr], y[tr]).predict(X[idx])

def wquantile(v, w, q):
    o = np.argsort(v); v, w = v[o], w[o]; cw = np.cumsum(w) / w.sum()
    return v[min(np.searchsorted(cw, q), len(v) - 1)]

def conformal(kind, tr, ca, te, curve_weighted):
    if kind == "ASME":                                            # code curve only defined in scope
        ca, te = ca[d.in_scope.values[ca]], te[d.in_scope.values[te]]
        if len(ca) < 10 or len(te) == 0: return None
    pc, pt = fit_predict(kind, tr, ca), fit_predict(kind, tr, te)
    w = 1.0 / d.curve.iloc[ca].map(d.curve.iloc[ca].value_counts()).values if curve_weighted else np.ones(len(ca))
    s_abs, s_up = np.abs(y[ca] - pc), y[ca] - pc
    if curve_weighted:
        q2, qu = wquantile(s_abs, w, 1 - ALPHA), wquantile(s_up, w, 1 - ALPHA)
    else:
        k = math.ceil((len(ca) + 1) * (1 - ALPHA))
        q2, qu = np.sort(s_abs)[min(k, len(ca)) - 1], np.sort(s_up)[min(k, len(ca)) - 1]
    r = y[te] - pt
    return pd.DataFrame(dict(idx=te, cover2=np.abs(r) <= q2, cover_up=r <= qu,
                             width_factor=10 ** (2 * q2), abs_err=np.abs(r)))

out = []
def add(res, kind, scheme, rep, held):
    if res is None: return
    res["model"], res["scheme"], res["rep"], res["held"] = kind, scheme, rep, held
    out.append(res)

n = len(d); models = ["ASME", "Linear", "GBR"]
for rep in range(200):                                            # point split (leaky)
    p = np.random.default_rng(rep).permutation(n)
    tr, ca, te = p[:int(.4 * n)], p[int(.4 * n):int(.75 * n)], p[int(.75 * n):]
    for k in models: add(conformal(k, tr, ca, te, False), k, "point", rep, "-")

curves = sorted(d.curve.unique())
for held in curves:                                               # leave one curve out
    te = np.where(d.curve == held)[0]; rest = [x for x in curves if x != held]
    for rep, cal in enumerate(itertools.combinations(rest, 2)):
        ca = np.where(d.curve.isin(cal))[0]; tr = np.where(d.curve.isin(set(rest) - set(cal)))[0]
        for k in models: add(conformal(k, tr, ca, te, True), k, "curve", rep, held)

for held in sorted(d.grade.unique()):                             # leave one grade out
    te = np.where(d.grade == held)[0]; rest = sorted(d.curve[d.grade != held].unique())
    for rep, cal in enumerate(itertools.combinations(rest, 2)):
        ca = np.where(d.curve.isin(cal))[0]; tr = np.where(d.curve.isin(set(rest) - set(cal)))[0]
        for k in models: add(conformal(k, tr, ca, te, True), k, "grade", rep, held)

R = pd.concat(out, ignore_index=True)
R = R.merge(d[["curve", "grade", "regime"]], left_on="idx", right_index=True)
R.to_csv("fcg_raw_v2.csv", index=False)

def table(g):
    per = R.groupby(g + ["rep", "held"], observed=True).agg(c2=("cover2", "mean"), cu=("cover_up", "mean"),
                                                            w=("width_factor", "mean"), n=("cover2", "size"))
    return per.groupby(g, observed=True).agg(coverage=("c2", "mean"), coverage_sd=("c2", "std"),
                                             upper_bound_coverage=("cu", "mean"), width_factor=("w", "median"),
                                             mean_n_test=("n", "mean")).round(3)

pd.set_option("display.width", 250)
s = d[d.in_scope]
print(f"hydrogen points = {n}, curves = {len(curves)}: {curves}")
print(f"ASME scope: PH <= {P_SCOPE} MPa -> {len(s)} points, {s.curve.nunique()} curves (X70@34MPa excluded)\n")
print("ASME exceedance (measured da/dN above code curve), in scope:")
print("overall", round((s.y > s.y_asme).mean(), 3))
print((s.y > s.y_asme).groupby(s.branch).agg(["mean", "size"]).round(3).to_string(), "\n")
print((s.y > s.y_asme).groupby(s.regime, observed=True).agg(["mean", "size"]).round(3).to_string(), "\n")
print((s.y > s.y_asme).groupby(s.grade).agg(["mean", "size"]).round(3).to_string(), "\n")
print("=== by scheme x model ===");            print(table(["scheme", "model"]).to_string(), "\n")
print("=== by scheme x model x dK regime ==="); print(table(["scheme", "model", "regime"]).to_string(), "\n")
print("=== leave-one-curve-out, per held-out curve (ASME blank = out of scope) ===")
print(R[R.scheme == "curve"].groupby(["held", "model"]).cover2.mean().unstack().round(3).to_string(), "\n")
print("=== leave-one-grade-out, per held-out grade ===")
print(R[R.scheme == "grade"].groupby(["held", "model"]).cover2.mean().unstack().round(3).to_string())
