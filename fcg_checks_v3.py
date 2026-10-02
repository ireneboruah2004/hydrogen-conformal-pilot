"""Crack-growth pilot v3 checks. Run: python fcg_checks_v3.py [data folder]
1. ASME exceedance with tolerance (1.0/1.2/1.5/2.0x) and max exceedance factor.
2. Fair held-out-curve comparison: ASME calibrated on ALL 7 remaining curves;
   GBR/Linear via curve-level CV+ (7 folds, one curve per fold).
   Scores weighted 1/n per curve. Two-sided 90% interval and ONE-SIDED 90% upper bound.
Reuses data loading and asme() from fcg_pilot_v2.py (same folder).
"""
import sys, numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
ROOT = sys.argv[1] if len(sys.argv) > 1 else "D:/research/pilots/fcg/dataset"
src = open("fcg_pilot_v2.py").read().split("Xall = pd.get_dummies")[0]
sys.argv = [sys.argv[0], ROOT]; exec(src)
ALPHA = 0.10

s = d[d.in_scope].copy(); s["ratio"] = 10 ** (s.y - s.y_asme)
print("=== 1. ASME exceedance by tolerance (in scope, P <= 21 MPa) ===")
for tol in [1.0, 1.2, 1.5, 2.0]:
    e = s.ratio > tol
    print(f">{tol}x curve: overall {e.mean():.3f} | by grade {e.groupby(s.grade).mean().round(3).to_dict()}"
          f" | by regime {e.groupby(s.regime, observed=True).mean().round(3).to_dict()}")
print("\nper curve: measured / code (median, max)")
print(s.groupby("curve").ratio.agg(["median", "max"]).round(2).to_string())
for g in ["X52", "X70", "X100"]:
    a = pd.read_csv(os.path.join(ROOT, g, "Air.csv"), header=None, names=["K", "da"])
    air = 3.8e-12 * (2.88 / (2.88 - R_RATIO)) ** 3.07 * a.K ** 3.07 * 1e3
    print(f"units check, {g} air data / code air curve (median): {np.median(a.da / air):.2f}")

Xall = pd.get_dummies(d.grade).astype(float); Xall["logK"] = np.log10(d.K); Xall["logP"] = np.log10(d.P)
X, y = Xall.values, d.y.values
def mk(kind): return LinearRegression() if kind == "Linear" else GradientBoostingRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=3, min_samples_leaf=5, random_state=0)
def wq(v, w, q):
    o = np.argsort(v); v, w = v[o], w[o]; c = np.cumsum(w) / w.sum()
    return v[min(np.searchsorted(c, q), len(v) - 1)]
curves = sorted(d.curve.unique()); rows = []
for held in curves:
    te = np.where(d.curve == held)[0]; rest = [c for c in curves if c != held]
    ca = np.where(d.curve.isin(rest))[0]
    w = 1.0 / d.curve.iloc[ca].map(d.curve.iloc[ca].value_counts()).values
    # ASME: no training, calibrate on all 7 (in scope only)
    if d.in_scope.values[te].all():
        m = d.in_scope.values[ca]; r = y[ca][m] - d.y_asme.values[ca][m]
        q2, qu = wq(np.abs(r), w[m], 1 - ALPHA), wq(r, w[m], 1 - ALPHA)
        rt = y[te] - d.y_asme.values[te]
        rows.append(dict(held=held, model="ASME", cover2=np.mean(np.abs(rt) <= q2), cover_up=np.mean(rt <= qu),
                         width=10 ** (2 * q2), n=len(te)))
    for kind in ["Linear", "GBR"]:                       # curve-level CV+
        lo, hi, up, W = [], [], [], []
        for k in rest:
            tr = np.where(d.curve.isin(set(rest) - {k}))[0]; kk = np.where(d.curve == k)[0]
            mdl = mk(kind).fit(X[tr], y[tr]); rk = y[kk] - mdl.predict(X[kk]); mt = mdl.predict(X[te])
            for ri in rk:
                lo.append(mt - abs(ri)); hi.append(mt + abs(ri)); up.append(mt + ri); W.append(1.0 / len(kk))
        lo, hi, up, W = np.array(lo), np.array(hi), np.array(up), np.array(W)
        L = np.array([-wq(-lo[:, j], W, 1 - ALPHA) for j in range(len(te))])
        H = np.array([wq(hi[:, j], W, 1 - ALPHA) for j in range(len(te))])
        U = np.array([wq(up[:, j], W, 1 - ALPHA) for j in range(len(te))])
        yt = y[te]
        rows.append(dict(held=held, model=kind, cover2=np.mean((yt >= L) & (yt <= H)), cover_up=np.mean(yt <= U),
                         width=np.median(10 ** (H - L)), n=len(te)))
R = pd.DataFrame(rows)
print("\n=== 2. Held-out curve: ASME calibrated on 7 curves; GBR/Linear curve-level CV+ ===")
print("cover2 = two-sided 90% interval; cover_up = ONE-SIDED 90% upper bound (nominal 0.90)")
print(R.pivot(index="held", columns="model", values="cover2").round(3).to_string())
print(R.pivot(index="held", columns="model", values="cover_up").round(3).to_string())
print("\nmean over held-out curves (ASME: 7 in-scope curves only):")
print(R.groupby("model").agg(cover2=("cover2", "mean"), cover2_sd=("cover2", "std"),
      cover_up=("cover_up", "mean"), cover_up_sd=("cover_up", "std"), width=("width", "median")).round(3).to_string())
sc = R[R.held != "X70@34MPa"]
print("\nsame, excluding X70@34MPa for all models (like-for-like with ASME):")
print(sc.groupby("model").agg(cover2=("cover2", "mean"), cover_up=("cover_up", "mean"), width=("width", "median")).round(3).to_string())
