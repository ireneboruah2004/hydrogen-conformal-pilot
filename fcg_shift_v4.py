"""Pressure-shift test. Run: python fcg_shift_v4.py [data folder]  (needs fcg_pilot_v2.py in same folder)
Hold out the highest-pressure curve of each grade; curve-level CV+ on the rest (ASME: calibrate on all rest, in scope).
Models: GBR (unconstrained), HGB (hist GBR) and HGB-mono (monotone increasing in logP and logK; sklearn >= 1.0), Linear, Hybrid (Linear + GBR on its residuals, no logP in GBR).
One-sided 90% upper bound + two-sided 90% interval.
"""
import sys, numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
ROOT = sys.argv[1] if len(sys.argv) > 1 else "D:/research/pilots/fcg/dataset"
sys.argv = [sys.argv[0], ROOT]; exec(open("fcg_pilot_v2.py").read().split("Xall = pd.get_dummies")[0])
ALPHA = 0.10
Xall = pd.get_dummies(d.grade).astype(float); Xall["logK"] = np.log10(d.K); Xall["logP"] = np.log10(d.P)
cols = list(Xall.columns); X, y = Xall.values, d.y.values
iK, iP = cols.index("logK"), cols.index("logP")
gbr = lambda **k: GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, min_samples_leaf=5, random_state=0, **k)
class Hybrid:
    def fit(self, X, y):
        self.l = LinearRegression().fit(X, y); keep = [i for i in range(X.shape[1]) if i != iP]
        self.keep = keep; self.g = gbr().fit(X[:, keep], y - self.l.predict(X)); return self
    def predict(self, X): return self.l.predict(X) + self.g.predict(X[:, self.keep])
mono = [0] * len(cols); mono[iK] = 1; mono[iP] = 1
hgb = lambda **k: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=3, min_samples_leaf=5, random_state=0, **k)
MODELS = {"GBR": lambda: gbr(), "HGB": lambda: hgb(), "HGB-mono": lambda: hgb(monotonic_cst=mono), "Linear": LinearRegression, "Hybrid": Hybrid}
def wq(v, w, q):
    o = np.argsort(v); v, w = v[o], w[o]; c = np.cumsum(w) / w.sum(); return v[min(np.searchsorted(c, q), len(v) - 1)]
top = d.groupby("grade").P.max()
held_list = [f"{g}@{p:g}MPa" for g, p in top.items()]
held_list += [c for c in sorted(d.curve.unique()) if c not in held_list]      # others, for contrast
rows = []
for held in held_list:
    te = np.where(d.curve == held)[0]; rest = [c for c in d.curve.unique() if c != held]
    ca = np.where(d.curve.isin(rest))[0]; yt = y[te]
    if d.in_scope.values[te].all():
        m = d.in_scope.values[ca]; w = 1.0 / d.curve.iloc[ca][m].map(d.curve.iloc[ca][m].value_counts()).values
        r = y[ca][m] - d.y_asme.values[ca][m]; rt = yt - d.y_asme.values[te]
        rows.append(dict(held=held, model="ASME", up=np.mean(rt <= wq(r, w, 1 - ALPHA)), two=np.mean(np.abs(rt) <= wq(np.abs(r), w, 1 - ALPHA)),
                         bias=np.median(rt)))
    for name, mk in MODELS.items():
        up, ab, W, mts = [], [], [], []
        for k in rest:
            tr = np.where(d.curve.isin(set(rest) - {k}))[0]; kk = np.where(d.curve == k)[0]
            mdl = mk().fit(X[tr], y[tr]); rk = y[kk] - mdl.predict(X[kk]); mt = mdl.predict(X[te]); mts.append(mt)
            for ri in rk: up.append(mt + ri); ab.append(abs(ri)); W.append(1.0 / len(kk))
        up, W = np.array(up), np.array(W); ab = np.array(ab)
        U = np.array([wq(up[:, j], W, 1 - ALPHA) for j in range(len(te))])
        # two-sided CV+ with |r|
        mt_rep = np.repeat(np.array(mts), [np.sum(d.curve == k) for k in rest], axis=0)
        L = np.array([-wq(-(mt_rep[:, j] - ab), W, 1 - ALPHA) for j in range(len(te))])
        H = np.array([wq(mt_rep[:, j] + ab, W, 1 - ALPHA) for j in range(len(te))])
        rows.append(dict(held=held, model=name, up=np.mean(yt <= U), two=np.mean((yt >= L) & (yt <= H)),
                         bias=np.median(yt - np.mean(mts, axis=0))))
R = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("held-out = highest-pressure curve per grade:", held_list[:3])
for v, lab in [("up", "ONE-SIDED 90% upper bound coverage"), ("two", "two-sided 90% coverage"), ("bias", "median residual log10 (measured - predicted; >0 = underpredicts)")]:
    print(f"\n=== {lab} ==="); print(R.pivot(index="held", columns="model", values=v).loc[held_list].round(3).to_string())
R["group"] = np.where(R.held.isin(held_list[:3]), "top-pressure", "other")
print("\n=== mean one-sided coverage: top-pressure vs other held-out curves ===")
print(R.pivot_table(index="group", columns="model", values="up").round(3).to_string())
