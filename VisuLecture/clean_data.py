"""clean_data.py (v2) — même interface CLI que la v1.

Pipeline de nettoyage v2 : recalage robuste + segmentation en lignes par Viterbi
(progression monotone, début ligne 0, fin dernière ligne) + recalage horizontal par ligne."""
import numpy as np, pandas as pd
from scipy.signal import savgol_filter, medfilt

def _line_geometry(words):
    g = words.groupby("ligne")
    ys = (g.y.mean() + g.hauteur.mean() / 2).round().to_numpy()   # centre des lignes
    xl = g.x.min().to_numpy().astype(float); xr = (words.x + words.largeur).groupby(words.ligne).max().to_numpy().astype(float)
    return ys, xl, xr

def _register(x, y, ys, xl, xr):
    """Mise à l'échelle robuste (percentiles) du nuage de regard sur la boîte de texte."""
    out = []
    for v, lo_t, hi_t in ((x, xl.min(), xr.max()), (y, ys.min(), ys.max())):
        lo, hi = np.percentile(v, [3, 97])
        s = (hi_t - lo_t) / max(hi - lo, 1e-6)
        out.append(lo_t + (v - lo) * s)
    return out

ADV=30.0; BAD=20.0; EM=2
def _viterbi_lines(x, y, ys, xr_total, sigma, adv_cost=None, reset_bonus=None, bad_cost=None, x_only=False):
    adv_cost = ADV if adv_cost is None else adv_cost; reset_bonus = adv_cost if reset_bonus is None else reset_bonus; bad_cost = BAD if bad_cost is None else bad_cost
    T, L = len(y), len(ys)
    d = (y[:, None] - ys[None, :]) / sigma
    emit = np.log1p(d * d)                                         # coût robuste (Cauchy)
    W = 8
    mp_ = np.array([np.median(x[max(0, t - W):t]) if t > 0 else x[0] for t in range(T)])
    mn_ = np.array([np.median(x[t:t + W]) for t in range(T)])
    reset = np.clip(((mp_ - mn_) - 0.25 * xr_total) / (0.35 * xr_total), 0, 1)
    if x_only: emit = emit * 0
    return _segment_dp(emit, reset, adv_cost, reset_bonus)

MINLEN = 15
def _segment_dp(emit, reset, adv_cost, reset_bonus):
    """Segmentation exacte en L segments consécutifs (ligne 0 -> L-1), durée min MINLEN images."""
    T, L = emit.shape
    ml = min(MINLEN, max(2, T // (2 * L)))
    cum = np.vstack([np.zeros((1, L)), np.cumsum(emit, 0)])            # cum[t, l] = somme emit[:t, l]
    trans = np.maximum(adv_cost - reset_bonus * reset, 0.5)           # coût d'un début de ligne en s
    INF = 1e18
    best = np.full((L, T + 1), INF); arg = np.zeros((L, T + 1), int)
    best[0, ml:] = cum[ml:, 0]
    for l in range(1, L):
        base = np.full(T + 1, INF)
        base[:T] = best[l - 1, :T] + trans - cum[:T, l]               # best[l-1][s] + trans[s] - cum[s, l]
        run_min = np.minimum.accumulate(base); run_arg = np.zeros(T + 1, int)
        # argmin courant
        idx = np.arange(T + 1); is_new = base == run_min
        run_arg = np.maximum.accumulate(np.where(is_new, idx, 0))
        for t in range(ml * (l + 1), T + 1):
            sidx = t - ml
            best[l, t] = run_min[sidx] + cum[t, l]; arg[l, t] = run_arg[sidx]
    path = np.zeros(T, int); t = T
    for l in range(L - 1, -1, -1):
        s = arg[l, t] if l > 0 else 0
        path[s:t] = l; t = s
    return path

def process(points, words, sg=True):
    """points: DataFrame timestamp,x,y. Retourne DataFrame timestamp,x,y,line,jump,saccade."""
    ys, xl, xr = _line_geometry(words)
    t = points.timestamp.to_numpy(); x = points.x.to_numpy(float); y = points.y.to_numpy(float)
    x = medfilt(x, 5); y = medfilt(y, 5)
    xg, yg = _register(x, y, ys, xl, xr)
    spacing = np.median(np.diff(ys)) if len(ys) > 1 else 100
    sigma = 0.9 * spacing
    path = _viterbi_lines(xg, yg, ys, xr.max() - xl.min(), sigma=sigma, x_only=True)   # init : retours à la ligne en x seulement
    tgt = ys[path]; a, b = np.polyfit(yg, tgt, 1) if False else (1.0, 0.0)
    meds = np.array([np.median(yg[path == l]) if (path == l).any() else np.nan for l in range(len(ys))])
    ok = ~np.isnan(meds)
    if ok.sum() >= 3:
        a, b = np.polyfit(meds[ok], ys[ok], 1); a = float(np.clip(a, 0.7, 1.4)); b = float(np.mean(ys[ok] - a * meds[ok])); yg = a * yg + b
    for it in range(EM):                                             # EM : lignes <-> recalage vertical affine robuste
        path = _viterbi_lines(xg, yg, ys, xr.max() - xl.min(), sigma=sigma)
        tgt = ys[path]
        w = np.abs(yg - tgt) < 1.5 * spacing
        if w.sum() < 20: break
        a, b = np.polyfit(yg[w], tgt[w], 1)
        a = float(np.clip(a, 0.7, 1.4)); b = float(np.mean(tgt[w] - a * yg[w]))
        yg = a * yg + b
    xo = np.clip(xg, xl.min(), xr.max())
    # recalage horizontal par ligne (si la ligne est suffisamment balayée)
    for l in range(len(ys)):
        idx = np.where(path == l)[0]
        if len(idx) < 15: continue
        lo, hi = np.percentile(xg[idx], [4, 96])
        if hi - lo > 0.5 * (xr[l] - xl[l]):
            xo[idx] = xl[l] + (xg[idx] - lo) * (xr[l] - xl[l]) / (hi - lo)
    xo = np.clip(xo, xl.min(), xr.max())
    jump = np.zeros(len(t), bool)
    for b in np.where(np.diff(path) > 0)[0] + 1:
        a = max(0, b - 5) + int(np.argmax(xo[max(0, b - 5):b])); c = b + int(np.argmin(xo[b:b + 5]))
        jump[a + 1:c] = True; jump[b] = True
    xs = xo.copy()
    if sg:                                                          # lissage unique, par segment de ligne
        for l in range(len(ys)):
            idx = np.where((path == l) & ~jump)[0]
            if len(idx) >= 9: xs[idx] = savgol_filter(xo[idx], 9, 2)
    step = np.abs(np.diff(xo, prepend=xo[0]))
    sacc = ((step > 0.12 * (xr.max() - xl.min())) & ~jump).astype(int)   # grands déplacements hors retour à la ligne
    return pd.DataFrame(dict(timestamp=t, x=np.rint(xo).astype(int), xs=np.rint(xs).astype(int),
                             y=np.rint(ys[path]).astype(int), yr=np.rint(ys[path] + np.clip(yg - ys[path], -0.5 * spacing, 0.5 * spacing)).astype(int),
                             line=path + 1, jump=jump.astype(int), saccade=sacc))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--points"); ap.add_argument("--words")
    ap.add_argument("--cleaned", default="points_cleaned.csv"); ap.add_argument("--normalized", default="points_normalized.csv")
    ap.add_argument("--smoothedBoth", default="points_smoothBoth.csv"); ap.add_argument("--smoothedY", default="points_smoothY.csv")
    a = ap.parse_args()
    pts = pd.read_csv(a.points); words = pd.read_csv(a.words, encoding="utf-8-sig")
    nl = words["ligne"].nunique()
    if len(pts) < 2 * nl:
        raise SystemExit(f"Trop peu de points ({len(pts)}) pour {nl} lignes de texte")
    o = process(pts, words)
    cols = ["timestamp", "x", "y", "jump", "saccade"]
    o.assign(y=o["yr"])[cols].to_csv(a.cleaned, index=False)                            # recalé, y continu
    o[o.jump == 0][cols].to_csv(a.normalized, index=False)                           # y calé sur la ligne, retours retirés
    o.assign(x=o["xs"])[cols].to_csv(a.smoothedBoth, index=False)                       # x lissé, y calé sur la ligne
    o[cols].to_csv(a.smoothedY, index=False)                                         # x recalé, y calé sur la ligne
    print(f"{len(o)} points, {nl} lignes, {int(o.jump.sum())} points de retour à la ligne")

if __name__ == "__main__":
    main()
