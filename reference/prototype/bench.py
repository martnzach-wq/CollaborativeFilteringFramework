# Benchmark driver for the regsim feasibility prototype. Expects Z, M, levels, parent, aggS, inds, native, geos in namespace.
import numpy as np, pandas as pd, time
import proto_regsim as pr

HP = dict(rank=12, lam_u=0.03, iters=2000)

def run_methods(Mtr, Mte, tag):
    out = []
    preds = {}
    preds["level mean"] = pr.pred_level_mean(Z, Mtr, levels)
    preds["parent carry-down"] = pr.pred_carry_down(Z, Mtr, levels, parent)
    preds["kNN Gower (D3.6-type)"] = pr.pred_knn_gower(Z, Mtr, levels, Mte)
    preds["SoftImpute"] = pr.pred_softimpute(Z, Mtr, lam=5.0, rank_max=12)
    Zb, Mb = pr.broadcast_fill(Z, Mtr, parent)
    preds["MF, broadcast (LEVITATE-type)"] = pr.fit_hmf(Zb, Mb, parent, None, hier=False, **HP)["Zhat"]
    preds["MF, hierarchical prior"] = pr.fit_hmf(Z, Mtr, parent, None, hier=True, **HP)["Zhat"]
    preds["MF, prior + aggregation"] = pr.fit_hmf(Z, Mtr, parent, aggS, hier=True, lam_agg=1.0, **HP)["Zhat"]
    for m, P in preds.items():
        e = P[Mte] - Z[Mte]
        out.append(dict(tag, method=m, n_cells=int(Mte.sum()), rmse=float(np.sqrt(np.mean(e ** 2))),
                        mae=float(np.mean(np.abs(e)))))
    return out

rows = []
obs_idx = np.argwhere(M)
t0 = time.time()
# M1 random cells
for seed in range(3):
    r = np.random.default_rng(seed)
    te = obs_idx[r.choice(len(obs_idx), int(.1 * len(obs_idx)), replace=False)]
    Mte = np.zeros_like(M); Mte[te[:, 0], te[:, 1]] = True
    rows += run_methods(M & ~Mte, Mte, dict(scheme="M1 random cells", seed=seed, k=np.nan))
    print("M1", seed, round(time.time() - t0), flush=True)
# M2 cold start of NUTS-2 regions
n2 = np.where((levels == "N2") & (M.sum(1) >= 15))[0]
for seed in range(2):
    r = np.random.default_rng(100 + seed)
    pick = r.choice(n2, int(.15 * len(n2)), replace=False)
    for k in [1, 3, 6, 10]:
        Mte = np.zeros_like(M)
        for g in pick:
            o = np.where(M[g])[0]
            keep = r.choice(o, k, replace=False)
            Mte[g, np.setdiff1d(o, keep)] = True
        rows += run_methods(M & ~Mte, Mte, dict(scheme="M2 cold start (NUTS-2)", seed=seed, k=k))
        print("M2", seed, k, round(time.time() - t0), flush=True)
# M3 country blocks of NUTS-2 indicators
ctry = np.array([g[:2] for g in geos])
n2_ind = [j for j, k in enumerate(inds) if native[k] == "N2"]
for seed in range(2):
    r = np.random.default_rng(200 + seed)
    Mte = np.zeros_like(M)
    cands = [(c, j) for c in np.unique(ctry) for j in n2_ind if (M[(levels == "N2") & (ctry == c), j]).sum() >= 4]
    for i in r.choice(len(cands), 10, replace=False):
        c, j = cands[i]
        Mte[(levels == "N2") & (ctry == c) & M[:, j], j] = True
    rows += run_methods(M & ~Mte, Mte, dict(scheme="M3 country blocks", seed=seed, k=np.nan))
    print("M3", seed, round(time.time() - t0), flush=True)
# M4 level transfer: all NUTS-3 values of one NUTS-3-native indicator hidden
for j, kname in enumerate(inds):
    if native[kname] != "N3":
        continue
    Mte = np.zeros_like(M); Mte[(levels == "N3") & M[:, j], j] = True
    rows += run_methods(M & ~Mte, Mte, dict(scheme="M4 level transfer (NUTS-2 to NUTS-3)", seed=0, k=np.nan, indicator=kname))
    print("M4", kname, round(time.time() - t0), flush=True)
bench = pd.DataFrame(rows)
bench.to_csv("prototype_benchmark.csv", index=False)
print("done", round(time.time() - t0))
