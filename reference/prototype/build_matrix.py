# Builds the prototype matrix Z (transformed, robust-scaled) from eurostat_raw/*.json (fetched by the download cell).
import json, os
import numpy as np, pandas as pd, scipy.sparse as sp
import proto_regsim as pr

W = (2018, 2022)

def parse_jsonstat(fn):
    d = json.load(open(fn, encoding="utf-8"))
    ids, sizes = d["id"], d["size"]
    cats = []
    for dim in ids:
        idx = d["dimension"][dim]["category"]["index"]
        inv = sorted(idx.items(), key=lambda kv: kv[1]) if isinstance(idx, dict) else list(enumerate(idx))
        cats.append([k for k, _ in inv])
    strides = np.cumprod([1] + sizes[::-1][:-1])[::-1]
    rows = []
    for key, v in d["value"].items():
        i = int(key); coord = []
        for s in strides:
            coord.append(i // s); i %= s
        rows.append([cats[j][c] for j, c in enumerate(coord)] + [v])
    return pd.DataFrame(rows, columns=ids + ["value"])

IND_DEF = [  # name, raw file, filters, native level, transform, eurostat dataset
    ("pop_density", "pop_density", {}, "N3", "log"),
    ("gdp_pps_hab", "gdp_pps_hab", {}, "N3", "log"),
    ("median_age", "popstruct", {"indic_de": "MEDAGEPOP"}, "N3", "id"),
    ("old_age_dependency", "popstruct", {"indic_de": "OLDDEP1"}, "N3", "id"),
    ("share_age_0_14", "popstruct", {"indic_de": "PC_Y0_14"}, "N3", "id"),
    ("pop_growth_rate", "popchange", {"indic_de": "GROWRT"}, "N3", "id"),
    ("net_migration_rate", "popchange", {"indic_de": "CNMIGRATRT"}, "N3", "id"),
    ("fertility_rate", "fertility", {}, "N3", "id"),
    ("unemployment_rate", "unemployment", {}, "N2", "log"),
    ("employment_rate", "employment", {}, "N2", "id"),
    ("tertiary_attainment", "tertiary", {}, "N2", "id"),
    ("life_expectancy", "life_exp", {}, "N2", "id"),
    ("cars_per_1000", "cars", {}, "N2", "id"),
    ("hh_income_pps", "hh_income", {}, "N2", "log"),
    ("road_deaths_pmh", "road_deaths", {}, "N2", "log1p"),
    ("rd_expenditure_gdp", "rd_exp", {}, "N2", "log1p"),
    ("poverty_exclusion_risk", "arope", {}, "N2", "id"),
    ("hospital_beds", "hosp_beds", {}, "N2", "log"),
    ("broadband_households", "broadband", {}, "N2", "id"),
    ("lifelong_learning", "lifelong_learning", {}, "N2", "log1p"),
    ("renewables_share", "renewables_share", {}, "N0", "id"),
    ("ghg_per_capita", "ghg_per_cap", {"src_crf": "TOTX4_MEMO"}, "N0", "log"),
    ("gini", "gini", {"age": "TOTAL"}, "N0", "id"),
    ("heating_degree_days", "degree_days", {"indic_nrg": "HDD"}, "N0", "id"),
    ("cooling_degree_days", "degree_days", {"indic_nrg": "CDD"}, "N0", "log1p"),
]

def lvl(code):
    return {2: "N0", 4: "N2", 5: "N3"}.get(len(code))

def valid(code):
    return lvl(code) is not None and not code.endswith("Z") and code[:2].isalpha() and code[:2].isupper()

def build(raw_dir="eurostat_raw", window=W):
    raw = {f[:-5]: parse_jsonstat(os.path.join(raw_dir, f)) for f in os.listdir(raw_dir) if f.endswith(".json")}
    pw = raw["population"]; pw = pw[pw.time.astype(int).between(*window)].groupby("geo").value.mean()
    pw = pw[[valid(g) for g in pw.index]]
    recs = []
    for name, rk, flt, native, tr in IND_DEF:
        df = raw[rk]
        for k, v in flt.items():
            df = df[df[k] == v]
        df = df[df.time.astype(int).between(*window) & df.geo.map(valid)]
        for g, v in df.groupby("geo").value.mean().items():
            recs.append((g, lvl(g), name, v))
    obs = pd.DataFrame(recs, columns=["geo", "level", "indicator", "value"])
    geos = sorted(pw.index, key=lambda g: (len(g), g)); gset = set(geos)
    geos = [g for g in geos if len(g) == 2 or (g[:2] in gset and (len(g) == 4 or g[:4] in gset))]
    obs = obs[obs.geo.isin(geos)]
    inds = [d[0] for d in IND_DEF]
    native = {d[0]: d[3] for d in IND_DEF}; trk = {d[0]: d[4] for d in IND_DEF}
    gi = {g: i for i, g in enumerate(geos)}; ki = {k: j for j, k in enumerate(inds)}
    n, K = len(geos), len(inds)
    Y = np.full((n, K), np.nan)
    for g, l, k, v in obs.itertuples(index=False):
        Y[gi[g], ki[k]] = v
    levels = np.array([lvl(g) for g in geos])
    parent = np.array([-1 if len(g) == 2 else gi[g[:2]] if len(g) == 4 else gi[g[:4]] for g in geos])
    Z = np.full_like(Y, np.nan); scal = {}
    for j, k in enumerate(inds):
        t = pr.transform(Y[:, j], trk[k]); nat = levels == native[k]
        med, s = pr.robust_scale(t[nat]); scal[k] = (med, s); Z[:, j] = (t - med) / s
    M = ~np.isnan(Z)
    pop = np.array([pw[g] for g in geos])
    rows_, cols_, vals_ = [], [], []
    for g in range(n):
        p = parent[g]
        if p >= 0:
            rows_.append(p); cols_.append(g); vals_.append(pop[g])
    A = sp.csr_matrix((vals_, (rows_, cols_)), shape=(n, n))
    rs = np.asarray(A.sum(1)).ravel(); rs[rs == 0] = 1
    aggS = sp.diags(1 / rs) @ A
    return dict(Y=Y, Z=Z, M=M, geos=geos, inds=inds, native=native, trk=trk, levels=levels, parent=parent,
                aggS=sp.csr_matrix(aggS), pop=pop, scal=scal, raw=raw, obs=obs)
