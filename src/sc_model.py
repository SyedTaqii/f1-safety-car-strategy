"""Safety-car hazard model: when a neutralisation starts, of which kind, and how long it lasts
(`python -m src.sc_model` fits it, validates it on the hold-out seasons and writes outputs/sc_*.csv).

Two hazards with the same recipe, one for SC-kind incidents (safety car or red flag) and one for VSC-only incidents:
    P(starts on this lap) = pooled logistic(lap features) x circuit multiplier x season multiplier
Both multipliers are shrunk towards 1: (observed + strength) / (expected + strength)."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize_scalar
from scipy.special import expit, gammaln

from src.data_loader import ROOT, load_config

KEY = ["year", "round"]
KINDS = ["SC", "VSC"]    # the two hazards. A red flag counts as SC-kind: most red flags begin as a safety car
OFF = 1e12               # a prior this strong pins a multiplier to 1


def incidents(events):
    """One row per incident (an `episode` of the event table): the lap it starts on, the last lap it touches, its
    type = the most severe row (VSC < SC < RED) and its length in leader laps (blank when a red flag is involved)."""
    rank = {"VSC": 0, "SC": 1, "RED": 2}
    inc = events.assign(rank=events.type.map(rank)).groupby(KEY + ["episode"]).agg(
        lap=("start_lap", "min"), end_lap=("end_lap", "max"), rank=("rank", "max"),
        length=("laps", lambda s: s.sum(skipna=False))).reset_index()
    return inc.assign(type=inc["rank"].map({v: k for k, v in rank.items()})).drop(columns="rank")


def _features(laps, m):
    """The candidate lap features. laps needs lap, race_laps, street and since."""
    frac, (a, b) = (laps.lap - 0.5) / laps.race_laps, m["phase_edges"]
    return laps.assign(lap1=laps.lap == 1, lap2=laps.lap == 2, mid=(frac > a) & (frac <= b), late=frac > b,
                       street_later=laps.street & (laps.lap > 2), restart=laps.since <= m["restart_laps"],
                       log_laps=np.log(laps.race_laps))     # coefficient -1 = the risk is per race, 0 = per lap


def at_risk(races, events, cfg):
    """One row per lap on which an incident could start: every lap of every race except the laps an earlier incident
    still covers (the laps after its first, up to its last). An incident's own first lap is at risk, and carries
    `kind` (SC | VSC, else ""), `type`, `span` (laps it runs on after the first) and `length`.
    `since` = laps since the last incident ended (blank before the first one)."""
    inc = incidents(events)
    laps = races.loc[races.index.repeat(races.laps), KEY + ["circuit", "street", "wet", "laps"]].rename(columns={"laps": "race_laps"})
    laps["lap"] = laps.groupby(KEY).cumcount() + 1
    laps = laps.merge(inc[KEY + ["lap", "end_lap", "type", "length"]], on=KEY + ["lap"], how="left")
    assert laps.type.notna().sum() == len(inc), "an incident starts outside its race"
    race = [laps.year, laps["round"]]
    laps["since"] = laps.lap - laps.end_lap.groupby(race).ffill().groupby(race).shift()   # <= 0: the last one still runs
    assert not (laps.type.notna() & (laps.since <= 0)).any(), "an incident starts while another is running"
    laps = laps[~(laps.since <= 0)].copy()
    laps["kind"], laps["span"] = laps.type.replace("RED", "SC").fillna(""), laps.end_lap - laps.lap
    return _features(laps.drop(columns="end_lap"), cfg["sc_model"]).reset_index(drop=True)


def prior_strength(obs, exp, bounds):
    """How much a group's own record counts against the pooled model. Each group's rate multiplier gets a
    Gamma(strength, strength) prior (mean 1, sd 1/sqrt(strength)); given the multiplier its incident count is
    Poisson, so on its own the count is negative binomial. Returns the strength that makes the observed counts
    most likely, within bounds: small = the groups really differ, at the upper bound = no sign that they do."""
    obs, exp = np.asarray(obs, float), np.asarray(exp, float)
    def nll(log_s):
        s = np.exp(log_s)
        return -np.sum(gammaln(obs + s) - gammaln(s) + s * np.log(s / (s + exp)) + obs * np.log(exp / (s + exp)))
    return float(np.exp(minimize_scalar(nll, bounds=np.log(bounds), method="bounded").x))


def multiplier(record, strength):
    """A group's shrunk rate multiplier from its (observed, expected) incidents: the posterior mean."""
    return (record[0] + strength) / (record[1] + strength)


def _design(X, features):
    return np.c_[np.ones(len(X)), X[features].to_numpy(float)]


def _shrink(k, name, groups, y, expected, m, off):
    t = pd.DataFrame({"o": np.asarray(y, float), "e": np.asarray(expected)}).groupby(np.asarray(groups)).sum()
    k[name] = dict(zip(t.index, zip(t.o, t.e)))
    k[name + "_strength"] = OFF if name in off else prior_strength(t.o, t.e, m["strength_bounds"])


def base(model, X):
    """Each kind's hazard on each lap of X before any season effect: pooled model x circuit multiplier. (2, len(X))."""
    A = _design(X, model["features"])
    return np.stack([expit(A @ model[k]["res"].params) * X.circuit.map(
        lambda c: multiplier(model[k]["circuit"].get(c, (0, 0)), model[k]["circuit_strength"])).to_numpy() for k in KINDS])


def fit(laps, cfg, features=None, off=()):
    """Fit both hazards on the at-risk laps of the training races.
    off: "circuit" and / or "season" switch that multiplier off (for the comparison table)."""
    m = cfg["sc_model"]
    model = {"features": features or m["features"], "street": set(cfg["circuits"]["street"]), "cfg": m}
    A = _design(laps, model["features"])
    for kind in KINDS:
        y = (laps.kind == kind).to_numpy()
        res = sm.Logit(y, A).fit(disp=0)
        # the pool: the observed incidents of this kind, drawn from for a simulated incident's type and length
        model[kind] = {"res": res, "pool": laps.loc[y, ["type", "span", "length"]].reset_index(drop=True)}
        _shrink(model[kind], "circuit", laps.circuit, y, res.predict(A), m, off)
    for kind, expected in zip(KINDS, base(model, laps)):      # the season explains what the circuits left over
        _shrink(model[kind], "season", laps.year, laps.kind == kind, expected, m, off)
    return model


def hazard(model, circuit, n_laps, year=None, draws=None, rng=None):
    """P(an incident of each kind starts on lap l | lap l is at risk), for laps 1..n_laps: shape (2, n_laps), rows
    in KINDS order. year: the season whose record to date scales the rates (None, or a season the model hasn't
    seen: no season effect). With draws: (draws, 2, n_laps), each row one draw of everything the model is unsure
    of: the logistic's coefficients and both multipliers."""
    X = _features(pd.DataFrame({"lap": np.arange(1, n_laps + 1), "race_laps": n_laps,
                                "street": circuit in model["street"], "since": np.nan}), model["cfg"])
    A, out = _design(X, model["features"]), []
    for kind in KINDS:
        k = model[kind]
        groups = [(k[g].get(key, (0, 0)), k[g + "_strength"]) for g, key in (("circuit", circuit), ("season", year))]
        if draws is None:
            h = expit(A @ k["res"].params) * np.prod([multiplier(r, s) for r, s in groups])
        else:
            h = expit(rng.multivariate_normal(k["res"].params, k["res"].cov_params(), draws) @ A.T)
            for (o, e), s in groups:
                h = h * rng.gamma(o + s, 1 / (e + s), (draws, 1))       # the multiplier's posterior
        out.append(h)
    return np.stack(out, axis=-2)


def p_at_least_one(h):
    """P(at least one incident in the race) from per-lap hazards (..., kinds, laps). Pass one kind's row for that kind."""
    return 1 - np.prod(1 - h.sum(axis=-2), axis=-1)


def sample(model, circuit, n_laps, rng, n=1, year=None):
    """n simulated races' neutralisation schedules. One row per incident: sim (0..n-1), lap (it starts on), type
    (SC | VSC | RED), end_lap (the last lap it touches) and laps (its length in leader laps; blank for a red flag).
    Every simulated race draws its own hazards, so what the model doesn't know is in the spread of the races."""
    h = hazard(model, circuit, n_laps, year, draws=n, rng=rng)
    busy_to, out = np.zeros(n, int), []
    for lap in range(1, n_laps + 1):
        u = np.where(lap > busy_to, rng.random(n), 2.0)      # 2: an incident is still running, none can start
        edge = np.cumsum(h[:, :, lap - 1], axis=1)           # u under the first edge: SC-kind; under the second: VSC
        for i, kind in enumerate(KINDS):
            sims, pool = np.flatnonzero((u < edge[:, i]) & (u >= (edge[:, i - 1] if i else 0))), model[kind]["pool"]
            got = pool.iloc[rng.integers(len(pool), size=len(sims))]
            end = np.minimum(lap + got.span.to_numpy(), n_laps).astype(int)
            busy_to[sims] = end
            out.append(pd.DataFrame({"sim": sims, "lap": lap, "type": got.type.to_numpy(), "end_lap": end,
                                     "laps": got.length.to_numpy()}))
    return pd.concat(out).sort_values(["sim", "lap"], ignore_index=True)


def validate(laps, cfg, features=None, off=()):
    """Rolling hold-out: each test season is predicted from the seasons before it, race by race. One row per test
    race. Before the race: p_any / p_sc = the model's P(at least one incident / SC-kind incident), next to two
    baselines (hist_: the share of that circuit's earlier races that had one; pool_: the share of all earlier
    races) and what happened (any, sc). After it: the model's score on the laps as they were run (logloss, and
    e_sc / e_vsc = the incidents it expected on them), and the race joins the season's record for the next one.
    Also returns the test laps with the model's p_sc and p_vsc for each."""
    rows, lap_rows = [], []
    for year in cfg["sc_model"]["test_years"]:
        train = laps[laps.year < year]
        model = fit(train, cfg, features, off)
        had = train.groupby(KEY + ["circuit"]).kind.agg(any=lambda k: (k != "").any(), sc=lambda k: (k == "SC").any()).reset_index()
        for (_, rnd), race in laps[laps.year == year].groupby(KEY):
            circuit, n_laps = race.circuit.iloc[0], int(race.race_laps.iloc[0])
            past = had[had.circuit == circuit] if (had.circuit == circuit).any() else had     # a new circuit has no history
            h, b = hazard(model, circuit, n_laps, year), base(model, race)
            p = b * [[multiplier(model[k]["season"].get(year, (0, 0)), model[k]["season_strength"])] for k in KINDS]
            y = np.stack([(race.kind == k).to_numpy() for k in KINDS])
            rows.append({"year": year, "round": rnd, "circuit": circuit, "laps": n_laps, "at_risk": len(race),
                         "n_sc": y[0].sum(), "n_vsc": y[1].sum(), "e_sc": p[0].sum(), "e_vsc": p[1].sum(),
                         "any": y.any(), "p_any": p_at_least_one(h), "hist_any": past["any"].mean(), "pool_any": had["any"].mean(),
                         "sc": y[0].any(), "p_sc": p_at_least_one(h[:1]), "hist_sc": past.sc.mean(), "pool_sc": had.sc.mean(),
                         "logloss": -(np.log(p[y]).sum() + np.log(1 - p.sum(0)[~y.any(0)]).sum())})
            lap_rows.append(race.assign(p_sc=p[0], p_vsc=p[1]))
            for i, k in enumerate(KINDS):
                o, e = model[k]["season"].get(year, (0, 0))
                model[k]["season"][year] = (o + y[i].sum(), e + b[i].sum())
    return pd.DataFrame(rows), pd.concat(lap_rows)


def brier(v, what="any"):
    """Brier scores of the model and the two baselines for one outcome (any | sc), per test season and overall, and
    the model's lead over each baseline with its standard error (paired over races)."""
    err = pd.DataFrame({c: (v[f"{c}_{what}"] - v[what]) ** 2 for c in ("p", "hist", "pool")})
    t = pd.concat([err.groupby(v.year).mean(), err.mean().to_frame("all").T]).rename(columns={"p": "model"})
    t.insert(0, "races", [*v.groupby("year").size(), len(v)])
    for c in ("hist", "pool"):
        d = err[c] - err.p
        t.loc["all", [f"lead_{c}", f"se_{c}"]] = d.mean(), d.std() / np.sqrt(len(d))
    return t


def by_lap(model, laps, circuit, n_laps, year, rng):
    """Chart 1's table for one race. Per lap: p = the chance that an incident starts on it (p_lo / p_hi = the
    band, p_sc / p_vsc = by kind) and cum = the chance that at least one has started by then, next to the average
    dry race in `laps` (avg, avg_cum)."""
    m = model["cfg"]
    h, hd = hazard(model, circuit, n_laps, year), hazard(model, circuit, n_laps, year, m["draws"], rng).sum(axis=1)
    avg = pd.Series(base(model, laps).sum(axis=0)).groupby(laps.lap.to_numpy()).mean().loc[:n_laps].to_numpy()
    (lo, hi), (clo, chi) = (np.percentile(x, m["band"], axis=0) for x in (hd, 1 - np.cumprod(1 - hd, axis=1)))
    return pd.DataFrame({"lap": np.arange(1, n_laps + 1), "p": h.sum(0), "p_lo": lo, "p_hi": hi, "p_sc": h[0], "p_vsc": h[1],
                         "cum": 1 - np.cumprod(1 - h.sum(0)), "cum_lo": clo, "cum_hi": chi,
                         "avg": avg, "avg_cum": 1 - np.cumprod(1 - avg)})


def expected_observed(lap_pred):
    """Incidents the model expected on the hold-out laps against the incidents that happened, by kind, split two
    ways: by test season and by where in the race (lap 1, lap 2, later)."""
    where = np.select([lap_pred.lap == 1, lap_pred.lap == 2], ["lap 1", "lap 2"], "laps 3+")
    return pd.concat([pd.DataFrame({"by": by, "kind": kind, "group": key, "expected": lap_pred[f"p_{kind.lower()}"].to_numpy(),
                                    "observed": (lap_pred.kind == kind).to_numpy()}).groupby(["by", "kind", "group"]).sum()
                      for by, key in (("season", lap_pred.year.astype(str).to_numpy()), ("lap", where))
                      for kind in KINDS]).reset_index()


def coefficients(model):
    """The pooled logistic's coefficients as a table: log-odds, standard error, and the odds multiplier."""
    names = ["baseline", *model["features"]]
    return pd.concat({k: pd.DataFrame({"coef": model[k]["res"].params, "se": model[k]["res"].bse,
                                       "odds_x": np.exp(model[k]["res"].params)}, index=names) for k in KINDS})


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pd.set_option("display.width", 250)
    cfg = load_config()
    m, out = cfg["sc_model"], ROOT / "outputs"
    out.mkdir(exist_ok=True)
    every = at_risk(pd.read_csv(ROOT / "data" / "races.csv"), pd.read_csv(ROOT / "data" / "sc_events.csv"), cfg)
    rate = every.groupby("wet").kind.agg(laps="size", incidents=lambda k: (k != "").sum())
    print("incidents per 100 at-risk laps, dry / wet:", (100 * rate.incidents / rate.laps).round(2).tolist(), rate.to_dict("list"))
    laps = every[~every.wet]

    # 1. which features, and what each multiplier buys, on the hold-out seasons
    variants = {name: (f, ()) for name, f in m["candidates"].items()}
    variants.update({"chosen, no circuit multiplier": (None, ("circuit",)), "chosen, no season multiplier": (None, ("season",)),
                     "chosen, no multipliers": (None, ("circuit", "season"))})
    rows = []
    for name, (features, off) in variants.items():
        v, _ = validate(laps, cfg, features, off)
        rows.append({"variant": name, "logloss_per_100_laps": 100 * v.logloss.sum() / v.at_risk.sum(),
                     "brier_any": ((v.p_any - v["any"]) ** 2).mean(), "brier_sc": ((v.p_sc - v.sc) ** 2).mean()})
    variants = pd.DataFrame(rows).round(4)
    variants.to_csv(out / "sc_variants.csv", index=False)
    print("\n-- hold-out comparison (lower is better) --\n" + variants.to_string(index=False))

    # 2. the chosen model on the hold-out seasons
    v, lap_pred = validate(laps, cfg)
    v.round(4).to_csv(out / "sc_validation.csv", index=False)
    scores = pd.concat({what: brier(v, what) for what in ("any", "sc")}).round(4)
    scores.to_csv(out / "sc_scores.csv")
    print("\n-- Brier scores on the hold-out (any = at least one incident, sc = at least one SC or red flag) --\n" + scores.to_string())
    eo = expected_observed(lap_pred).round(2)
    eo.to_csv(out / "sc_expected_observed.csv", index=False)
    print("\n-- incidents expected and observed on the hold-out laps --\n" + eo.to_string(index=False))

    # 3. the model fitted on everything
    model = fit(laps, cfg)
    coefficients(model).round(3).to_csv(out / "sc_coefficients.csv")
    print("\n-- pooled logistic, all dry races --\n" + coefficients(model).round(3).to_string())
    for k in KINDS:
        print(f"{k}: prior strength circuit {model[k]['circuit_strength']:.0f}, season {model[k]['season_strength']:.0f}; "
              f"{len(model[k]['pool'])} incidents, types {model[k]['pool'].type.value_counts().to_dict()}")
    last = int(laps.year.max())
    print(f"\nseason multipliers:\n" + pd.DataFrame({k: {y: multiplier(r, model[k]["season_strength"]) for y, r in model[k]["season"].items()}
                                                      for k in KINDS}).round(2).T.to_string())
    rng = np.random.default_rng(m["seed"])
    circuit, n_laps = "Las Vegas", 50
    for year in (None, last):
        h, hd = hazard(model, circuit, n_laps, year), hazard(model, circuit, n_laps, year, m["draws"], rng)
        lo, hi = np.percentile(p_at_least_one(hd), m["band"])
        print(f"{circuit}, {n_laps} laps, season effect {year}: P(at least one incident) {p_at_least_one(h):.3f} "
              f"({lo:.2f} to {hi:.2f}); P(SC or red) {p_at_least_one(h[:1]):.3f}; P(VSC) {p_at_least_one(h[1:]):.3f}; "
              f"expected SC-kind {h[0].sum():.2f}, VSC {h[1].sum():.2f}; "
              f"circuit multipliers {[round(multiplier(model[k]['circuit'][circuit], model[k]['circuit_strength']), 2) for k in KINDS]}")

    # 4. the charts, from the tables just written
    from src import plots
    table = by_lap(model, laps, circuit, n_laps, last, rng)
    table.round(4).to_csv(out / "sc_by_lap_las_vegas.csv", index=False)
    plots.sc_by_lap(table, f"{circuit} {last}", out / "chart1_sc_by_lap.png")
    plots.sc_calibration(v, eo, out / "chart2_sc_calibration.png")
    print("charts -> outputs/chart1_sc_by_lap.png, outputs/chart2_sc_calibration.png")
