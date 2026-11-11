"""SC hazard model: the at-risk table on a made-up race, the shrinkage strength on made-up groups, and the sampler
against the fitted hazards on the real event table (data/*.csv is committed, so nothing here needs FastF1)."""
import numpy as np
import pandas as pd
import pytest

from src import sc_model as M
from src.data_loader import ROOT, load_config

CFG = load_config()


def test_at_risk_made_up_race():
    races = pd.DataFrame({"year": [2030], "round": [1], "circuit": ["X"], "street": [True], "wet": [False], "laps": [10]})
    # a VSC on lap 3 that becomes an SC on lap 4 (one incident, laps 3-6), then a short VSC inside lap 9
    events = pd.DataFrame({"year": 2030, "round": 1, "episode": [1, 1, 2], "type": ["VSC", "SC", "VSC"],
                           "start_lap": [3, 4, 9], "end_lap": [4, 6, 9], "laps": [0.8, 2.1, 0.5]})
    laps = M.at_risk(races, events, CFG)
    assert laps.lap.tolist() == [1, 2, 3, 7, 8, 9, 10]                      # laps 4-6 belong to the first incident
    assert laps.kind.tolist() == ["", "", "SC", "", "", "VSC", ""]          # the worst row names the incident
    first = laps[laps.lap == 3].iloc[0]
    assert (first.span, first.length) == (3, pytest.approx(2.9))
    assert laps.since.fillna(-1).tolist() == [-1, -1, -1, 1, 2, 3, 1]
    assert laps.lap1.tolist() == [True] + [False] * 6 and laps.street.all()


def test_prior_strength():
    rng = np.random.default_rng(0)
    exp = np.full(400, 10.0)
    assert M.prior_strength(rng.poisson(exp), exp, [1, 100]) > 99                                  # groups that don't differ
    assert 3 < M.prior_strength(rng.poisson(exp * rng.gamma(5, 1 / 5, 400)), exp, [1, 100]) < 8    # true strength 5
    assert M.multiplier((4, 2), 2) == 1.5                                                           # (4 + 2) / (2 + 2)


def test_sample_agrees_with_the_hazard():
    laps = M.at_risk(pd.read_csv(ROOT / "data" / "races.csv"), pd.read_csv(ROOT / "data" / "sc_events.csv"), CFG)
    model = M.fit(laps[~laps.wet], CFG)
    n, h = 20000, M.hazard(model, "Las Vegas", 50, 2026)
    s = M.sample(model, "Las Vegas", 50, np.random.default_rng(1), n, year=2026)
    assert s.sim.nunique() / n == pytest.approx(M.p_at_least_one(h), abs=0.015)
    assert (s.lap == 1).sum() / n == pytest.approx(h[:, 0].sum(), abs=0.015)             # lap 1 is always at risk
    assert (s.groupby("sim").lap.shift(-1) > s.end_lap).sum() == len(s) - s.sim.nunique()   # none starts while one runs
    assert s.end_lap.max() <= 50 and set(s.type) == {"SC", "VSC", "RED"} and s.laps[s.type == "RED"].isna().all()
    assert M.p_at_least_one(M.hazard(model, "Las Vegas", 50)) < M.p_at_least_one(h)      # 2026's VSC rate pushes it up
    assert M.hazard(model, "A new circuit", 60).shape == (2, 60)                         # no record: multiplier 1
