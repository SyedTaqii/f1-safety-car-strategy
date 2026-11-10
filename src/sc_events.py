"""SC / VSC / red-flag event table, built from the raw race tables (`python -m src.sc_events`).
Reads data/raw/*.csv only: no FastF1, no login."""
import numpy as np
import pandas as pd

from src.data_loader import ROOT, load_config

STATE = {"4": "SC", "5": "RED", "6": "VSC", "7": "VSC"}   # FastF1 track status. 1 (all clear) and 2 (yellow) are green
KEY = ["year", "round"]


def read_raw(name, cfg):
    """A raw table. Duplicate rows are dropped: a crash mid-download repeats a race's rows on the next run."""
    return pd.read_csv(ROOT / cfg["raw_dir"] / f"{name}.csv").drop_duplicates()


def circuit_names(races, cfg):
    """One name per track: the event-name override, else the location alias, else FastF1's Location."""
    c = cfg["circuits"]
    return races.event.map(c["by_event"]).fillna(races.location.replace(c["aliases"]))


def deployments(status, start_s, end_s):
    """One race's track status as non-green stretches inside the race: state (SC | VSC | RED), t0, t1.
    A stretch already running at the start begins at start_s; one still running at the flag ends at end_s."""
    s = status.sort_values("time_s")
    seg = pd.DataFrame({"state": s.status.astype(str).map(STATE).fillna("GREEN").to_numpy(), "t0": s.time_s.to_numpy()})
    seg = seg[seg.state != seg.state.shift()]              # 6 then 7 is one VSC; 1, 2, 1, 2 is one green
    seg["t1"] = seg.t0.shift(-1).fillna(np.inf)
    seg[["t0", "t1"]] = seg[["t0", "t1"]].clip(start_s, end_s)
    return seg[(seg.state != "GREEN") & (seg.t1 > seg.t0)].reset_index(drop=True)


def race_events(status, lead_end, start_s, same_incident_laps):
    """One race's events, one row each, in time order.

    status: its track_status rows (time_s, status). lead_end: when the leader finished each lap. start_s: lights
    out. All on the session clock.
    A VSC that turns into an SC with no green in between is one SC event (upgraded = True). A red flag is its own
    row. Events with less than same_incident_laps of green running between them share an `episode`: an SC, the
    red flag it turned into and the SC that led the restart are one incident, three rows.
    Laps are the leader's (the lap he is on when it starts or ends); `laps` is the length in leader laps. A red
    flag's end is when the track status cleared, not the restart, and `laps` is blank for every row of its
    episode: the stoppage sits inside one of the leader's laps and stretches it."""
    lead_end = np.asarray(lead_end, float)
    n = len(lead_end)
    pos = lambda t: np.interp(t, np.r_[start_s, lead_end], np.arange(n + 1))    # race distance covered, in leader laps
    lap = lambda t: int(min(np.floor(pos(t)) + 1, n))
    rows, prev_end = [], None
    for d in deployments(status, start_s, lead_end[-1]).itertuples():
        if prev_end == d.t0 and d.state != "RED" and rows[-1]["type"] != "RED":     # VSC straight into SC
            rows[-1].update(type="SC", t1=d.t1, upgraded=True)
        else:
            same = bool(rows) and pos(d.t0) - pos(prev_end) < same_incident_laps
            rows.append(dict(episode=rows[-1]["episode"] + (not same) if rows else 1, type=d.state,
                             t0=d.t0, t1=d.t1, upgraded=False))
        prev_end = d.t1
    ev = pd.DataFrame([dict(episode=r["episode"], type=r["type"], start_lap=lap(r["t0"]), end_lap=lap(r["t1"]),
                            laps=round(float(pos(r["t1"]) - pos(r["t0"])), 2), upgraded=r["upgraded"],
                            start_s=round(r["t0"] - start_s, 1), end_s=round(r["t1"] - start_s, 1)) for r in rows],
                      columns=["episode", "type", "start_lap", "end_lap", "laps", "upgraded", "start_s", "end_s"])
    ev.loc[ev.episode.isin(ev.episode[ev.type == "RED"]), "laps"] = np.nan
    return ev


def rc_calls(rc):
    """Race control's own record of one race: a row per SC or VSC deployment and per red flag, with the lap it
    gives. Only messages that start with the call (a stewards' note can quote "... UNDER SAFETY CAR DEPLOYED"), and
    only up to the chequered flag (race control keeps talking after it: a red flag at Melbourne 2023)."""
    rc = rc.sort_values("time")
    m = rc.message.fillna("")
    rc, m = rc[~m.str.match("CHEQUERED FLAG").cummax()], m[~m.str.match("CHEQUERED FLAG").cummax()]
    kind = np.select([m.str.match("(?:VSC|VIRTUAL SAFETY CAR) DEPLOYED"), m.str.match("SAFETY CAR DEPLOYED"),
                      m.str.match("RED FLAG")], ["VSC", "SC", "RED"], "")
    # int: a few unrelated messages have no lap, which makes the whole column float
    return pd.DataFrame({"type": kind, "lap": rc.lap.to_numpy()})[kind != ""].astype({"lap": int}).reset_index(drop=True)


def check_against_rc(status, lead_end, start_s, rc, same_incident_laps):
    """Compare the track-status deployments with race control's messages, type by type and in order.
    An SC that comes out within same_incident_laps of a red flag is left out: it leads the restart, and race control
    doesn't announce those.
    Returns (verdict, detail). Verdicts: ok | lap +-1 (same events, a lap apart: see D-016) | lap off |
    count differs | no race control."""
    if rc is None or rc.empty:
        return "no race control", ""
    n = len(lead_end)
    pos = lambda t: np.interp(t, np.r_[start_s, np.asarray(lead_end, float)], np.arange(n + 1))
    ts = deployments(status, start_s, lead_end[-1])
    restart = (ts.state == "SC") & (ts.state.shift() == "RED") & (pos(ts.t0) - pos(ts.t1.shift()) < same_incident_laps)
    ts = pd.DataFrame({"type": ts.state, "lap": [int(min(np.floor(pos(t)) + 1, n)) for t in ts.t0]})[~restart]
    calls, worst, detail = rc_calls(rc), 0, []
    for kind in ("SC", "VSC", "RED"):
        a, b = ts.lap[ts.type == kind].tolist(), calls.lap[calls.type == kind].tolist()
        if len(a) != len(b):
            return "count differs", f"{kind}: track status laps {a}, race control laps {b}"
        gaps = [x - y for x, y in zip(a, b)]
        if any(gaps):
            detail.append(f"{kind}: track status laps {a}, race control laps {b}")
            worst = max(worst, max(map(abs, gaps)))
    return ("ok", "lap +-1", "lap off")[min(worst, 2)], "; ".join(detail)


def build(cfg):
    """(events, races) for every downloaded race. events: one row per SC / VSC / red flag. races: one row per race
    with its circuit, type, wet flag, event counts and the race-control check."""
    raw = {name: read_raw(name, cfg) for name in ("races", "lap_clock", "track_status", "race_control")}
    races = raw["races"][raw["races"].error.isna()].drop_duplicates(KEY, keep="last").sort_values(KEY)
    bad = {(b["year"], b["round"]) for b in cfg["events"]["bad_races"]}
    races = races[[k not in bad for k in zip(races.year, races["round"])]]
    per_race = {name: dict(tuple(raw[name].groupby(KEY))) for name in ("lap_clock", "track_status", "race_control")}
    events, checks, same = [], [], cfg["events"]["same_incident_laps"]
    for r in races.itertuples():
        k = (r.year, r.round)
        lead_end = per_race["lap_clock"][k].sort_values("lap").lead_end_s.to_numpy()
        assert (np.diff(np.r_[r.start_s, lead_end]) > 0).all(), f"{k}: the leader's lap clock goes backwards"
        ev = race_events(per_race["track_status"][k], lead_end, r.start_s, same)
        if len(ev):                                        # an empty frame would turn the int columns into objects
            events.append(ev.assign(year=r.year, round=r.round))
        checks.append(check_against_rc(per_race["track_status"][k], lead_end, r.start_s, per_race["race_control"].get(k), same))
    events = pd.concat(events, ignore_index=True)
    circuit = circuit_names(races, cfg)
    counts = events.groupby(KEY + ["type"]).size().unstack(fill_value=0).reindex(columns=["SC", "VSC", "RED"], fill_value=0)
    # the largest share of the field on wet tyres on any one lap
    wet_peak = raw["lap_clock"].groupby(KEY).wet_tyre_share.max().reindex(pd.MultiIndex.from_frame(races[KEY])).to_numpy()
    races = pd.DataFrame({
        "year": races.year, "round": races["round"], "event": races.event, "circuit": circuit, "date": races.date,
        "laps": races.laps.astype(int), "street": circuit.isin(cfg["circuits"]["street"]),
        "wet": wet_peak >= cfg["wet"]["min_peak_share"], "wet_peak_share": wet_peak.round(2),
        "wet_tyre_share": races.wet_tyre_share.round(3), "rain_share": races.rain_share.round(3),
        "rc_check": [c[0] for c in checks], "rc_detail": [c[1] for c in checks]})
    races = races.join(counts.add_prefix("n_").rename(columns=str.lower), on=KEY).fillna({"n_sc": 0, "n_vsc": 0, "n_red": 0})
    races[["n_sc", "n_vsc", "n_red"]] = races[["n_sc", "n_vsc", "n_red"]].astype(int)
    events = events.merge(races[KEY + ["circuit", "laps"]].rename(columns={"laps": "race_laps"}), on=KEY)
    return events[KEY + ["circuit", "race_laps", "episode", "type", "start_lap", "end_lap", "laps", "upgraded",
                         "start_s", "end_s"]], races.reset_index(drop=True)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 110)
    cfg = load_config()
    events, races = build(cfg)
    events.to_csv(ROOT / "data" / "sc_events.csv", index=False)
    races.to_csv(ROOT / "data" / "races.csv", index=False)

    print(f"{len(races)} races, {len(events)} events -> data/sc_events.csv, data/races.csv")
    by_year = races.groupby("year").agg(races=("round", "size"), wet=("wet", "sum"), sc=("n_sc", "sum"),
                                        vsc=("n_vsc", "sum"), red=("n_red", "sum"))
    print(by_year.to_string())
    print("\nrace-control check:", races.rc_check.value_counts().to_dict())
    bad = races[~races.rc_check.isin(["ok", "lap +-1"])]
    if len(bad):
        print(bad[KEY + ["circuit", "rc_check", "rc_detail"]].to_string(index=False))
