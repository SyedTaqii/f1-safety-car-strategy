"""Event-table logic on a made-up race, then on two real ones whose safety cars are known."""
import pandas as pd
import pytest

from src.data_loader import ROOT, load_config
from src.sc_events import build, check_against_rc, race_events

START = 1000.0                                             # lights out, on the session clock
LEAD_END = [START + 100 * n for n in range(1, 11)]         # a 10-lap race, 100 s a lap
STATUS = pd.DataFrame([
    (0, "1"), (500, "5"), (600, "1"),                      # a red flag before the start: not a race event
    (1250, "6"), (1300, "7"), (1310, "4"), (1520, "1"),    # lap 3: VSC, upgraded to SC with no green in between
    (1650, "2"), (1660, "1"),                              # a yellow is green
    (1700, "4"), (1750, "5"), (1790, "1"), (1820, "4"), (1850, "1"),   # SC as lap 8 starts -> red flag -> track status
                                                           # clears during the stoppage -> SC leads the restart: one incident
    (1960, "6"),                                           # 1.1 laps of green later: a new incident. Still out at the flag
    (2100, "4"),                                           # after the flag: ignored
], columns=["time_s", "status"])


def test_race_events_made_up_race():
    ev = race_events(STATUS, LEAD_END, START, same_incident_laps=1)
    got = list(ev.fillna({"laps": -1})[["episode", "type", "start_lap", "end_lap", "laps", "upgraded"]]
               .itertuples(index=False, name=None))
    assert got == [(1, "SC", 3, 6, 2.7, True),
                   (2, "SC", 8, 8, -1, False), (2, "RED", 8, 8, -1, False), (2, "SC", 9, 9, -1, False),   # no lengths in a red-flag incident
                   (3, "VSC", 10, 10, 0.4, False)]
    assert (ev.start_s.iloc[0], ev.end_s.iloc[-1]) == (250.0, 1000.0)      # seconds since lights out


def test_check_against_race_control():
    # race control doesn't announce the SC that leads the restart (lap 9); it does talk after the flag, and a
    # stewards' note can quote a call: none of those count
    rc = pd.DataFrame({"time": range(9), "lap": [3, 4, 7, 8, 9, 10, 10, 10, 10],
                       "message": ["VSC DEPLOYED", "SAFETY CAR DEPLOYED", "SAFETY CAR DEPLOYED", "RED FLAG",
                                   "INCIDENT NOTED - OVERTAKING UNDER SAFETY CAR DEPLOYED", "VIRTUAL SAFETY CAR DEPLOYED",
                                   "CHEQUERED FLAG", "RED FLAG", "VIRTUAL SAFETY CAR DEPLOYED"]})
    verdict, detail = check_against_rc(STATUS, LEAD_END, START, rc, 1)
    assert verdict == "lap +-1" and detail == "SC: track status laps [4, 8], race control laps [4, 7]"
    assert check_against_rc(STATUS, LEAD_END, START, rc.iloc[1:], 1)[0] == "count differs"   # the VSC call is missing
    assert check_against_rc(STATUS, LEAD_END, START, None, 1)[0] == "no race control"


needs_raw = pytest.mark.skipif(not (ROOT / load_config()["raw_dir"] / "races.csv").exists(),
                               reason="needs the downloaded raw tables")


@needs_raw
def test_exact_races():
    events, races = build(load_config())
    of = lambda year, rnd: list(events[(events.year == year) & (events["round"] == rnd)]
                                [["type", "start_lap", "upgraded"]].itertuples(index=False, name=None))
    # Sepang 2026 (project #1): SC on lap 9; VSC on lap 43 that became an SC
    assert of(2026, 16) == [("SC", 9, False), ("SC", 43, True)]
    sepang = races[(races.year == 2026) & (races["round"] == 16)].iloc[0]
    assert (sepang.circuit, sepang.laps, bool(sepang.wet), bool(sepang.street)) == ("Sepang", 55, True, False)
    # Las Vegas 2025 (OpenF1 race control, an independent source): VSC on lap 2 and on lap 16, nothing else
    assert of(2025, 22) == [("VSC", 2, False), ("VSC", 16, False)]


# What Wikipedia's race reports say (read 2026-11-10; for every race but Las Vegas 2025, before our table had it).
# (type, earliest lap, latest lap) it should start on. One lap = the article states the lap. A range = the article
# gives the incident's lap, or only the story, and the deployment lap is bracketed from that.
# Source: https://en.wikipedia.org/wiki/<year>_<name>_Grand_Prix
KNOWN = {
    (2018, 4): [("SC", 1, 1), ("SC", 40, 41)],                      # Azerbaijan: first-lap SC; the Red Bulls collide on lap 40
    (2019, 20): [("SC", 52, 54), ("SC", 66, 67)],                   # Brazil: Bottas stops on lap 52; the Ferraris collide on lap 66
    (2020, 8): [("SC", 19, 21), ("SC", 23, 27), ("RED", 23, 27)],   # Italy: Magnussen lap 19; Leclerc's crash, SC then red; restart lap 28
    (2020, 15): [("RED", 1, 1), ("SC", 3, 3), ("SC", 54, 55)],      # Bahrain: Grosjean; Stroll at the lap-3 restart; Perez lap 54, to the flag
    (2021, 6): [("SC", 30, 31), ("SC", 45, 47), ("RED", 45, 49)],   # Azerbaijan: Stroll lap 30; Verstappen's tyre, SC then red; restart lap 50
    (2021, 22): [("VSC", 35, 36), ("SC", 53, 53)],                  # Abu Dhabi: Giovinazzi lap 35; Latifi lap 53
    (2022, 16): [("VSC", 10, 11), ("SC", 44, 49)],                  # Italy: Vettel lap 10; Ricciardo (lap not stated), SC to the flag
    (2023, 3): [("SC", 1, 1), ("SC", 7, 8), ("RED", 7, 9), ("VSC", 18, 18),      # Australia: Leclerc; Albon lap 7, SC then red;
                ("RED", 54, 56), ("RED", 56, 58)],                               # Russell lap 18; Magnussen lap 55; the restart crash
    (2023, 21): [("VSC", 1, 1), ("SC", 3, 4), ("SC", 24, 28)],      # Las Vegas: turn 1; Norris crashes on lap 3; Russell-Verstappen debris
    (2024, 6): [("VSC", 20, 24), ("SC", 26, 30)],                   # Miami: Verstappen's bollard; Magnussen-Sargeant, SC in on lap 32
    (2024, 22): [],                                                 # Las Vegas: none
    (2025, 22): [("VSC", 2, 2)],                                    # Las Vegas: Lawson. The article leaves out a 77 s VSC on lap 16
}


def unmatched(events, year, rnd, known):
    """(what the article has and we don't, what we have and the article doesn't) for one race."""
    ours = events[(events.year == year) & (events["round"] == rnd)]
    left = list(zip(ours.type, ours.start_lap))
    missing = []
    for kind, lo, hi in known:
        hit = next((e for e in left if e[0] == kind and lo <= e[1] <= hi), None)
        left.remove(hit) if hit else missing.append((kind, lo, hi))
    return missing, left


@needs_raw
def test_known_races_against_wikipedia():
    events, races = build(load_config())
    have = set(zip(races.year, races["round"]))
    assert set(KNOWN) <= have
    result = {k: unmatched(events, *k, v) for k, v in KNOWN.items()}
    # The one article event we don't have where it says: the Monza 2022 VSC, "lap 10" there. Track status puts it
    # 983 s after lights out, which is the leader's lap 12, and race control's own message says lap 12
    assert {k: missing for k, (missing, _) in result.items() if missing} == {(2022, 16): [("VSC", 10, 11)]}
    # What we have and the articles leave out. Race control confirms both: a VSC of 77 s at Las Vegas 2025, and
    # the SC that came out a lap before Melbourne 2023's second red flag
    assert {k: extra for k, (_, extra) in result.items() if extra} == {
        (2022, 16): [("VSC", 12)], (2023, 3): [("SC", 54)], (2025, 22): [("VSC", 16)]}
