"""Real-data check on 2026 round 16 (Sepang), whose facts project #1 verified. Needs that race in cache/, or an
F1TV sign-in to download it."""
from src.data_loader import load_config, load_session, race_tables


def test_race_tables_sepang_2026():
    cfg = load_config()
    t = race_tables(load_session(2026, 16, "R", cfg), cfg)
    race, clock = t["races"].iloc[0], t["lap_clock"]
    assert (race.year, race["round"], race.laps, race.location, race.drivers) == (2026, 16, 55, "Kuala Lumpur", 22)
    assert list(clock.lap) == list(range(1, 56)) and clock.lead_end_s.is_monotonic_increasing

    # SC deployed on lap 9; VSC on lap 43, upgraded to SC. Race control calls the upgrade lap 45; on the leader's
    # clock it is lap 46, because VER had crossed the line 7.4 s before the SC came out (the cars behind him had not)
    lap_of = lambda time_s: int((clock.lead_end_s <= time_s).sum()) + 1
    deployed = t["track_status"][t["track_status"].status.isin(["4", "6"])]
    assert [(lap_of(x.time_s), x.status) for x in deployed.itertuples()] == [(9, "4"), (43, "6"), (46, "4")]
    called = t["race_control"][t["race_control"].message.str.contains("DEPLOYED")]
    assert list(called.lap) == [9, 43, 45]

    assert race.wet_tyre_share > 0 and race.rain_share == 0   # inters early on, but no rain during the race
    assert race.results_ok
