"""FastF1 load + cache, and the bulk race download (`python -m src.data_loader`)."""
import sys
import time
from pathlib import Path

import fastf1
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
RACE_COLUMNS = ["year", "round", "event", "location", "country", "date", "format", "laps", "drivers", "start_s",
                "rain_share", "wet_tyre_share", "results_ok", "seconds", "error"]


def load_config(path=ROOT / "config.yaml"):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))    # utf-8: circuit names (Montréal); Windows defaults to cp1252


def load_session(year, rnd, kind, cfg):
    """One session by year + round number (event names are ambiguous: the 2026 'Bahrain GP' ran at Sepang).
    Laps, track status, race control and weather; no telemetry (D-003). A cached session needs no F1 login.
    Raises DataNotLoadedError if the timing data didn't arrive (not signed in, or F1 has none)."""
    cache = ROOT / cfg["cache_dir"]
    cache.mkdir(exist_ok=True)
    fastf1.Cache.configure(cache_dir=str(cache))
    session = fastf1.get_session(year, rnd, kind)
    session.load(laps=True, telemetry=False, weather=True, messages=True)
    session.laps, session.track_status      # FastF1 only logs a failed download; touching the data raises
    return session


def completed_races(cfg, today=None):
    """The schedule rows of every race day before `today` (UTC), seasons.first..last, newest first."""
    today = today or pd.Timestamp.now("UTC").tz_localize(None).normalize()
    years = range(cfg["seasons"]["first"], cfg["seasons"]["last"] + 1)
    # plain DataFrame: on FastF1's EventSchedule, `.year` is the schedule's own attribute, not the column
    s = pd.DataFrame(pd.concat(fastf1.get_event_schedule(y, include_testing=False).assign(year=y) for y in years))
    return s[s.EventDate < today].sort_values(["year", "RoundNumber"], ascending=False)


def race_tables(session, cfg):
    """The small tables the safety-car work needs from one race, as plain DataFrames keyed by year + round:
    races (one row), lap_clock (one row per lap), track_status, session_status and race_control (as FastF1
    gives them, times in seconds of session time)."""
    ev, laps = session.event, session.laps
    seconds = lambda t: t.dt.total_seconds()
    key = {"year": int(ev.EventDate.year), "round": int(ev.RoundNumber)}
    keyed = lambda df: df.assign(**key)[list(key) + list(df.columns)]

    by_lap = laps.assign(lap=laps.LapNumber.astype(int), lap_s=seconds(laps.LapTime), pit_in=laps.PitInTime.notna(),
                         wet=laps.Compound.isin(cfg["wet_compounds"])).groupby("lap")
    clock = pd.DataFrame({
        "lead_end_s": seconds(by_lap.Time.min()),      # the first car to finish lap n is the leader: the race's lap clock
        "cars": by_lap.Driver.nunique(), "median_lap_s": by_lap.lap_s.median(),
        "pit_ins": by_lap.pit_in.sum(), "wet_tyre_share": by_lap.wet.mean()}).reset_index()

    status = session.session_status
    started = status.Time[status.Status == "Started"]
    start = started.iloc[0] if len(started) else laps.LapStartTime.min()
    w = session.weather_data
    in_race = w[(w.Time >= start) & (w.Time <= laps.Time.max())]
    race = dict(key, event=ev.EventName, location=ev.Location, country=ev.Country, date=ev.EventDate.date(),
                format=ev.EventFormat, laps=int(clock.lap.max()), drivers=laps.Driver.nunique(),
                start_s=start.total_seconds(), rain_share=in_race.Rainfall.mean(),
                wet_tyre_share=laps.Compound.isin(cfg["wet_compounds"]).mean(),
                results_ok=bool(session.results.Status.notna().all()))   # False: FastF1 fell back to timing-derived results
    rcm = session.race_control_messages
    return {
        "lap_clock": keyed(clock),
        "track_status": keyed(pd.DataFrame({"time_s": seconds(session.track_status.Time),
                                            "status": session.track_status.Status,
                                            "message": session.track_status.Message})),
        "session_status": keyed(pd.DataFrame({"time_s": seconds(status.Time), "status": status.Status})),
        "race_control": keyed(rcm[["Time", "Lap", "Category", "Flag", "Scope", "Status", "Message"]]
                              .rename(columns=str.lower)),
        "races": pd.DataFrame([race]),       # last: download_races writes it last, so a crash leaves the race undone
    }


def download_races(cfg):
    """Load every completed race into the FastF1 cache, newest first, and append its race_tables to raw_dir/*.csv.
    Safe to re-run: a race already in races.csv without an error is skipped. Readers should drop duplicate rows
    (a crash between two appends repeats that race's rows on the next run)."""
    d, raw = cfg["download"], ROOT / cfg["raw_dir"]
    raw.mkdir(parents=True, exist_ok=True)
    done = set()
    if (raw / "races.csv").exists():
        old = pd.read_csv(raw / "races.csv")
        done = set(zip(old.year[old.error.isna()], old["round"][old.error.isna()]))
    fastf1.set_log_level("ERROR")
    todo = [r for r in completed_races(cfg).itertuples() if (r.year, r.RoundNumber) not in done]
    print(f"{len(done)} races already done, {len(todo)} to load", flush=True)
    failures = 0
    for i, r in enumerate(todo, 1):
        t0, label = time.time(), f"[{i}/{len(todo)}] {r.year} R{r.RoundNumber} {r.EventName}"
        while True:
            try:
                tables, failures = race_tables(load_session(r.year, r.RoundNumber, "R", cfg), cfg), 0
            except fastf1.RateLimitExceededError as e:      # FastF1's own guard for F1's servers: wait, never bypass
                print(f"{label}: rate limit ({e}); waiting {d['rate_limit_wait_s']} s", flush=True)
                time.sleep(d["rate_limit_wait_s"])
                continue
            except Exception as e:                          # one bad race must not stop a 4-hour job
                tables, failures = {"races": pd.DataFrame([{"year": r.year, "round": r.RoundNumber, "event": r.EventName,
                                                            "error": f"{type(e).__name__}: {e}"}])}, failures + 1
            break
        tables["races"] = tables["races"].assign(seconds=round(time.time() - t0, 1)).reindex(columns=RACE_COLUMNS)
        for name, df in tables.items():
            df.to_csv(raw / f"{name}.csv", mode="a", header=not (raw / f"{name}.csv").exists(), index=False)
        race = tables["races"].iloc[0]
        print(f"{label}: {race.error if pd.notna(race.error) else f'{race.laps} laps'} ({race.seconds} s)", flush=True)
        if failures >= d["max_consecutive_failures"]:
            sys.exit(f"{failures} races in a row failed. Signed in? Check: python -m fastf1 auth f1tv --status")


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")    # an event name the console can't encode must not crash the loop
    download_races(load_config())
