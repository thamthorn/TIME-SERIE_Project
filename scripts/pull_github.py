"""Pull 2 years of weekly commit counts for a curated set of repos via
`git clone --bare --shallow-since` and `git log`.

Why not the REST API: the unauthenticated /commits endpoint caps at 60 req/hr,
which we burned trying to fetch 14 repos. Bare+shallow git cloning has no
rate limit and gives us exact author timestamps for every commit in the
window.

Cached CSV written to data/github_commits.csv — one row per (repo, week).
"""
from __future__ import annotations
import subprocess, tempfile
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
OUT  = HERE / "data" / "github_commits.csv"

SINCE = "2023-04-01"
UNTIL = "2025-04-01"

REPOS = [
    # (owner/repo, "alive" | "dying")
    ("pallets/flask",             "alive"),
    ("tiangolo/fastapi",          "alive"),
    ("django/django",             "alive"),
    ("psf/requests",              "alive"),
    ("numpy/numpy",               "alive"),
    ("pandas-dev/pandas",         "alive"),
    ("scikit-learn/scikit-learn", "alive"),

    ("atom/atom",                 "dying"),   # archived 2022-12
    ("jashkenas/coffeescript",    "dying"),
    ("bower/bower",               "dying"),   # archived
    ("gulpjs/gulp",               "dying"),
    ("ariya/phantomjs",           "dying"),   # dead since 2018
    ("pugjs/pug",                 "dying"),
    ("airbnb/enzyme",             "dying"),
]


def commit_timestamps(repo: str, since: str, until: str) -> list[str]:
    """Clone a bare repo shallowly and read author dates via git log."""
    url = f"https://github.com/{repo}.git"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp) / "r.git"
        # --bare : no working tree.  --shallow-since : only commits after date.
        # --filter=blob:none : don't fetch file blobs (huge bandwidth saver).
        clone = subprocess.run(
            ["git", "clone", "--bare", "--filter=blob:none",
             f"--shallow-since={since}", url, str(tmp)],
            capture_output=True, text=True, timeout=240)
        if clone.returncode != 0:
            # --shallow-since can fail for archived/ancient repos; retry full bare
            clone = subprocess.run(
                ["git", "clone", "--bare", "--filter=blob:none", url, str(tmp)],
                capture_output=True, text=True, timeout=300)
            if clone.returncode != 0:
                print(f"    clone failed: {clone.stderr[:200]}")
                return []
        log = subprocess.run(
            ["git", "-C", str(tmp), "log", "--all",
             f"--since={since}", f"--until={until}",
             "--pretty=format:%aI"],
            capture_output=True, text=True, timeout=60)
        if log.returncode != 0:
            print(f"    log failed: {log.stderr[:200]}")
            return []
        return [line for line in log.stdout.splitlines() if line.strip()]


def iso_week_start(ts: str) -> str | None:
    """Return ISO-week Monday for an author-date timestamp."""
    try:
        d = pd.to_datetime(ts, utc=True)
    except Exception:
        return None
    d_local = d - pd.Timedelta(days=d.weekday())
    return d_local.strftime("%Y-%m-%d")


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for repo, label in REPOS:
        print(f"{repo:32s} [{label}] ...", flush=True)
        stamps = commit_timestamps(repo, SINCE, UNTIL)
        print(f"    {len(stamps):5d} commits", flush=True)
        weeks: dict[str, int] = {}
        for ts in stamps:
            w = iso_week_start(ts)
            if w and SINCE <= w < UNTIL:
                weeks[w] = weeks.get(w, 0) + 1
        for w, n in weeks.items():
            rows.append({"repo": repo, "label": label, "week_start": w, "commits": n})
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}  rows={len(df)}  repos={df['repo'].nunique()}", flush=True)


if __name__ == "__main__":
    main()
