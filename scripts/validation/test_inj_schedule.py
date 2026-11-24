#!/usr/bin/env python3
"""CI gate for the yearly SO2 injection schedule (src/inj_schedule.py; no GPU).

Checks the shipped schedule file parses, that the rate steps on the 365-day year
boundary and holds the last row, that the checksum tells schedules apart, and
that malformed files are refused.
"""
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
import inj_schedule as S  # noqa: E402

failed = 0


def check(label, ok):
    global failed
    print(f"  [{'ok' if ok else 'FAILED'}] {label}")
    failed += not ok


def refused(text):
    with tempfile.NamedTemporaryFile('w', suffix='.csv', delete=False) as f:
        f.write(text)
    try:
        S.load(f.name)
    except SystemExit:
        return True
    return False


f = REPO / "inputs/injection/ccmi2022_sai_30NS_2030-2034.csv"
r = S.load(f)
check("shipped CCMI schedule parses (5 rows)", r.tolist() == [4.27, 5.15, 6.0, 6.39, 6.87])
Y = S.HOURS_PER_YEAR
check("year 0 at hour 0 and at the last hour of the year",
      S.rate_at(r, 0) == 4.27 and S.rate_at(r, Y - 1) == 4.27)
check("steps to year 1 at hour 8760", S.rate_at(r, Y) == 5.15)
check("last row held after the list ends", S.rate_at(r, 50 * Y) == 6.87)
check("checksum is deterministic", S.checksum(r) == S.checksum(S.load(f)))
check("checksum differs when one rate changes",
      S.checksum(r) != S.checksum([4.27, 5.15, 6.0, 6.39, 6.88]))
check("checksum is exact in float64", float(int(S.checksum(r))) == S.checksum(r))
check("bad header refused", refused("year,rate\n0,1\n"))
check("gap in year_offset refused", refused("year_offset,tg_so2_yr\n0,1\n2,1\n"))
check("negative rate refused", refused("year_offset,tg_so2_yr\n0,-1\n"))
check("empty schedule refused", refused("year_offset,tg_so2_yr\n"))

print("inj_schedule:", "OK" if not failed else f"{failed} FAILED")
sys.exit(1 if failed else 0)
