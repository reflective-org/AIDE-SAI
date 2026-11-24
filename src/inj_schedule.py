"""Yearly SO2 injection schedule (INJ_SO2_SCHED_FILE).

A CSV with columns year_offset, tg_so2_yr (and any further columns, ignored);
'#' lines are comments. Row k is the TOTAL SO2 release (Tg SO2/yr) in model
year k, counted in 365-day years from the run start (hour 0 = H0). The rate is
constant within a year (annual steps) and the last row holds after the list ends.

Kept out of coupling.py so CI can test it without a GPU or the input data.
"""
import zlib

import numpy as np

HOURS_PER_YEAR = 365 * 24


def load(path):
    """Return the per-year rates (Tg SO2/yr) as a float64 array."""
    offs, rates = [], []
    header = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            cols = [c.strip() for c in line.split(',')]
            if header is None:
                header = cols
                if cols[:2] != ['year_offset', 'tg_so2_yr']:
                    raise SystemExit(f"{path}: header must start with "
                                     f"'year_offset,tg_so2_yr', got {line!r}")
                continue
            offs.append(int(cols[0]))
            rates.append(float(cols[1]))
    if not rates:
        raise SystemExit(f"{path}: no schedule rows")
    if offs != list(range(len(offs))):
        raise SystemExit(f"{path}: year_offset must run 0,1,2,... without gaps, got {offs}")
    rates = np.asarray(rates, dtype=np.float64)
    if (rates < 0).any() or not np.isfinite(rates).all():
        raise SystemExit(f"{path}: rates must be finite and >= 0, got {rates.tolist()}")
    return rates


def checksum(rates):
    """CRC32 of the rates, exact in float64; stamped into INJ_CFG so a RESUME
    cannot continue under a different schedule."""
    return float(zlib.crc32(','.join(repr(float(r)) for r in rates).encode()))


def rate_at(rates, hour):
    """Rate (Tg SO2/yr) in force at model hour `hour` since the run start."""
    return float(rates[min(int(hour) // HOURS_PER_YEAR, len(rates) - 1)])
