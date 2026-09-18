"""Download public Ohio weather + river flow into the pipeline CSV format.

Builds data/raw/ohio_public_daily.csv from:
  - NCEI GHCND station USW00014821 (John Glenn Columbus Intl): TMAX/TMIN/PRCP
  - USGS NWIS site 03227500 (Scioto River at Columbus OH): daily discharge as demand_proxy

Discharge is a stand-in for demand, not metered retail use.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
import gzip
import io
import urllib.request

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

GHCND_STATION = "USW00014821"
GHCND_URL = (
    "https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/"
    f"{GHCND_STATION}.csv.gz"
)
USGS_SITE = "03227500"


def usgs_dv_url(start: str, end: str) -> str:
    return (
        "https://waterservices.usgs.gov/nwis/dv/"
        f"?format=rdb&sites={USGS_SITE}&parameterCd=00060"
        f"&startDT={start}&endDT={end}"
    )


def _download(url: str) -> bytes:
    """Prefer urllib; fall back to curl when macOS Python SSL certs are missing."""
    req = urllib.request.Request(
        url, headers={"User-Agent": "water-demand-factors/0.1"}
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.read()
    except Exception:
        import subprocess
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        try:
            subprocess.run(
                ["curl", "-fsSL", "-o", tmp_path, url],
                check=True,
                timeout=180,
            )
            return Path(tmp_path).read_bytes()
        finally:
            Path(tmp_path).unlink(missing_ok=True)


def load_ghcnd_daily(start: str, end: str) -> pd.DataFrame:
    """Return daily mean temp_c and precip_mm for the pinned Columbus station."""
    raw = _download(GHCND_URL)
    text = gzip.decompress(raw).decode("utf-8", errors="replace")
    df = pd.read_csv(io.StringIO(text), header=None, low_memory=False)
    if df.shape[1] < 4:
        raise ValueError("Unexpected GHCND by_station layout")
    df = df.iloc[:, :4]
    df.columns = ["station", "date", "element", "value"]
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    wide = df.pivot_table(index="date", columns="element", values="value", aggfunc="first")
    out = pd.DataFrame(index=wide.index)
    if "TMAX" in wide.columns and "TMIN" in wide.columns:
        out["temp_c"] = ((wide["TMAX"] + wide["TMIN"]) / 2.0) / 10.0
    elif "TAVG" in wide.columns:
        out["temp_c"] = wide["TAVG"] / 10.0
    else:
        raise ValueError("GHCND file missing TMAX/TMIN or TAVG")
    if "PRCP" not in wide.columns:
        raise ValueError("GHCND file missing PRCP")
    out["precip_mm"] = wide["PRCP"] / 10.0
    return out.reset_index()


def load_usgs_discharge(start: str, end: str) -> pd.DataFrame:
    raw = _download(usgs_dv_url(start, end)).decode("utf-8", errors="replace")
    lines = [ln for ln in raw.splitlines() if ln and not ln.startswith("#")]
    usgs = pd.read_csv(io.StringIO("\n".join(lines)), sep="\t")
    if "site_no" in usgs.columns:
        usgs = usgs[usgs["site_no"].astype(str).str.fullmatch(r"\d+", na=False)]
    if "datetime" in usgs.columns:
        date_col = "datetime"
    else:
        date_candidates = [c for c in usgs.columns if "date" in c.lower()]
        if not date_candidates:
            raise ValueError(
                f"No date column in USGS response; columns={list(usgs.columns)}"
            )
        date_col = date_candidates[0]
    value_cols = [
        c for c in usgs.columns if "00060" in str(c) and not str(c).endswith("_cd")
    ]
    if not value_cols:
        raise ValueError(
            f"No discharge column in USGS response; columns={list(usgs.columns)}"
        )
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(usgs[date_col], errors="coerce"),
            "demand_proxy": pd.to_numeric(usgs[value_cols[0]], errors="coerce"),
        }
    ).dropna()
    return out


def build_ohio_public_csv(
    start: str = "2020-01-01",
    end: str | None = None,
    out_path: Path | None = None,
) -> Path:
    end = end or date.today().isoformat()
    out_path = Path(out_path) if out_path else (RAW_DIR / "ohio_public_daily.csv")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    weather = load_ghcnd_daily(start, end)
    flow = load_usgs_discharge(start, end)
    merged = pd.merge(flow, weather, on="date", how="inner").sort_values("date")
    merged = merged.dropna(subset=["demand_proxy", "temp_c", "precip_mm"])
    merged["date"] = pd.to_datetime(merged["date"]).dt.strftime("%Y-%m-%d")
    merged = merged[["date", "demand_proxy", "temp_c", "precip_mm"]]
    merged.to_csv(out_path, index=False)

    meta = RAW_DIR / "ohio_public_retrieval.txt"
    meta.write_text(
        f"retrieval_date={date.today().isoformat()}\n"
        f"start={start}\nend={end}\n"
        f"ghcnd_station={GHCND_STATION} (JOHN GLENN INTERNATIONAL AIRPORT, OH US)\n"
        f"ghcnd_url={GHCND_URL}\n"
        f"usgs_site={USGS_SITE} (Scioto River at Columbus OH)\n"
        "usgs_param=00060 daily mean discharge (cfs)\n"
        "join=inner on calendar date; USGS daily values as local calendar day\n"
        "proxy_note=demand_proxy is USGS discharge (cfs), NOT metered retail water demand\n"
    )
    return out_path


if __name__ == "__main__":
    path = build_ohio_public_csv()
    n = sum(1 for _ in open(path)) - 1
    print(f"Wrote {path} ({n} rows)")
