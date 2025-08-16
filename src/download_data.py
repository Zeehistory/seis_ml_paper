#!/usr/bin/env python3
"""
Download earthquake catalog from USGS ComCat and waveform data from IRIS.

Cluster-safe version for MIT Engaging:
- Uses /home/$USER/orcd/pool/seis-data for persistent storage
- CSV datetime parsing fixed to avoid UTCDateTime errors
- Resumable: skips already-downloaded events
- Clear logging for SLURM job outputs
"""

import os
from datetime import datetime, timezone
import pandas as pd
from pathlib import Path
from libcomcat.search import search
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from obspy.clients.fdsn.mass_downloader import CircularDomain, Restrictions, MassDownloader

# ========= USER & STORAGE SETTINGS ========= #
USER = os.environ["USER"]

STARTTIME = datetime(2020, 1, 1, tzinfo=timezone.utc)
ENDTIME   = datetime(2020, 12, 31, tzinfo=timezone.utc)
MINMAG    = 5.5

BASE_DIR   = Path(f"/home/{USER}/orcd/pool/seis-data")
CATALOG_CSV = BASE_DIR / f"catalog_{STARTTIME.year}.csv"
MSEED_DIR  = BASE_DIR / "raw" / "mseed"
STXML_DIR  = BASE_DIR / "raw" / "stations"

NETWORKS   = "IU,II,GE,IC,MB"  # can expand later
CHAN_PRIOR = ["BH?", "HH?"]
PRE_EVENT_SEC  = 60
POST_EVENT_SEC = 600
# =========================================== #


def ensure_dirs():
    """Make sure necessary directories exist."""
    for d in [MSEED_DIR, STXML_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def download_catalog():
    """Fetch earthquake catalog from USGS ComCat."""
    print(f"[INFO] Querying ComCat: {STARTTIME} → {ENDTIME}, M≥{MINMAG}")
    events = search(
        starttime=STARTTIME,
        endtime=ENDTIME,
        minmagnitude=MINMAG
    )
    if not events:
        print("[WARN] No events found for given parameters.")
        return pd.DataFrame()

    df = pd.DataFrame([
        {
            "id": e.id,
            "time": e.time.datetime if hasattr(e.time, "datetime") else e.time,
            "magnitude": e.magnitude,
            "lat": e.latitude,
            "lon": e.longitude,
            "depth_km": e.depth / 1000.0 if e.depth is not None else None
        }
        for e in events
    ])

    df.to_csv(CATALOG_CSV, index=False)
    print(f"[INFO] Saved catalog with {len(df)} events to {CATALOG_CSV}")
    return df


def download_waveforms(catalog_df):
    """Download waveform & station data from IRIS."""
    md = MassDownloader(providers=["IRIS"])

    for _, row in catalog_df.iterrows():
        evid = str(row["id"]).split(",")[0]
        event_time = UTCDateTime(row["time"].to_pydatetime())
        magnitude = row["magnitude"]

        marker_file = MSEED_DIR / f"{evid}.done"
        if marker_file.exists():
            print(f"[SKIP] Event {evid} (M{magnitude}) already downloaded.")
            continue

        print(f"[INFO] Downloading Event {evid} | Time: {event_time} | M={magnitude}")

        domain = CircularDomain(
            latitude=row["lat"],
            longitude=row["lon"],
            minradius=0.0,
            maxradius=90.0
        )

        restrictions = Restrictions(
            starttime=event_time - PRE_EVENT_SEC,
            endtime=event_time + POST_EVENT_SEC,
            reject_channels_with_gaps=True,
            minimum_interstation_distance_in_m=1e3,
            network=NETWORKS,
            channel_priorities=CHAN_PRIOR,
            location_priorities=["", "00", "01"]
        )

        try:
            md.download(
                domain, restrictions,
                mseed_storage=str(MSEED_DIR),
                stationxml_storage=str(STXML_DIR)
            )
            marker_file.touch()
        except Exception as e:
            print(f"[ERROR] Failed to download event {evid}: {e}")


def main():
    ensure_dirs()

    if CATALOG_CSV.exists():
        print(f"[INFO] Loading existing catalog from {CATALOG_CSV}")
        catalog_df = pd.read_csv(CATALOG_CSV, parse_dates=["time"])
    else:
        catalog_df = download_catalog()
        if catalog_df.empty:
            return

    download_waveforms(catalog_df)
    print("[DONE] All events processed.")


if __name__ == "__main__":
    main()
