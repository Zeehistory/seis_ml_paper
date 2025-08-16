#!/usr/bin/env python3
"""
Preprocess earthquake waveform data for machine learning (parallelized with progress).

MIT Engaging cluster version:
- Reads catalog CSV and raw waveform files (.mseed)
- Cleans, trims, normalizes
- Extracts physics-informed features
- Saves NumPy arrays for deep learning models
- Saves CSV with features for classical ML
- Parallelized with multiprocessing + live progress/ETA

Author: Zahid Syed
"""

import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from obspy import read, UTCDateTime
from obspy.signal.filter import bandpass
from scipy.stats import skew, kurtosis
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# ================== CONFIG ================== #
USER = os.environ["USER"]

# Data locations
BASE_DIR = Path(f"/home/{USER}/orcd/pool/seis-data")
CATALOG_CSV = BASE_DIR / "catalog_2020.csv"
MSEED_DIR = BASE_DIR / "raw" / "mseed"
OUTPUT_DIR = BASE_DIR / "processed"
WAVEFORM_DIR = OUTPUT_DIR / "waveforms"

# Processing params
SAMPLING_RATE = 50          # Hz
PRE_EVENT_SEC = 60          # seconds before event
POST_EVENT_SEC = 600        # seconds after event
BANDPASS_FREQ = (0.5, 20.0) # Hz (low, high)
TARGET_LENGTH = int((PRE_EVENT_SEC + POST_EVENT_SEC) * SAMPLING_RATE)

# Output files
FEATURES_CSV = OUTPUT_DIR / "features.csv"
# ============================================= #

def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WAVEFORM_DIR.mkdir(parents=True, exist_ok=True)

def extract_physics_features(trace_data):
    pga = np.max(np.abs(trace_data))
    pgv = np.max(np.abs(np.cumsum(trace_data)))
    rms_amp = np.sqrt(np.mean(trace_data**2))
    spec_centroid = np.sum(
        np.fft.rfftfreq(len(trace_data)) * np.abs(np.fft.rfft(trace_data))
    ) / np.sum(np.abs(np.fft.rfft(trace_data)))
    skewness = skew(trace_data)
    kurt_val = kurtosis(trace_data)
    return {
        "PGA": pga,
        "PGV": pgv,
        "RMS": rms_amp,
        "SpectralCentroid": spec_centroid,
        "Skewness": skewness,
        "Kurtosis": kurt_val
    }

def pad_or_crop(data, target_len):
    if len(data) < target_len:
        return np.pad(data, (0, target_len - len(data)), mode='constant')
    elif len(data) > target_len:
        return data[:target_len]
    return data

def process_event(row):
    evid = row["id"].split(",")[0]
    event_time = UTCDateTime(pd.to_datetime(row["time"]).to_pydatetime())

    matched_files = []
    for file_path in MSEED_DIR.glob("*.mseed"):
        fname_parts = file_path.name.split("__")
        if len(fname_parts) < 3:
            continue
        try:
            start_t = UTCDateTime(fname_parts[1].replace("T", "").replace("Z", ""))
            end_t = UTCDateTime(fname_parts[2].replace("T", "").replace("Z", "").replace(".mseed", ""))
            if start_t <= event_time + POST_EVENT_SEC and end_t >= event_time - PRE_EVENT_SEC:
                matched_files.append(file_path)
        except Exception:
            continue

    if not matched_files:
        return None

    all_traces = []
    features_agg = []
    for mseed_file in matched_files:
        try:
            st = read(str(mseed_file))
            st.detrend("linear")
            st.resample(SAMPLING_RATE)
            st.trim(starttime=event_time - PRE_EVENT_SEC,
                    endtime=event_time + POST_EVENT_SEC)

            for tr in st:
                tr.data = bandpass(tr.data, BANDPASS_FREQ[0], BANDPASS_FREQ[1],
                                   SAMPLING_RATE, corners=4, zerophase=True)
                tr.data = (tr.data - np.mean(tr.data)) / np.std(tr.data)
                tr.data = pad_or_crop(tr.data, TARGET_LENGTH)

                all_traces.append(tr.data)
                features_agg.append(extract_physics_features(tr.data))
        except Exception:
            continue

    if not all_traces:
        return None

    features_mean = {k: np.mean([f[k] for f in features_agg]) for k in features_agg[0]}
    waveform_array = np.stack(all_traces)
    np.save(WAVEFORM_DIR / f"{evid}.npy", waveform_array)

    meta = {
        "EventID": evid,
        "Magnitude": row["magnitude"],
        "Latitude": row["lat"],
        "Longitude": row["lon"],
        "Depth_km": row["depth_km"],
        "WaveformFile": str(WAVEFORM_DIR / f"{evid}.npy")
    }
    meta.update(features_mean)
    return meta

def main():
    ensure_dirs()
    if not CATALOG_CSV.exists():
        raise FileNotFoundError(f"Catalog file not found: {CATALOG_CSV}")

    catalog_df = pd.read_csv(CATALOG_CSV, parse_dates=["time"])
    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", cpu_count()))
    print(f"[INFO] Using {num_workers} parallel workers")

    start_time = time.time()

    results = []
    with Pool(processes=num_workers) as pool:
        for result in tqdm(pool.imap_unordered(process_event, [row for _, row in catalog_df.iterrows()]),
                           total=len(catalog_df), desc="Processing events", unit="event"):
            if result:
                results.append(result)

    elapsed = time.time() - start_time
    print(f"[INFO] Processing completed in {elapsed/60:.2f} minutes")

    if not results:
        print("[ERROR] No events processed successfully.")
        return

    features_df = pd.DataFrame(results)
    features_df.to_csv(FEATURES_CSV, index=False)
    print(f"[INFO] Saved features CSV: {FEATURES_CSV}")
    print(f"[INFO] Waveform .npy files in: {WAVEFORM_DIR}")

if __name__ == "__main__":
    main()
