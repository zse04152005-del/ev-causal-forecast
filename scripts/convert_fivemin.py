"""Convert the five-minute UrbanEV files into the arrays used by the causal analysis.

The causal analysis (scripts/estimate_causal_5min.py, semisynthetic_5min.py, boot_cd_diff.py) reads
data/interim/fiveMin/, which this script writes from the two UrbanEV archives:

  --station-zip  UrbanEVDataset.zip with the folder 20220901-20230228_station-processed/
                 (the copy linked in the UrbanEV README; it is the only copy with station-processed/)
  --dryad-zip    UrbanEVDataset.zip from Dryad (doi:10.5061/dryad.np5hqc04z); it provides
                 zone-cleaned-aggregated/station_information.csv, the 1,362 stations of the 275 zones

Output (one .npz per variable with arrays time [T], cols [N], X [T, N] float32):
  station_{e_price,s_price,duration,occupancy,volume}.npz   (1,423 stations)
  zone_{e_price,s_price,duration,occupancy,volume}.npz      (275 zones)
  station_inf.csv, station_information_zone.csv

Usage:
  python scripts/convert_fivemin.py --station-zip data/raw/urbanev_station/UrbanEVDataset.zip \
      --dryad-zip data/raw/urbanev_dryad/UrbanEVDataset.zip --out data/interim/fiveMin
The archives are read in place; nothing is unpacked to disk.
"""
from __future__ import annotations

import argparse
import os
import zipfile

import numpy as np
import pandas as pd

ROOT = "UrbanEVDataset/20220901-20230228_"
VARS = ["e_price", "s_price", "duration", "occupancy", "volume"]


def wide(zf: zipfile.ZipFile, member: str):
    df = pd.read_csv(zf.open(member))
    time = df["time"].astype(str).to_numpy(dtype=object)
    cols = np.array([int(c) for c in df.columns[1:]], dtype=np.int64)
    return time, cols, df.iloc[:, 1:].to_numpy(dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--station-zip", required=True)
    ap.add_argument("--dryad-zip", required=True)
    ap.add_argument("--out", default=os.path.join("data", "interim", "fiveMin"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    zs = zipfile.ZipFile(a.station_zip)
    zd = zipfile.ZipFile(a.dryad_zip)
    for v in VARS:
        t, c, x = wide(zs, f"{ROOT}station-processed/features/{v}.csv")
        np.savez_compressed(os.path.join(a.out, f"station_{v}.npz"), time=t, cols=c, X=x)
        t, c, x = wide(zd, f"{ROOT}zone-cleaned-aggregated/charge_5min/{v}.csv")
        np.savez_compressed(os.path.join(a.out, f"zone_{v}.npz"), time=t, cols=c, X=x)
        print(v, "done")
    pd.read_csv(zs.open(f"{ROOT}station-processed/features/station_inf.csv")).to_csv(
        os.path.join(a.out, "station_inf.csv"), index=False)
    pd.read_csv(zd.open(f"{ROOT}zone-cleaned-aggregated/station_information.csv")).to_csv(
        os.path.join(a.out, "station_information_zone.csv"), index=False)
    print("written to", a.out)


if __name__ == "__main__":
    main()
