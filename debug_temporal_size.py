"""
Does the leak-fix gut the temporal signal? Pure counting, no model.

Loads the RAW cohort (no landmark, no row-drop) and, for every patient, counts
how many temporal observations survive each windowing rule:

  RAW_ALL   : all observations on record
  W[-6,24]  : fixed leak-free window (what the fix uses)
  W_onset   : [-6, min(24, onset)] (old leaky cutoff, positives only)

Reports per class: obs/patient median+mean, % patients with <3 obs, and the
number of distinct timestamps in [-6,24] (= RNN sequence length). Also prints
how many patients would be DROPPED by the landmark rule (onset <= 24h), so you
can see separately (a) sequence shrinkage from the window and (b) cohort
shrinkage from the landmark.

Run:  python diag_temporal_size.py
"""
import numpy as np, pandas as pd
from utils.class_patient import Patients
from TimeEmbeddingVal import get_all_temporal_features


def count_obs(patient, feats, start_h, end_h):
    intime = patient.intime
    n_obs, ts_set = 0, set()
    for name in feats:
        mv = patient.measures.get(name, None)
        if mv is None or not hasattr(mv, "items"):
            continue
        for ts, _ in mv.items():
            h = (pd.Timestamp(ts) - intime).total_seconds() / 3600
            if start_h <= h <= end_h:
                n_obs += 1; ts_set.add(pd.Timestamp(ts))
    return n_obs, len(ts_set)


def summarize(tag, arr):
    arr = np.array(arr)
    print(f"    {tag:10s} median={np.median(arr):6.1f}  mean={arr.mean():6.1f}  "
          f"<3obs={100*np.mean(arr<3):5.1f}%  max={arr.max():.0f}")


def main():
    patients = Patients.loadPatients()
    feats = get_all_temporal_features(patients)
    print(f"RAW cohort n={len(patients)} | temporal features={len(feats)}\n")

    LANDMARK_H, HORIZON_H = 24.0, 48.0
    n_drop = 0
    pos = {"raw": [], "w24": [], "wonset": [], "ts24": []}
    neg = {"raw": [], "w24": [], "ts24": []}

    for p in patients.patientList:
        ispos = bool(getattr(p, "akdPositive", False))
        onset_h = p.akdTime.total_seconds()/3600 if ispos else 1e9
        if ispos and onset_h <= LANDMARK_H:
            n_drop += 1  # would be removed by landmark
        raw, _ = count_obs(p, feats, -1e9, 1e9)
        w24, ts24 = count_obs(p, feats, -6, 24)
        if ispos:
            wonset, _ = count_obs(p, feats, -6, min(24, onset_h))
            pos["raw"].append(raw); pos["w24"].append(w24)
            pos["wonset"].append(wonset); pos["ts24"].append(ts24)
        else:
            neg["raw"].append(raw); neg["w24"].append(w24); neg["ts24"].append(ts24)

    print(f"Landmark would DROP {n_drop} positives with onset <= {LANDMARK_H:.0f}h "
          f"(cohort shrinkage, separate from sequence length).\n")

    print(f"=== POSITIVE (n={len(pos['raw'])}) ===")
    summarize("RAW_ALL", pos["raw"])
    summarize("W[-6,24]", pos["w24"])
    summarize("W_onset", pos["wonset"])
    ts = np.array(pos["ts24"])
    print(f"    seq_len[-6,24] (distinct ts): median={np.median(ts):.0f} "
          f"mean={ts.mean():.1f}  <3={100*np.mean(ts<3):.1f}%  <5={100*np.mean(ts<5):.1f}%")
    keep = np.array(pos["w24"]) / np.maximum(np.array(pos["raw"]), 1)
    print(f"    frac of all-record obs inside [-6,24]: median={np.median(keep):.2f} mean={keep.mean():.2f}\n")

    print(f"=== NEGATIVE (n={len(neg['raw'])}) ===")
    summarize("RAW_ALL", neg["raw"])
    summarize("W[-6,24]", neg["w24"])
    ts = np.array(neg["ts24"])
    print(f"    seq_len[-6,24] (distinct ts): median={np.median(ts):.0f} "
          f"mean={ts.mean():.1f}  <3={100*np.mean(ts<3):.1f}%  <5={100*np.mean(ts<5):.1f}%")
    keep = np.array(neg["w24"]) / np.maximum(np.array(neg["raw"]), 1)
    print(f"    frac of all-record obs inside [-6,24]: median={np.median(keep):.2f} mean={keep.mean():.2f}\n")

    print("READ:")
    print(" - If W[-6,24] median obs is tiny (<~5) -> temporal sequences are")
    print("   nearly empty regardless of leak; the RNN had little to chew on.")
    print(" - If W_onset (positives) >> W[-6,24], the OLD code was feeding the")
    print("   post-... data the fix removed (that was the leak, correctly cut).")
    print(" - If 'frac inside [-6,24]' is low, most measurements land after 24h,")
    print("   so a 24h landmark inherently sees little — a cohort/task issue, not")
    print("   a bug in the fix.")


if __name__ == "__main__":
    main()