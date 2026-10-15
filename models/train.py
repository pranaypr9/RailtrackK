"""
Train the PyTorch LSTM ETA model on the (real-history-grounded) synthetic
journey dataset produced by data/build_dataset.py.

Run:
    python3 models/train.py
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from models.model import JourneySeqDataset, collate_pad, ETALSTM, FEATURE_COLS, TARGET_COL

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR = os.path.dirname(__file__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def prepare_features(df: pd.DataFrame, stats=None):
    df = df.copy()

    # cyclical encodings for time features
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    for w in ["clear", "rain", "fog", "storm"]:
        df[f"weather_{w}"] = (df["weather"].str.lower() == w).astype(float)
    for c in ["low", "medium", "high"]:
        df[f"congestion_{c}"] = (df["congestion"].str.lower() == c).astype(float)

    norm_cols = ["dist_remaining_km", "cum_distance_km", "avg_hist_delay_min",
                 "delay_so_far_min", "target_delay_min"]
    if stats is None:
        stats = {c: {"mean": float(df[c].mean()), "std": float(df[c].std() + 1e-6)} for c in norm_cols}
    for c in norm_cols:
        df[f"{c}_norm"] = (df[c] - stats[c]["mean"]) / stats[c]["std"]

    return df, stats


def masked_mse(pred, target, mask):
    diff2 = (pred - target) ** 2 * mask
    return diff2.sum() / mask.sum().clamp(min=1)


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "journeys.csv"))
    df, stats = prepare_features(df)

    run_ids = df["run_id"].unique().tolist()
    train_ids, val_ids = train_test_split(run_ids, test_size=0.15, random_state=42)

    train_ds = JourneySeqDataset(df, train_ids)
    val_ds = JourneySeqDataset(df, val_ids)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_pad)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_pad)

    model = ETALSTM(n_features=len(FEATURE_COLS)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)

    best_val = float("inf")
    epochs = 25
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb, mask, lengths in train_loader:
            xb, yb, mask = xb.to(DEVICE), yb.to(DEVICE), mask.to(DEVICE)
            opt.zero_grad()
            pred = model(xb, lengths)
            loss = masked_mse(pred, yb, mask)
            loss.backward()
            opt.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb, mask, lengths in val_loader:
                xb, yb, mask = xb.to(DEVICE), yb.to(DEVICE), mask.to(DEVICE)
                pred = model(xb, lengths)
                loss = masked_mse(pred, yb, mask)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_ds)
        sched.step(val_loss)

        # convert normalized MSE loss to approx real-world MAE (minutes) for readability
        std = stats["target_delay_min"]["std"]
        approx_mae_min = (val_loss ** 0.5) * std

        print(f"Epoch {epoch:2d}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  ~val_RMSE={approx_mae_min:.1f} min")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "eta_lstm.pt"))

    with open(os.path.join(OUT_DIR, "norm_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # ---- Baseline comparison: "static schedule" baseline = the REAL historical
    # average delay for that station (avg_hist_delay_min, taken straight from
    # the source IR dataset). This is exactly what today's static, schedule-
    # based ETA estimates amount to in practice, per the problem statement.
    # Our dynamic LSTM model instead conditions on the live journey-so-far
    # (weather, congestion, delay accumulated on THIS run) to correct that
    # static estimate. This is the key number for the hackathon pitch.
    all_targets, all_preds_model, all_baseline_norm = [], [], []
    baseline_feat_idx = FEATURE_COLS.index("avg_hist_delay_min_norm")
    model.eval()
    with torch.no_grad():
        for xb, yb, mask, lengths in val_loader:
            xb_d, yb_d, mask_d = xb.to(DEVICE), yb.to(DEVICE), mask.to(DEVICE)
            pred = model(xb_d, lengths)
            m = mask_d.bool()
            all_targets.append(yb_d[m].cpu().numpy())
            all_preds_model.append(pred[m].cpu().numpy())
            all_baseline_norm.append(xb_d[:, :, baseline_feat_idx][m].cpu().numpy())

    y_true_norm = np.concatenate(all_targets)
    y_pred_norm = np.concatenate(all_preds_model)
    y_base_norm = np.concatenate(all_baseline_norm)

    mean_, std_ = stats["target_delay_min"]["mean"], stats["target_delay_min"]["std"]
    y_true = y_true_norm * std_ + mean_
    y_pred = y_pred_norm * std_ + mean_
    # static baseline = real historical avg delay at that station (avg_hist_delay_min
    # is itself normalized with the SAME target stats since it's on a similar scale)
    base_mean = stats["avg_hist_delay_min"]["mean"]
    base_std = stats["avg_hist_delay_min"]["std"]
    y_base = y_base_norm * base_std + base_mean

    mae_model = np.mean(np.abs(y_true - y_pred))
    mae_base = np.mean(np.abs(y_true - y_base))
    improvement = (1 - mae_model / mae_base) * 100

    print("\n===== Model vs Static-Schedule Baseline (validation set) =====")
    print(f"Static baseline MAE : {mae_base:.1f} min")
    print(f"LSTM model MAE      : {mae_model:.1f} min")
    print(f"Improvement         : {improvement:.1f}%")

    with open(os.path.join(OUT_DIR, "eval_report.json"), "w") as f:
        json.dump({
            "baseline_mae_min": float(mae_base),
            "model_mae_min": float(mae_model),
            "improvement_pct": float(improvement),
        }, f, indent=2)

    print(f"\nSaved best model -> {os.path.join(OUT_DIR, 'eta_lstm.pt')}")
    print(f"Saved norm stats  -> {os.path.join(OUT_DIR, 'norm_stats.json')}")
    print(f"Saved eval report -> {os.path.join(OUT_DIR, 'eval_report.json')}")


if __name__ == "__main__":
    main()
