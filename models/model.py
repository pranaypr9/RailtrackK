"""
PyTorch model definitions for Dynamic ETA Prediction.

We frame this as a SEQUENCE regression problem: given a train's journey so
far (stations visited, delays observed, distance covered, live conditions),
predict the delay (-> ETA) at the NEXT station, and we can roll this forward
station-by-station to get a full dynamic ETA forecast for the rest of the
journey. This matches the problem statement's ask for a system that
"continuously adapts" / "dynamically updates ETAs in response to real-time
events" rather than a static one-shot prediction.

Architecture: a small LSTM encodes the sequence of station-events seen so
far; a regression head predicts the delay at the next station. At inference
time we feed the model the journey prefix up to "now" and let it predict
forward autoregressively, giving fresh predictions every time a new live
delay signal arrives (exactly the update loop used in api/main.py).
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset

FEATURE_COLS = [
    "dist_remaining_km_norm",
    "cum_distance_km_norm",
    "avg_hist_delay_min_norm",
    "delay_so_far_min_norm",
    "hour_sin", "hour_cos",
    "day_sin", "day_cos",
    "weather_clear", "weather_rain", "weather_fog", "weather_storm",
    "congestion_low", "congestion_medium", "congestion_high",
]
TARGET_COL = "target_delay_min_norm"


class JourneySeqDataset(Dataset):
    """
    Each item = one train run, as a sequence of station-events.
    x: (seq_len, n_features)  y: (seq_len,) next/target delay at each step
    """

    def __init__(self, df, run_ids):
        self.samples = []
        for rid in run_ids:
            g = df[df["run_id"] == rid].sort_values("seq")
            if len(g) < 2:
                continue
            x = torch.tensor(g[FEATURE_COLS].values, dtype=torch.float32)
            y = torch.tensor(g[TARGET_COL].values, dtype=torch.float32)
            self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_pad(batch):
    """Pad variable-length journeys to the max length in the batch."""
    lengths = torch.tensor([x.shape[0] for x, _ in batch])
    max_len = int(lengths.max())
    n_feat = batch[0][0].shape[1]
    xb = torch.zeros(len(batch), max_len, n_feat)
    yb = torch.zeros(len(batch), max_len)
    mask = torch.zeros(len(batch), max_len)
    for i, (x, y) in enumerate(batch):
        L = x.shape[0]
        xb[i, :L] = x
        yb[i, :L] = y
        mask[i, :L] = 1.0
    return xb, yb, mask, lengths


class ETALSTM(nn.Module):
    """LSTM sequence regressor for dynamic per-station delay/ETA prediction."""

    def __init__(self, n_features=len(FEATURE_COLS), hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x, lengths=None):
        out, _ = self.lstm(x)          # (B, T, H)
        pred = self.head(out).squeeze(-1)  # (B, T)
        return pred
