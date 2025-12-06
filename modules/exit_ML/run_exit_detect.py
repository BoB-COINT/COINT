"""
DB-backed exit scam inference using Attention MIL artifacts.

Reads preprocessed features from:
  - ExitProcessedDataInstance (per-event)
  - ExitProcessedDataStatic (per-token)
Applies attention MIL model artifacts in this directory and saves results to:
  - ExitMlResult
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

# Artifact paths (within this directory)
ARTIFACT_DIR = Path(__file__).resolve().parent
TRAINING_JSON = ARTIFACT_DIR / "attention_mil_training.json"
MODEL_PATH = ARTIFACT_DIR / "attention_mil_model.pt"
SCALER_PATH = ARTIFACT_DIR / "attention_mil_scaler.npz"

# Reported feature sets (for output)
INSTANCE_OUTPUT_FEATURES = [
    "reserve_base_drop_frac",
    "reserve_quote",
    "reserve_quote_drop_frac",
    "price_ratio",
    "time_since_last_mint_sec",
]
STATIC_OUTPUT_FEATURES = [
    "liquidity_age_days",
    "reserve_quote_drawdown_global",
]


# Django bootstrap -----------------------------------------------------------
def setup_django(settings_module: str = "config.settings") -> None:
    if "DJANGO_SETTINGS_MODULE" not in os.environ:
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", settings_module)
    import django
    from django.apps import apps

    if not apps.ready:
        django.setup()


# Model definition -----------------------------------------------------------
class AttentionPooling(nn.Module):
    def __init__(self, input_dim: int, attention_dim: int):
        super().__init__()
        self.V = nn.Linear(input_dim, attention_dim)
        self.w = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        scores = self.w(torch.tanh(self.V(h)))
        weights = torch.softmax(scores.squeeze(-1), dim=0).unsqueeze(-1)
        return weights


class SequenceStaticAttentionMIL(nn.Module):
    def __init__(
        self,
        input_dim: int,
        static_dim: int,
        hidden_dim: int = 256,
        embed_dim: int = 128,
        attention_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.use_static = static_dim > 0
        self.instance_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
        )
        self.attention = AttentionPooling(embed_dim, attention_dim)
        if self.use_static:
            self.static_proj = nn.Sequential(
                nn.Linear(static_dim, embed_dim),
                nn.ReLU(),
            )
            classifier_in = embed_dim * 2
        else:
            self.static_proj = None
            classifier_in = embed_dim
        self.head_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(classifier_in, 1)

    def forward(
        self,
        bags: Sequence[torch.Tensor],
        static_features: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        bag_embeddings: List[torch.Tensor] = []
        attention_weights: List[torch.Tensor] = []

        for bag in bags:
            if bag.ndim != 2:
                raise ValueError(f"Each bag must be 2D (instances x features); got {bag.shape}.")
            h = self.instance_encoder(bag)
            weights = self.attention(h)
            embedding = torch.sum(weights * h, dim=0)
            bag_embeddings.append(embedding)
            if return_attention:
                attention_weights.append(weights.squeeze(-1))

        bag_stack = torch.stack(bag_embeddings, dim=0)
        if self.use_static:
            static_emb = self.static_proj(static_features)
            fused = torch.cat([bag_stack, static_emb], dim=1)
        else:
            fused = bag_stack
        fused = self.head_dropout(fused)
        logits = self.classifier(fused).squeeze(-1)
        if return_attention:
            return logits, attention_weights
        return logits, None


# Utilities ------------------------------------------------------------------
def _load_training_meta(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Training summary not found at {path}")
    import json

    with path.open() as fp:
        return json.load(fp)


def _load_scaler(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Scaler npz not found at {path}")
    data = np.load(path, allow_pickle=True)
    mean = data["mean"].astype(np.float32, copy=False)
    scale = data["scale"].astype(np.float32, copy=False)
    binary_cols = list(data["binary_cols"])
    return {"mean": mean, "scale": scale, "binary_cols": binary_cols}


def _ensure_columns(frame: pd.DataFrame, feature_columns: List[str], mask_default_one: bool = True) -> pd.DataFrame:
    for col in feature_columns:
        if col not in frame.columns:
            if mask_default_one and col.endswith("_mask"):
                frame[col] = 1.0
            else:
                frame[col] = 0.0
    extra = [c for c in frame.columns if c not in feature_columns]
    if extra:
        frame = frame.drop(columns=extra)
    return frame[feature_columns]


def _select_top(weights: np.ndarray, k: int = 1) -> List[int]:
    if weights.size == 0:
        return []
    k = min(k, weights.size)
    return np.argsort(weights)[::-1][:k].tolist()


def run_exit_detection(token_info_id: int) -> Dict[str, object]:
    """Run exit MIL inference for a token (by token_info.id) using DB features."""
    setup_django()
    from api.models import (
        ExitProcessedDataInstance,
        ExitProcessedDataStatic,
        ExitMlResult,
        TokenInfo,
    )

    token_info = TokenInfo.objects.get(id=token_info_id)

    # Load training meta and scaler
    meta = _load_training_meta(TRAINING_JSON)
    params = meta.get("training_params") or meta.get("params") or {}
    feature_columns: List[str] = meta.get("feature_columns") or meta.get("instance_feature_columns") or []
    static_columns: List[str] = meta.get("static_feature_columns") or []
    best_threshold = float(meta.get("best_threshold", 0.5))

    scaler = _load_scaler(SCALER_PATH)
    mean = scaler["mean"]
    scale = scaler["scale"]

    # Load DB instance features
    qs = ExitProcessedDataInstance.objects.filter(token_info=token_info).order_by("id")
    if not qs.exists():
        raise ValueError(f"No ExitProcessedDataInstance for token_info id={token_info_id}")
    df = pd.DataFrame.from_records(qs.values())

    # Preserve meta for top instance
    meta_cols = ["event_time", "tx_hash"]
    df_meta = df[meta_cols].reset_index(drop=True)

    # Align feature columns
    df_feat = df.drop(columns=[c for c in df.columns if c in ["id", "token_info_id"] + meta_cols], errors="ignore")
    mask_cols = [c for c in df_feat.columns if c.endswith("_mask")]
    if mask_cols:
        df_feat = df_feat.drop(columns=mask_cols, errors="ignore")
    df_feat = df_feat.replace({np.inf: np.nan, -np.inf: np.nan, "": np.nan})
    df_feat = df_feat.clip(lower=-1e12, upper=1e12)
    for col in list(df_feat.columns):
        mask = df_feat[col].isna()
        if mask.any():
            df_feat[f"{col}_mask"] = (~mask).astype(float)
            df_feat[col] = df_feat[col].fillna(0.0)
    df_feat = df_feat.fillna(0.0)
    df_feat = _ensure_columns(df_feat, feature_columns, mask_default_one=True)
    df_feat = df_feat.fillna(0.0)
    safe_scale = np.where(scale == 0.0, 1.0, scale)
    X = df_feat.to_numpy(dtype=np.float32, copy=False)
    if mean.size != X.shape[1] or safe_scale.size != X.shape[1]:
        raise ValueError(f"Scaler shape mismatch: mean {mean.size}, scale {safe_scale.size}, X {X.shape[1]}")
    X_scaled = (X - mean) / safe_scale

    # Load static features
    try:
        static_obj = ExitProcessedDataStatic.objects.get(token_info=token_info)
    except ExitProcessedDataStatic.DoesNotExist:
        raise ValueError(f"No ExitProcessedDataStatic for token_info id={token_info_id}")
    static_dict = {c: getattr(static_obj, c) for c in static_columns if hasattr(static_obj, c)}
    static_row = pd.DataFrame([static_dict])
    for col in static_columns:
        if col not in static_row.columns:
            static_row[col] = 0.0
    static_row = static_row[static_columns]
    static_row = static_row.apply(pd.to_numeric, errors="coerce")
    static_row = static_row.replace([np.inf, -np.inf], np.nan)
    static_row = static_row.mask(static_row.abs() > 1e12)
    static_row = static_row.fillna(static_row.mean()).fillna(0.0)
    static_vec = static_row.iloc[0].to_numpy(dtype=np.float32, copy=False)

    static_scaler = meta.get("static_scaler") or {}
    static_mean = np.array(static_scaler.get("mean", [0.0] * len(static_columns)), dtype=np.float32)
    static_scale = np.array(static_scaler.get("scale", [1.0] * len(static_columns)), dtype=np.float32)
    if static_vec.size != static_mean.size or static_vec.size != static_scale.size:
        raise ValueError(f"Static feature shape mismatch: vec {static_vec.size}, mean {static_mean.size}")
    safe_static_scale = np.where(static_scale == 0.0, 1.0, static_scale)
    static_scaled = (static_vec - static_mean) / safe_static_scale

    # Model
    device = torch.device("cpu")
    model = SequenceStaticAttentionMIL(
        input_dim=len(feature_columns),
        static_dim=len(static_columns),
        hidden_dim=int(params.get("hidden_dim", 256)),
        embed_dim=int(params.get("embed_dim", 128)),
        attention_dim=int(params.get("attention_dim", 128)),
        dropout=float(params.get("dropout", 0.3)),
    ).to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    bag_tensor = torch.from_numpy(X_scaled).to(device)
    static_tensor = torch.from_numpy(static_scaled).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, attn_list = model([bag_tensor], static_tensor, return_attention=True)
        prob = torch.sigmoid(logits.squeeze(0)).item()
        attn_weights = attn_list[0].detach().cpu().numpy() if attn_list else np.array([])

    top_pos = _select_top(attn_weights, k=1)
    instance_record: Dict[str, object] = {"timestamp": None, "tx_hash": None, "feature_values": {}}
    if top_pos:
        pos = top_pos[0]
        ts = df_meta.loc[pos, "event_time"]
        if hasattr(ts, "isoformat"):
            ts = ts.isoformat()
            if ts.endswith("+00:00"):
                ts = ts.replace("+00:00", "Z")
        txh = df_meta.loc[pos, "tx_hash"]
        feat_vals = {}
        for name in INSTANCE_OUTPUT_FEATURES:
            if name in df_feat.columns:
                col_idx = df_feat.columns.get_loc(name)
                feat_vals[name] = float(X[pos, col_idx])
        instance_record = {"timestamp": ts, "tx_hash": txh, "feature_values": feat_vals}

    static_record = {}
    for name in STATIC_OUTPUT_FEATURES:
        if name in static_columns:
            idx = static_columns.index(name)
            if idx < static_vec.size:
                static_record[name] = float(static_vec[idx])
                continue
        static_record[name] = None

    # Flatten result to mirror run_mil_inference.py (base + top instance + static)
    result = {
        "token_addr": token_info.token_addr,
        "probability": float(prob),
        "tx_cnt": int(df_feat.shape[0]),
        "timestamp": instance_record.get("timestamp"),
        "tx_hash": instance_record.get("tx_hash"),
        **{k: instance_record["feature_values"].get(k) for k in INSTANCE_OUTPUT_FEATURES},
        **static_record,
    }

    return result
