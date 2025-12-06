"""
DB-backed computation of static exit features (v4) without CSV.

Computes the same columns as features_exit_static_v4.csv and saves to
ExitProcessedDataStatic.
"""

from __future__ import annotations

import os
from collections import deque
from datetime import datetime, timezone
from math import log
from typing import Dict, List, Optional, Tuple

import numpy as np


def setup_django(settings_module: str = "config.settings") -> None:
    if "DJANGO_SETTINGS_MODULE" not in os.environ:
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", settings_module)
    import django
    from django.apps import apps

    if not apps.ready:
        django.setup()


def _parse_reserves(evt_log: Dict[str, object]) -> Tuple[Optional[float], Optional[float]]:
    """Extract reserve0/reserve1 (or reserve0New/1New) as floats."""
    r0 = evt_log.get("reserve0")
    r1 = evt_log.get("reserve1")
    if r0 is None and r1 is None:
        r0 = evt_log.get("reserve0New") or evt_log.get("reserve0new")
        r1 = evt_log.get("reserve1New") or evt_log.get("reserve1new")
    try:
        r0f = float(r0) if r0 is not None else None
        r1f = float(r1) if r1 is not None else None
    except Exception:
        return None, None
    return r0f, r1f


def _realized_vol(values: List[float]) -> float:
    arr = np.array(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0]
    if arr.size <= 1:
        return 0.0
    log_vals = np.log(arr)
    diffs = np.diff(log_vals)
    if diffs.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(diffs))))


def _range(values: List[float]) -> float:
    arr = np.array(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(arr.max() - arr.min())


class ExitStaticPreprocessor:
    def __init__(self):
        from api.models import PairEvent

        self.PairEvent = PairEvent

    def _determine_base_is_token0(self, token_info) -> bool:
        if getattr(token_info, "pair_idx", None) in (0, 1):
            return token_info.pair_idx == 0
        return True

    def compute_static_features(self, token_info) -> Dict[str, float]:
        """Compute static exit features for a single token from DB events."""
        events = (
            self.PairEvent.objects.filter(token_info=token_info)
            .order_by("timestamp")
            .only("evt_type", "evt_log", "timestamp")
        )

        swap_count = 0
        burn_count = 0
        total_count = 0
        last5 = deque(maxlen=5)
        last_ts: Optional[datetime] = None

        price_series: List[float] = []
        reserve_quote_series: List[float] = []
        max_rq = 0.0
        final_rq: Optional[float] = None

        token0 = None
        token1 = None

        # Pick up token0/token1 from PairCreated if present
        has_paircreated = False
        for evt in events:
            evt_type = (evt.evt_type or "").lower()
            if evt_type == "paircreated":
                payload = evt.evt_log or {}
                token0 = (payload.get("token0") or "").lower() or None
                token1 = (payload.get("token1") or "").lower() or None
                has_paircreated = True
                break

        base_is_token0 = self._determine_base_is_token0(token_info)
        if token_info.token_addr and token0 and token1:
            if token0 == token_info.token_addr.lower():
                base_is_token0 = True
            elif token1 == token_info.token_addr.lower():
                base_is_token0 = False

        # Iterate events again for metrics
        for evt in events:
            evt_type = (evt.evt_type or "").lower()
            total_count += 1
            if evt_type == "swap":
                swap_count += 1
            if evt_type == "burn":
                burn_count += 1

            last5.append(evt_type)

            ts = evt.timestamp
            if ts and (last_ts is None or ts > last_ts):
                last_ts = ts

            r0, r1 = _parse_reserves(evt.evt_log or {})
            if r0 is None and r1 is None:
                continue
            if r0 is None:
                r0 = 0.0
            if r1 is None:
                r1 = 0.0

            if base_is_token0:
                base_reserve, quote_reserve = r0, r1
            else:
                base_reserve, quote_reserve = r1, r0

            if quote_reserve > 0 and base_reserve >= 0:
                price = base_reserve / quote_reserve
                price_series.append(price)
            reserve_quote_series.append(max(quote_reserve, 0.0))

            if quote_reserve > max_rq:
                max_rq = quote_reserve
            final_rq = quote_reserve

        swap_share = float(swap_count / total_count) if total_count > 0 else 0.0
        swaps_last5 = float(sum(1 for e in last5 if e == "swap") / len(last5)) if last5 else 0.0
        burn_ratio_all = float(burn_count / total_count) if total_count > 0 else 0.0

        # If PairCreated event was absent in DB, account for it to align counts with CSV
        if not has_paircreated:
            total_count += 1
            swap_share = float(swap_count / total_count) if total_count > 0 else 0.0
            burn_ratio_all = float(burn_count / total_count) if total_count > 0 else 0.0

        price_ratio_realized_vol = _realized_vol(price_series)
        price_ratio_range = _range(price_series)
        reserve_quote_realized_vol = _realized_vol(reserve_quote_series)

        drawdown = 0.0
        if max_rq > 0 and final_rq is not None:
            drawdown = max(0.0, min(1.0, 1.0 - (final_rq / max_rq)))

        liquidity_age_days = 0.0
        if last_ts and token_info.lp_create_ts:
            liquidity_age_days = max((last_ts - token_info.lp_create_ts).total_seconds() / 86400.0, 0.0)

        holder_cnt = float(token_info.holder_cnt) if token_info.holder_cnt is not None else 0.0

        return {
            "price_ratio_realized_vol": price_ratio_realized_vol,
            "price_ratio_range": price_ratio_range,
            "reserve_quote_realized_vol": reserve_quote_realized_vol,
            "burn_ratio_all": burn_ratio_all,
            "reserve_quote_drawdown_global": drawdown,
            "swap_share": swap_share,
            "swaps_last5": swaps_last5,
            "liquidity_age_days": liquidity_age_days,
            "holder_cnt": holder_cnt,
        }
