"""
DB-backed implementation of make_exit_feat_v3_2.py.

Reads TokenInfo/PairEvent via ORM and produces per-event features matching
features_exit_mil_v3_2.csv (19 columns incl. masks) for bulk insert into
ExitProcessedDataInstance.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, getcontext
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Ensure adequate precision for on-chain numbers
getcontext().prec = 28


# Django bootstrap -----------------------------------------------------------
def setup_django(settings_module: str = "config.settings") -> None:
    if "DJANGO_SETTINGS_MODULE" not in os.environ:
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", settings_module)
    import django
    from django.apps import apps

    if not apps.ready:
        django.setup()


# Helpers --------------------------------------------------------------------
def to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return None
        try:
            return float(text)
        except Exception:
            return None
    try:
        return float(value)
    except Exception:
        return None


def float_or_zero(value: Optional[float]) -> float:
    return float(value) if value is not None else 0.0


def log1p_positive(value: Optional[float]) -> float:
    import math

    if value is None or value <= 0:
        return 0.0
    return float(math.log1p(float(value)))


def signed_log1p(value: Optional[float]) -> float:
    import math

    if value is None or value == 0:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    return float(math.log1p(abs(float(value))) * sign)


def safe_div(numer: Optional[float], denom: Optional[float]) -> Optional[float]:
    if numer is None or denom in (None, 0):
        return None
    try:
        return float(numer) / float(denom)
    except Exception:
        return None


# Data shapes ----------------------------------------------------------------
@dataclass
class PairEventRecord:
    timestamp: datetime
    block_number: Optional[int]
    tx_hash: Optional[str]
    evt_idx: Optional[int]
    evt_type: str
    lp_total_supply: Optional[Decimal]
    evt_log: Dict[str, object]


class ExitInstancePreprocessor:
    def __init__(self):
        from api.models import PairEvent

        self.PairEvent = PairEvent

    def _determine_base_is_token0(self, token_info, token0: Optional[str], token1: Optional[str]) -> bool:
        # Match make_exit_feat_v3_2.py: pair_idx 0/1이면 그대로 신뢰, 아니면 기본 True.
        if getattr(token_info, "pair_idx", None) in (0, 1):
            return token_info.pair_idx == 0
        return True

    def _fetch_records(self, token_info) -> List[PairEventRecord]:
        qs = self.PairEvent.objects.filter(token_info=token_info).only(
            "timestamp",
            "block_number",
            "tx_hash",
            "evt_idx",
            "evt_type",
            "lp_total_supply",
            "evt_log",
        )
        records = [
            PairEventRecord(
                timestamp=evt.timestamp,
                block_number=evt.block_number,
                tx_hash=evt.tx_hash,
                evt_idx=evt.evt_idx,
                evt_type=(evt.evt_type or "").lower(),
                lp_total_supply=to_float(evt.lp_total_supply),
                evt_log=evt.evt_log or {},
            )
            for evt in qs
        ]
        records.sort(
            key=lambda r: (
                r.timestamp,
                r.block_number if r.block_number is not None else float("inf"),
                r.tx_hash or "",
                r.evt_idx if r.evt_idx is not None else 0,
            )
        )
        return records

    def _group_by_tx(self, records: Sequence[PairEventRecord]) -> List[List[PairEventRecord]]:
        grouped: List[List[PairEventRecord]] = []
        current: List[PairEventRecord] = []
        current_key: Optional[Tuple[datetime, Optional[int], Optional[str]]] = None
        for record in records:
            key = (record.timestamp, record.block_number, record.tx_hash)
            if current_key is None or key != current_key:
                if current:
                    grouped.append(current)
                current = [record]
                current_key = key
            else:
                current.append(record)
        if current:
            grouped.append(current)
        return grouped

    def compute_exit_features(self, token_info) -> List[Dict[str, object]]:
        """
        Build per-event features from DB events for a single token.
        Matches make_exit_feat_v3_2.py outputs (log1p + masks, 19 cols).
        """
        records = self._fetch_records(token_info)
        if not records:
            return []

        token0 = None
        token1 = None
        for rec in records:
            if rec.evt_type == "paircreated":
                payload = rec.evt_log or {}
                token0 = payload.get("token0") or token0
                token1 = payload.get("token1") or token1
                break

        base_is_token0 = self._determine_base_is_token0(token_info, token0, token1)

        rows: List[Dict[str, object]] = []
        prev_ts: Optional[datetime] = None
        prev_lp: Optional[float] = None
        prev_base: Optional[float] = None
        prev_quote: Optional[float] = None
        prev_price: Optional[float] = None
        last_mint_ts: Optional[datetime] = None

        rq_series: List[Optional[float]] = []
        swap_series: List[int] = []
        mint_series: List[int] = []
        burn_series: List[int] = []
        grouped = self._group_by_tx(records)

        for local_id, group in enumerate(grouped):
            first_record = group[0]
            timestamp = first_record.timestamp.astimezone(timezone.utc)
            delta_t = Decimal("0")
            if prev_ts is not None:
                seconds = int((timestamp - prev_ts).total_seconds())
                if seconds < 0:
                    seconds = 0
                delta_t = Decimal(seconds)
            prev_ts = timestamp

            quote_before = prev_quote
            base_before = prev_base
            price_before = prev_price
            lp_before = prev_lp

            lp_current = None
            base_current = prev_base
            quote_current = prev_quote
            price_current = prev_price

            lp_minted_amount = 0.0
            lp_burned_amount = 0.0
            swap_base_in_total = 0.0
            swap_base_out_total = 0.0
            swap_quote_in_total = 0.0
            swap_quote_out_total = 0.0

            is_mint = False
            is_burn = False
            is_swap = False

            for record in group:
                evt_type = record.evt_type
                payload = record.evt_log or {}
                swap_base_in = 0.0
                swap_base_out = 0.0
                swap_quote_in = 0.0
                swap_quote_out = 0.0

                if evt_type == "mint":
                    is_mint = True
                elif evt_type == "burn":
                    is_burn = True
                elif evt_type == "swap":
                    is_swap = True

                if evt_type == "swap":
                    amount0_in = to_float(payload.get("amount0In"))
                    amount0_out = to_float(payload.get("amount0Out"))
                    amount1_in = to_float(payload.get("amount1In"))
                    amount1_out = to_float(payload.get("amount1Out"))
                    if base_is_token0:
                        swap_base_in = float_or_zero(amount0_in)
                        swap_base_out = float_or_zero(amount0_out)
                        swap_quote_in = float_or_zero(amount1_in)
                        swap_quote_out = float_or_zero(amount1_out)
                    else:
                        swap_base_in = float_or_zero(amount1_in)
                        swap_base_out = float_or_zero(amount1_out)
                        swap_quote_in = float_or_zero(amount0_in)
                        swap_quote_out = float_or_zero(amount0_out)
                elif evt_type == "sync":
                    reserve0 = to_float(payload.get("reserve0"))
                    reserve1 = to_float(payload.get("reserve1"))
                    if base_is_token0:
                        base_current = reserve0
                        quote_current = reserve1
                    else:
                        base_current = reserve1
                        quote_current = reserve0
                    if reserve0 not in (None, Decimal("0")) and reserve1 not in (None, Decimal("0")):
                        if base_is_token0:
                            price_current = safe_div(reserve0, reserve1)
                        else:
                            price_current = safe_div(reserve1, reserve0)

                swap_base_in_total += swap_base_in
                swap_base_out_total += swap_base_out
                swap_quote_in_total += swap_quote_in
                swap_quote_out_total += swap_quote_out

                lp_after_event = record.lp_total_supply if record.evt_type == "sync" else None
                if lp_before is not None and lp_after_event is not None:
                    delta_lp = lp_after_event - lp_before
                    if delta_lp > 0:
                        lp_minted_amount += delta_lp
                    elif delta_lp < 0:
                        lp_burned_amount += -delta_lp
                lp_current = lp_after_event if lp_after_event is not None else None

            quote_drop_frac = 0.0
            quote_delta = 0.0
            base_drop_frac = 0.0
            base_delta = 0.0
            if quote_before is not None and quote_current is not None:
                quote_delta = quote_current - quote_before
                if quote_before > 0:
                    drop = (quote_before - quote_current) / quote_before
                    if drop > 0:
                        quote_drop_frac = drop
            if base_before is not None and base_current is not None:
                base_delta = base_current - base_before
                if base_before > 0:
                    drop = (base_before - base_current) / base_before
                    if drop > 0:
                        base_drop_frac = drop

            price_ratio_delta: Optional[float] = None
            if price_current is not None and price_before is not None:
                price_ratio_delta = price_current - price_before
            elif price_current is not None:
                price_ratio_delta = price_current

            time_since_last_mint: Optional[float] = None
            if last_mint_ts is not None:
                delta_sec = int((timestamp - last_mint_ts).total_seconds())
                if delta_sec < 0:
                    delta_sec = 0
                time_since_last_mint = float(delta_sec)
            if is_mint:
                time_since_last_mint = 0.0
                last_mint_ts = timestamp

            divisor = delta_t if delta_t > 0 else None

            def per_sec(value: Optional[float]) -> Optional[float]:
                if value is None:
                    return None
                if divisor is None or divisor == 0:
                    return value
                return float(value) / float(divisor)

            lp_delta_per_sec = per_sec(lp_current - lp_before if lp_current is not None and lp_before is not None else None)
            quote_delta_per_sec = per_sec(quote_delta)
            base_delta_per_sec = per_sec(base_delta)
            lp_minted_per_sec = per_sec(lp_minted_amount)
            lp_burned_per_sec = per_sec(lp_burned_amount)

            row: Dict[str, object] = {
                "local_id": local_id,
                "token_addr_idx": token_info.id,
                "event_time": timestamp,
                "tx_hash": first_record.tx_hash or "",
                "delta_t_sec": log1p_positive(delta_t),
                "is_mint_event": int(is_mint),
                "is_burn_event": int(is_burn),
                "is_swap_event": int(is_swap),
                # raw values (before masking/log1p) kept as Decimal; masks added later
                "lp_total_supply": lp_current,
                "reserve_base_drop_frac": float(base_drop_frac) if base_drop_frac is not None else None,
                "reserve_quote": quote_current,
                "reserve_quote_drop_frac": float(quote_drop_frac) if quote_drop_frac is not None else None,
                "price_ratio": price_current,
                "time_since_last_mint_sec": time_since_last_mint,
                "price_ratio_delta_per_sec": signed_log1p(per_sec(price_ratio_delta)),
                "lp_supply_delta_per_sec": signed_log1p(lp_delta_per_sec),
                "reserve_quote_delta_per_sec": signed_log1p(quote_delta_per_sec),
                "lp_minted_amount_per_sec": lp_minted_per_sec,
                "lp_burned_amount_per_sec": lp_burned_per_sec,
                "swap_base_in_amount_per_sec": per_sec(swap_base_in_total),
                "swap_base_out_amount_per_sec": per_sec(swap_base_out_total),
                "swap_quote_in_amount_per_sec": per_sec(swap_quote_in_total),
                "swap_quote_out_amount_per_sec": per_sec(swap_quote_out_total),
            }
            rows.append(row)

            prev_lp = lp_current
            prev_base = base_current
            prev_quote = quote_current
            prev_price = price_current if price_current is not None else prev_price

            rq_series.append(quote_current)
            swap_series.append(1 if is_swap else 0)
            mint_series.append(1 if is_mint else 0)
            burn_series.append(1 if is_burn else 0)

        # Token-level additions
        rq_values = [float(x) for x in rq_series if x is not None]
        drawdown_val = 0.0
        if rq_values:
            max_rq = max(rq_values)
            final_rq = next((float(x) for x in reversed(rq_series) if x is not None), None)
            if max_rq > 0 and final_rq is not None:
                drawdown_val = max(0.0, min(1.0, 1.0 - (final_rq / max_rq)))

        def last_n_ratio(series: List[int], n: int) -> Optional[float]:
            tail = series[-n:]
            if not tail:
                return None
            return sum(tail) / len(tail)

        recent_mint10 = last_n_ratio(mint_series, 10)
        recent_mint20 = last_n_ratio(mint_series, 20)
        recent_burn10 = last_n_ratio(burn_series, 10)
        recent_burn20 = last_n_ratio(burn_series, 20)

        for row in rows:
            row["recent_mint_ratio_last10"] = recent_mint10
            row["recent_mint_ratio_last20"] = recent_mint20
            row["recent_burn_ratio_last10"] = recent_burn10
            row["recent_burn_ratio_last20"] = recent_burn20
            row["reserve_quote_drawdown"] = drawdown_val

            # Mask columns as in v3_2
            def add_mask(field: str, mask: str, transform=None):
                val = row.get(field)
                is_missing = val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val)))
                if is_missing:
                    row[field] = None if transform else None
                    row[mask] = 0.0
                else:
                    row[field] = transform(val) if transform else val
                    row[mask] = 1.0

            add_mask("lp_total_supply", "lp_total_supply_mask", lambda x: log1p_positive(x))
            add_mask("reserve_quote", "reserve_quote_mask", lambda x: log1p_positive(x))
            add_mask("price_ratio", "price_ratio_mask", lambda x: log1p_positive(x))
            add_mask("time_since_last_mint_sec", "time_since_last_mint_sec_mask", lambda x: log1p_positive(x))

            # Per-second/log1p fields: if None, set to 0.0 (as in original CSV process)
            for f in [
                "lp_minted_amount_per_sec",
                "lp_burned_amount_per_sec",
                "swap_base_in_amount_per_sec",
                "swap_base_out_amount_per_sec",
                "swap_quote_in_amount_per_sec",
                "swap_quote_out_amount_per_sec",
            ]:
                if row.get(f) is None:
                    row[f] = None
                else:
                    row[f] = log1p_positive(Decimal(row[f]) if not isinstance(row[f], Decimal) else row[f])

            # price_ratio_delta_per_sec, lp_supply_delta_per_sec, reserve_quote_delta_per_sec: already signed_log1p
            for f in ["price_ratio_delta_per_sec", "lp_supply_delta_per_sec", "reserve_quote_delta_per_sec"]:
                if row.get(f) is None:
                    row[f] = 0.0

            # is_swap_event to int (0/1)
            row["is_swap_event"] = int(row.get("is_swap_event", 0))


        return rows

if __name__ == "__main__":
    raise SystemExit("Run this via exit_test.py or integrate via PreprocessorAdapter.")
