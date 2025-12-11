"""
Detect Unformed LP (a.k.a. fake LP) using on-chain event data from DB.

Behavior is based on the former CSV script make_fake_lp_list_merge.py, but:
 - Reads from DB tables (TokenInfo, PairEvent) instead of CSVs.
 - Uses local price CSVs (price_eth.csv, price_btc.csv) in this module directory.
 - Dates newer than the price CSV coverage use the latest available price.
 - Returns a single-token result (DB contains exactly one TokenInfo at a time).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

# Token address constants (same as legacy script)
WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
WBTC = "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599"
STABLES = {
    "0xdac17f958d2ee523a2206206994597c13d831ec7",  # USDT
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",  # USDC
    "0x6b175474e89094c44da98b954eedeac495271d0f",  # DAI
    "0x853d955acef822db058eb8505911ed77f175b99e",  # FRAX
    "0x956f47f50a910163d8bf957cf5846d573e7f87ca",  # FEI
}
FIXED_PRICES = {
    "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984": 6.0,   # UNI
    "0xd533a949740bb3306d119cc777fa900ba034cd52": 0.5,   # CRV
    "0xe41d2489571d322189246dafa5ebde1f4699f498": 0.2,   # ZRX
    # SHIB intentionally ignored
}


def setup_django(settings_module: str = "config.settings") -> None:
    """Initialize Django if not already configured."""
    if "DJANGO_SETTINGS_MODULE" not in os.environ:
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", settings_module)
    import django
    from django.apps import apps

    if not apps.ready:
        django.setup()


def _load_price_series(path: Path) -> pd.Series:
    """Load price CSV with columns date,usd → Series indexed by date (naive)."""
    df = pd.read_csv(path)
    if "date" not in df.columns or "usd" not in df.columns:
        raise ValueError(f"Price CSV must have 'date' and 'usd' columns: {path}")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values("date")
    series = df.set_index("date")["usd"].astype(float)
    # normalize index to naive timestamps for comparison
    series.index = series.index.tz_localize(None)
    return series


def _price_lookup(addr: str, ts, price_eth: pd.Series, price_btc: pd.Series) -> Optional[float]:
    """
    Resolve USD price for a quote token at timestamp ts.
    If ts is newer than available data, use the latest available price.
    """
    addr = (addr or "").lower()
    if ts is None or pd.isna(ts):
        return None
    dt = pd.to_datetime(ts)
    if dt.tzinfo is not None:
        dt = dt.tz_convert(None)
    dt = dt.normalize()

    def _lookup(series: pd.Series) -> Optional[float]:
        if series.empty:
            return None
        if dt <= series.index[-1]:
            idx = series.index.get_indexer([dt], method="pad")
            if idx.size == 0 or idx[0] == -1:
                return None
            return float(series.iloc[idx[0]])
        # dt is newer than data → use latest available
        return float(series.iloc[-1])

    if addr == WETH:
        return _lookup(price_eth)
    if addr == WBTC:
        return _lookup(price_btc)
    if addr in STABLES:
        return 1.0
    if addr in FIXED_PRICES:
        return float(FIXED_PRICES[addr])
    return None


def _parse_evt_log(raw) -> Dict:
    """Parse evt_log which may be dict or JSON-string; fallback to empty dict."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            try:
                return eval(raw)
            except Exception:
                return {}
    return {}


def detect_unformed_lp(
    usd_threshold: float = 2000.0,
    swap_threshold: int = 2,
    price_eth_path: Optional[Path] = None,
    price_btc_path: Optional[Path] = None,
) -> Dict[str, object]:
    """
    Determine whether the single TokenInfo in DB is an Unformed LP.

    Returns dict:
        {
            "token_info": TokenInfo instance,
            "token_id": int,
            "token_addr": str,
            "is_unformed_lp": bool,
            "metrics": {...}
        }
    """
    setup_django()
    from api.models import TokenInfo, PairEvent

    token_qs = TokenInfo.objects.order_by("id")
    count = token_qs.count()
    if count == 0:
        raise RuntimeError("No TokenInfo rows found.")
    if count > 1:
        raise RuntimeError("Multiple TokenInfo rows found; expected exactly one.")
    token = token_qs.first()
    token_addr = (token.token_addr or "").lower()

    module_dir = Path(__file__).resolve().parent
    price_eth_series = _load_price_series(price_eth_path or (module_dir / "price_eth.csv"))
    price_btc_series = _load_price_series(price_btc_path or (module_dir / "price_btc.csv"))

    # Pass 1: pick up paircreated token0/token1 and creation ts
    token0 = None
    token1 = None
    pair_create_ts = None

    events_qs = (
        PairEvent.objects.filter(token_info=token)
        .order_by("timestamp")
        .only("evt_type", "evt_log", "timestamp", "tx_from")
    )
    for evt in events_qs:
        evt_type = (evt.evt_type or "").lower()
        if evt_type == "paircreated":
            data = _parse_evt_log(evt.evt_log)
            token0 = (str(data.get("token0", "")).lower() or None)
            token1 = (str(data.get("token1", "")).lower() or None)
            pair_create_ts = evt.timestamp
            break

    if token.pair_idx in (0, 1):
        base_is_token0 = token.pair_idx == 0
    else:
        if token_addr and token0 and token_addr == token0:
            base_is_token0 = True
        elif token_addr and token1 and token_addr == token1:
            base_is_token0 = False
        else:
            base_is_token0 = True

    quote_addr = token1 if base_is_token0 else token0
    ts_for_price = pair_create_ts or token.lp_create_ts
    price = _price_lookup(quote_addr, ts_for_price, price_eth_series, price_btc_series) if quote_addr else None

    # If quote token or price is unavailable, we cannot judge → treat as not unformed.
    if quote_addr is None or price is None:
        return {
            "token_info": token,
            "token_id": token.id,
            "token_addr": token.token_addr,
            "is_unformed_lp": False,
            "metrics": {
                "reason": "quote_or_price_missing",
                "quote_addr": quote_addr,
                "price_used": price,
            },
        }

    mint_inflow_usd = 0.0
    reserve_max_usd = 0.0
    swap_senders = set()

    for evt in events_qs.iterator():
        evt_type = (evt.evt_type or "").lower()
        evt_log = _parse_evt_log(evt.evt_log)

        if evt_type == "swap" and evt.tx_from:
            swap_senders.add(str(evt.tx_from).lower())

        # If quote/price unavailable, skip USD-based metrics
        if quote_addr is None or price is None:
            continue

        if evt_type == "mint":
            amt0 = evt_log.get("amount0")
            amt1 = evt_log.get("amount1")
            try:
                amt0f = float(amt0)
            except Exception:
                amt0f = None
            try:
                amt1f = float(amt1)
            except Exception:
                amt1f = None
            quote_amt = amt1f if base_is_token0 else amt0f
            if quote_amt is not None:
                mint_inflow_usd += quote_amt * price
        elif evt_type == "sync":
            r0 = evt_log.get("reserve0") or evt_log.get("reserve0New") or evt_log.get("reserve0new")
            r1 = evt_log.get("reserve1") or evt_log.get("reserve1New") or evt_log.get("reserve1new")
            try:
                r0f = float(r0)
            except Exception:
                r0f = None
            try:
                r1f = float(r1)
            except Exception:
                r1f = None
            quote_reserve = r1f if base_is_token0 else r0f
            if quote_reserve is not None:
                usd_val = quote_reserve * price
                if usd_val > reserve_max_usd:
                    reserve_max_usd = usd_val

    unique_swaps = len(swap_senders)
    usd_flag = (mint_inflow_usd < usd_threshold) or (reserve_max_usd < usd_threshold)
    swap_flag = unique_swaps <= swap_threshold
    is_unformed_lp = usd_flag or swap_flag

    return {
        "token_info": token,
        "token_id": token.id,
        "token_addr": token.token_addr,
        "is_unformed_lp": is_unformed_lp,
        "metrics": {
            "mint_inflow_usd": mint_inflow_usd,
            "reserve_max_usd": reserve_max_usd,
            "unique_swap_senders": unique_swaps,
            "usd_flag": usd_flag,
            "swap_flag": swap_flag,
            "quote_addr": quote_addr,
            "price_used": price,
            "base_is_token0": base_is_token0,
        },
    }
