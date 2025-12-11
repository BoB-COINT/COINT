"""
Honeypot feature generation module (pure feature logic only).

- compute_holder_features(holders_df)
- process_token(token_addr, owner_addr, pair_evt_df, holders_df)
- compute_honeypot_features(token_addr, owner_addr, pair_evt_records, holder_records)

이 모듈은 Django/DB를 전혀 알지 못하고,
순수하게 pair_evt + holders 기반 피처 계산만 담당한다.
"""

import ast
from typing import Iterable, Mapping, Dict, Any

import numpy as np
import pandas as pd
from scipy.stats import variation


# =========================================
# 1. 홀더 피처 계산
#    (비활성 토큰 구분용 홀더 기반 추가 피처 포함)
# =========================================
def compute_holder_features(holders_df: pd.DataFrame, total_holders_override: int | None = None, ) -> Dict[str, Any]:
    """
    holders_df: 한 토큰에 대한 홀더 정보 DataFrame
        - rel_to_total: 각 홀더의 지분 비율(%) 컬럼 사용
    """
    # holders 정보가 전혀 없는 경우
    if holders_df is None or len(holders_df) == 0:
        total_holders = int(total_holders_override or 0)

        return {
            "gini_coefficient": 0.0,
            "total_holders": total_holders,
            "whale_count": 0,
            "whale_total_pct": 0.0,
            "small_holders_pct": 0.0,
            "holder_balance_std": 0.0,
            "holder_balance_cv": 0.0,
            "hhi_index": 0.0,
            "whale_domination_ratio": 0.0,
            "whale_presence_flag": 0,
            "few_holders_flag": int(total_holders <= 3),
            "airdrop_like_flag": 0,
            "concentrated_large_community_score": 0.0,
            "hhi_per_holder": 0.0,
            "whale_but_no_small_flag": 0,
        }

    # rel_to_total(%) → 0~1 비율로 변환
    balances = holders_df["rel_to_total"].astype(float).values
    balances_norm = balances / 100.0  # 0~1

    # total_holders: 실제 홀더 수
    base_total_holders = len(balances_norm)
    total_holders = int(total_holders_override) if total_holders_override is not None else int(base_total_holders)

    # 전체 합이 0인 경우(이론상 거의 없음) 방어 코드
    total_sum = balances_norm.sum()
    if total_sum <= 0:
        balances_norm = np.full_like(balances_norm, 1.0 / total_holders)
        total_sum = 1.0

    # '고래' 기준: 5% 이상 보유
    whale_mask = balances_norm >= 0.05
    whale_count = int(whale_mask.sum())

    # % 단위로 다시 변환 (0~100)
    whale_total_pct = float(balances_norm[whale_mask].sum() * 100.0)
    small_holders_pct = float(balances_norm[~whale_mask].sum() * 100.0)

    # Gini 계수
    sorted_bal = np.sort(balances_norm)
    n = len(sorted_bal)
    gini = float(
        (2 * np.sum((np.arange(1, n + 1) * sorted_bal)) / (n * np.sum(sorted_bal)))
        - ((n + 1) / n)
    )

    # HHI (Herfindahl-Hirschman Index)
    hhi_index = float(np.sum(balances_norm**2))

    # 표준편차 / CV
    holder_balance_std = float(np.std(balances_norm))
    holder_balance_cv = float(variation(balances_norm)) if np.mean(balances_norm) > 0 else 0.0

    # ===== 추가 홀더 기반 피처들 =====
    eps = 1e-6

    # 1) 고래 지배도 비율: 고래 지분 / 소액홀더 지분
    whale_domination_ratio = float((whale_total_pct + eps) / (small_holders_pct + eps))

    # 2) 고래 존재 여부 플래그
    whale_presence_flag = int(whale_count > 0)

    # 3) 소수 홀더 토큰 플래그 (ex. total_holders <= 3)
    few_holders_flag = int(total_holders <= 3)

    # 4) 에어드롭 느낌 플래그
    airdrop_like_flag = int(
        (total_holders >= 20)
        and (holder_balance_cv <= 1.0)
        and (small_holders_pct >= 90.0)
    )

    # 5) "홀더도 어느 정도 있는데, 집중도가 높은" 점수
    concentrated_large_community_score = float(hhi_index * np.log1p(total_holders))

    # 6) HHI를 홀더 수로 나눈 정규화 버전
    hhi_per_holder = float(hhi_index / (total_holders + eps))

    # 7) 고래만 있고 개미는 거의 없는 토큰 플래그
    whale_but_no_small_flag = int(
        (whale_total_pct > 90.0) and (small_holders_pct < 10.0)
    )

    return {
        "gini_coefficient": gini,
        "total_holders": int(total_holders),
        "whale_count": whale_count,
        "whale_total_pct": whale_total_pct,
        "small_holders_pct": small_holders_pct,
        "holder_balance_std": holder_balance_std,
        "holder_balance_cv": holder_balance_cv,
        "hhi_index": hhi_index,
        "whale_domination_ratio": whale_domination_ratio,
        "whale_presence_flag": whale_presence_flag,
        "few_holders_flag": few_holders_flag,
        "airdrop_like_flag": airdrop_like_flag,
        "concentrated_large_community_score": concentrated_large_community_score,
        "hhi_per_holder": hhi_per_holder,
        "whale_but_no_small_flag": whale_but_no_small_flag,
    }


# =========================================
# 2. 한 토큰에 대한 pair_evt + holders 기반 피처 계산
# =========================================
def process_token(
    token_addr: str,
    owner_addr: str,
    pair_evt_df: pd.DataFrame,
    holders_df: pd.DataFrame,
    total_holders_override: int | None = None,  
) -> Dict[str, Any]:
    """
    token_addr: 토큰 주소 (string)
    owner_addr: 토큰 생성자 주소 (string, 없으면 "")
    pair_evt_df: 해당 토큰의 pair_evt DataFrame
                 (timestamp, evt_type, tx_from, tx_to, evt_log ...)
    holders_df: 해당 토큰의 holder_info DataFrame
                 (holder_addr, balance, rel_to_total ...)
    """
    if pair_evt_df is None:
        pair_evt_df = pd.DataFrame([])

    pair_evt = pair_evt_df.copy()
    if not pair_evt.empty and "timestamp" in pair_evt.columns:
        pair_evt = pair_evt.sort_values("timestamp")

    owner = (owner_addr or "").lower()

    buys, sells = [], []
    buyers, sellers = set(), set()

    total_buy_vol = 0.0
    total_sell_vol = 0.0
    total_owner_sell_vol = 0.0
    mint_count = 0
    burn_count = 0
    windows_with_activity = 0

    consecutive_sell_windows = 0
    total_sell_block_windows = 0
    last_was_sell = False

    total_windows = len(pair_evt)

    # evt_log는 JSONField 또는 str일 수 있음
    for _, row in pair_evt.iterrows():
        evt_type = row.get("evt_type", "")

        raw_log = row.get("evt_log", {})
        log = {}
        if isinstance(raw_log, dict):
            log = raw_log
        elif isinstance(raw_log, str):
            raw_log = raw_log.strip()
            if raw_log:
                try:
                    # CSV 시절 문자열 형태를 쓰던 경우 대비
                    log = ast.literal_eval(raw_log)
                    if not isinstance(log, dict):
                        log = {}
                except Exception:
                    log = {}
        else:
            log = {}

        if evt_type == "Mint":
            mint_count += 1

        if evt_type == "Burn":
            burn_count += 1

        if evt_type == "Swap":
            amount0_in = float(log.get("amount0In", 0) or 0)
            amount0_out = float(log.get("amount0Out", 0) or 0)

            # sell
            if amount0_in > 0:
                sells.append(amount0_in)
                tx_from = row.get("tx_from") or ""
                if isinstance(tx_from, str) and tx_from:
                    sellers.add(tx_from.lower())
                total_sell_vol += amount0_in

                if isinstance(tx_from, str) and tx_from.lower() == owner:
                    total_owner_sell_vol += amount0_in

                if last_was_sell:
                    consecutive_sell_windows += 1

                last_was_sell = True
                total_sell_block_windows += 1

            # buy
            elif amount0_out > 0:
                buys.append(amount0_out)
                tx_to = row.get("tx_to") or ""
                if isinstance(tx_to, str) and tx_to:
                    buyers.add(tx_to.lower())
                total_buy_vol += amount0_out
                last_was_sell = False

            windows_with_activity += 1

    total_buy_cnt = len(buys)
    total_sell_cnt = len(sells)

    if total_buy_cnt + total_sell_cnt > 0:
        imbalance_rate = (total_buy_cnt - total_sell_cnt) / (total_buy_cnt + total_sell_cnt)
    else:
        imbalance_rate = 0.0

    if len(sells) > 0 and sum(sells) > 0:
        max_sell_share = max(sells) / sum(sells)
    else:
        max_sell_share = 0.0

    # 홀더 피처 계산
    holder_features = compute_holder_features(
        holders_df,
        total_holders_override=total_holders_override,
    )

    inactive_token_flag = int(
        (total_buy_cnt == 0) and (total_sell_cnt == 0) and (windows_with_activity == 0)
    )

    total_owner_sell_cnt = sum(1 for s in sellers if s == owner)
    total_non_owner_sell_cnt = total_sell_cnt - total_owner_sell_cnt

    s_owner_count = sum(1 for b in buyers if b == owner)

    return {
        "token_addr": token_addr,
        "total_buy_cnt": int(total_buy_cnt),
        "total_sell_cnt": int(total_sell_cnt),
        "total_owner_sell_cnt": int(total_owner_sell_cnt),
        "total_non_owner_sell_cnt": int(total_non_owner_sell_cnt),
        "imbalance_rate": float(imbalance_rate),
        "total_windows": int(total_windows),
        "windows_with_activity": int(windows_with_activity),
        "total_burn_events": int(burn_count),
        "total_mint_events": int(mint_count),
        "s_owner_count": int(s_owner_count),
        "total_sell_vol": float(total_sell_vol),
        "total_buy_vol": float(total_buy_vol),
        "total_owner_sell_vol": float(total_owner_sell_vol),
        "total_sell_vol_log": float(np.log1p(total_sell_vol)),
        "total_buy_vol_log": float(np.log1p(total_buy_vol)),
        "total_owner_sell_vol_log": float(np.log1p(total_owner_sell_vol)),
        "liquidity_event_mask": int(1 if (mint_count + burn_count) > 0 else 0),
        "max_sell_share": float(max_sell_share),
        "unique_sellers": int(len(sellers)),
        "unique_buyers": int(len(buyers)),
        "consecutive_sell_block_windows": int(consecutive_sell_windows),
        "total_sell_block_windows": int(total_sell_block_windows),
        "inactive_token_flag": int(inactive_token_flag),
        **holder_features,
    }


# =========================================
# 3. 어댑터 / 서비스에서 쓰기 좋은 래퍼
#    (records → DataFrame → process_token)
# =========================================
def compute_honeypot_features(
    token_addr: str,
    owner_addr: str,
    pair_evt_records: Iterable[Mapping[str, Any]],
    holder_records: Iterable[Mapping[str, Any]],
    holder_cnt: int | None = None,
) -> Dict[str, Any]:
    # ✅ QuerySet truthiness 피하려고 None 만 체크
    pair_evt_list = list(pair_evt_records) if pair_evt_records is not None else []
    holder_list = list(holder_records) if holder_records is not None else []

    pair_evt_df = pd.DataFrame.from_records(pair_evt_list)
    holders_df = pd.DataFrame.from_records(holder_list)

    return process_token(
        token_addr=token_addr,
        owner_addr=owner_addr,
        pair_evt_df=pair_evt_df,
        holders_df=holders_df,
        total_holders_override=holder_cnt,
    )
