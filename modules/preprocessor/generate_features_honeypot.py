import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import variation
from pathlib import Path
import ast

# =========================================
# 0. Django 설정 로딩 (프로젝트 루트 추가)
# =========================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # .../COINT
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django  # noqa: E402

django.setup()

from api.models import (  # noqa: E402
    TokenInfo,
    PairEvent,
    HolderInfo,
    HoneypotDaResult,
    HoneypotProcessedData,
)

BASE = Path(".")


# =========================================
# 1. 홀더 피처 계산
#    (비활성 토큰 구분용 홀더 기반 추가 피처 포함)
# =========================================
def compute_holder_features(holders_df):
    """
    holders_df: 한 토큰에 대한 홀더 정보 DataFrame
        - rel_to_total: 각 홀더의 지분 비율(%) 컬럼 사용
    """
    # holders 정보가 전혀 없는 경우
    if holders_df is None or len(holders_df) == 0:
        total_holders = 0
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
            "few_holders_flag": 1,  # 홀더가 없으니 사실상 극단적인 소수
            "airdrop_like_flag": 0,
            "concentrated_large_community_score": 0.0,
            "hhi_per_holder": 0.0,
            "whale_but_no_small_flag": 0,
        }

    # rel_to_total(%) → 0~1 비율로 변환
    balances = holders_df["rel_to_total"].astype(float).values
    balances_norm = balances / 100.0  # 0~1

    # total_holders: 실제 홀더 수
    total_holders = len(balances_norm)

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
def process_token(token_addr: str, owner_addr: str, pair_evt_df: pd.DataFrame, holders_df: pd.DataFrame):
    """
    token_addr: 토큰 주소 (string)
    owner_addr: 토큰 생성자 주소 (string, 없으면 "")
    pair_evt_df: 해당 토큰의 pair_evt DataFrame
    holders_df: 해당 토큰의 holder_info DataFrame
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
    holder_features = compute_holder_features(holders_df)

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
# 3. 메인 루프: DB → 피처 계산 → honeypot_processed_data 저장
# =========================================
def main():
    start = time.time()

    token_qs = TokenInfo.objects.all().order_by("id")
    total_tokens = token_qs.count()
    da_count = HoneypotDaResult.objects.count()
    pair_evt_count = PairEvent.objects.count()
    holder_count = HolderInfo.objects.count()

    print("=================================================")
    print("🚀 Honeypot Feature Generator - Django DB version")
    print("=================================================")
    print(f"  TokenInfo           : {total_tokens:,}")
    print(f"  PairEvent           : {pair_evt_count:,}")
    print(f"  HolderInfo          : {holder_count:,}")
    print(f"  HoneypotDaResult    : {da_count:,}")
    print("  Output Model        : HoneypotProcessedData")
    print("-------------------------------------------------")

    for i, token in enumerate(token_qs.iterator(), start=1):
        token_addr = token.token_addr
        owner_addr = token.token_creator_addr or ""

        # 1) pair_evt → DataFrame
        evt_qs = PairEvent.objects.filter(token_info=token).values(
            "timestamp", "evt_type", "tx_from", "tx_to", "evt_log"
        )
        pair_evt_df = pd.DataFrame.from_records(evt_qs)

        # 2) holder_info → DataFrame
        holder_qs = HolderInfo.objects.filter(token_info=token).values(
            "holder_addr", "balance", "rel_to_total"
        )
        holders_df = pd.DataFrame.from_records(holder_qs)

        # 3) 정적/홀더 피처 계산
        row = process_token(
            token_addr=token_addr,
            owner_addr=owner_addr,
            pair_evt_df=pair_evt_df,
            holders_df=holders_df,
        )

        # 3-1) 🔹 동적 분석 결과(HoneypotDaResult)를 HoneypotProcessedData 컬럼으로 매핑
        da = HoneypotDaResult.objects.filter(token_info=token).first()
        dyn_defaults = {}
        if da:
            dyn_defaults = {
                "balance_manipulation": int(bool(da.balance_manipulation_result)),
                "buy_1": int(bool(da.buy_1)),
                "buy_2": int(bool(da.buy_2)),
                "buy_3": int(bool(da.buy_3)),
                "existing_holders_check": int(bool(da.existing_holders_result)),
                "exterior_call_check": int(bool(da.exterior_call_result)),
                "sell_fail_type_1": int(da.sell_fail_type_1),
                "sell_fail_type_2": int(da.sell_fail_type_2),
                "sell_fail_type_3": int(da.sell_fail_type_3),
                "sell_result_1": int(bool(da.sell_1)),
                "sell_result_2": int(bool(da.sell_2)),
                "sell_result_3": int(bool(da.sell_3)),
                "tax_manipulation": int(bool(da.tax_manipulation_result)),
                "trading_suspend_check": int(bool(da.trading_suspend_result)),
                "unlimited_mint": int(bool(da.unlimited_mint_result)),
            }

        # 4) HoneypotProcessedData upsert (정적 + 동적 피처 모두 포함)
        defaults = dict(
            token_addr=row["token_addr"],
            total_buy_cnt=row["total_buy_cnt"],
            total_sell_cnt=row["total_sell_cnt"],
            total_owner_sell_cnt=row["total_owner_sell_cnt"],
            total_non_owner_sell_cnt=row["total_non_owner_sell_cnt"],
            imbalance_rate=row["imbalance_rate"],
            total_windows=row["total_windows"],
            windows_with_activity=row["windows_with_activity"],
            total_burn_events=row["total_burn_events"],
            total_mint_events=row["total_mint_events"],
            s_owner_count=row["s_owner_count"],
            total_sell_vol=row["total_sell_vol"],
            total_buy_vol=row["total_buy_vol"],
            total_owner_sell_vol=row["total_owner_sell_vol"],
            total_sell_vol_log=row["total_sell_vol_log"],
            total_buy_vol_log=row["total_buy_vol_log"],
            total_owner_sell_vol_log=row["total_owner_sell_vol_log"],
            liquidity_event_mask=row["liquidity_event_mask"],
            max_sell_share=row["max_sell_share"],
            unique_sellers=row["unique_sellers"],
            unique_buyers=row["unique_buyers"],
            consecutive_sell_block_windows=row["consecutive_sell_block_windows"],
            total_sell_block_windows=row["total_sell_block_windows"],
            gini_coefficient=row["gini_coefficient"],
            total_holders=row["total_holders"],
            whale_count=row["whale_count"],
            whale_total_pct=row["whale_total_pct"],
            small_holders_pct=row["small_holders_pct"],
            holder_balance_std=row["holder_balance_std"],
            holder_balance_cv=row["holder_balance_cv"],
            hhi_index=row["hhi_index"],
            inactive_token_flag=row["inactive_token_flag"],
            whale_domination_ratio=row["whale_domination_ratio"],
            whale_presence_flag=row["whale_presence_flag"],
            few_holders_flag=row["few_holders_flag"],
            airdrop_like_flag=row["airdrop_like_flag"],
            concentrated_large_community_score=row["concentrated_large_community_score"],
            hhi_per_holder=row["hhi_per_holder"],
            whale_but_no_small_flag=row["whale_but_no_small_flag"],
        )

        # 🔹 정적 + 동적 피처 합치기
        defaults.update(dyn_defaults)

        HoneypotProcessedData.objects.update_or_create(
            token_info=token,
            defaults=defaults,
        )

        if i % 50 == 0 or i == total_tokens:
            elapsed = time.time() - start
            speed = i / elapsed if elapsed > 0 else 0
            print(f"  ✅ [{i}/{total_tokens}] done ({speed:.1f} tok/s)")

    elapsed = time.time() - start
    print("-------------------------------------------------")
    print(f"🎉 Completed in {elapsed:.2f}s ({elapsed/60:.2f}m)")


if __name__ == "__main__":
    main()
