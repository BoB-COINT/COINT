#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Honeypot Prediction - v12 (Module)

- final_model_v12.json + metadata_v12.csv 기반 예측 모듈
- 입력: HoneypotProcessedData 스키마와 호환되는 DataFrame (df_raw)
- 출력: y_proba, y_pred, (옵션) top-k feature 이름이 포함된 DataFrame
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import shap

# ======================================================================
# Django 초기화
# ======================================================================

BASE_DIR = Path(__file__).resolve().parent

# 파일 상단 쪽에 추가
_model = None
_best_thr = None


def load_model_and_threshold():
    """모델과 best threshold를 한 번만 로드해서 캐시."""
    global _model, _best_thr

    if _model is not None and _best_thr is not None:
        return _model, _best_thr

    if not Config.META_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found: {Config.META_PATH}")

    meta_row = pd.read_csv(Config.META_PATH).iloc[0]
    best_thr = float(meta_row.get("best_threshold_val", 0.5))

    if not Config.MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {Config.MODEL_PATH}")

    model = xgb.XGBClassifier()
    model.load_model(Config.MODEL_PATH)

    _model = model
    _best_thr = best_thr
    return model, best_thr

# ======================================================================
# Config
# ======================================================================

class Config:
    # 🔹 모델/메타데이터 파일 경로 (input 폴더)
    MODEL_DIR = BASE_DIR / "input"
    MODEL_PATH = MODEL_DIR / "final_model_v12.json"
    META_PATH = MODEL_DIR / "metadata_v12.csv"

    # 🔹 결과 CSV 저장 경로 (원래 로직 유지)
    OUTPUT_DIR = BASE_DIR / "output"
    PREDICTION_OUTPUT = OUTPUT_DIR / "predictions_inference_v12.csv"

    # 🔹 Dynamic feature 목록 (기존 그대로)
    DYNAMIC_FEATURES = [
        "balance_manipulation",
        "blacklist_check",
        "buy_sell",
        "existing_holders_check",
        "exterior_call_check",
        "tax_manipulation",
        "trading_suspend_check",
        "unlimited_mint",
    ]

# ======================================================================
# 피처 엔지니어링 (기존 로직 그대로)
# ======================================================================

def create_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # -------------------- 기존 Static 피처 (v1 베이스) --------------------
    if df["max_sell_share"].max() > 1.0:
        print("  ⚠️  max_sell_share contains values > 1, normalizing...")
        df["max_sell_share"] = df["max_sell_share"].clip(upper=1.0)

    df["sell_vol_per_cnt"] = df["total_sell_vol"] / (df["total_sell_cnt"] + 1)
    df["buy_vol_per_cnt"] = df["total_buy_vol"] / (df["total_buy_cnt"] + 1)

    df["sell_buy_cnt_ratio"] = df["total_sell_cnt"] / (df["total_buy_cnt"] + 1)
    df["sell_buy_vol_ratio"] = df["total_sell_vol"] / (df["total_buy_vol"] + 1)

    df["owner_sell_ratio"] = df["total_owner_sell_cnt"] / (df["total_sell_cnt"] + 1)
    df["non_owner_sell_ratio"] = df["total_non_owner_sell_cnt"] / (
        df["total_sell_cnt"] + 1
    )

    df["seller_buyer_ratio"] = df["unique_sellers"] / (df["unique_buyers"] + 1)
    df["avg_sell_per_seller"] = df["total_sell_cnt"] / (df["unique_sellers"] + 1)
    df["avg_buy_per_buyer"] = df["total_buy_cnt"] / (df["unique_buyers"] + 1)

    df["trade_balance"] = (
        df["total_buy_cnt"] - df["total_sell_cnt"]
    ) / (df["total_buy_cnt"] + df["total_sell_cnt"] + 1)

    df["liquidity_ratio"] = df["windows_with_activity"] / (df["total_windows"] + 1)
    df["sell_concentration"] = df["max_sell_share"] * df["total_sell_cnt"]

    df["activity_intensity"] = (
        (df["windows_with_activity"] / (df["total_windows"] + 1))
        * (df["total_sell_cnt"] + df["total_buy_cnt"])
    )

    df["vol_log_diff"] = df["total_sell_vol_log"] - df["total_buy_vol_log"]

    df["block_window_ratio"] = df["total_sell_block_windows"] / (
        df["consecutive_sell_block_windows"] + 1
    )

    for col in ["sell_vol_per_cnt", "buy_vol_per_cnt", "sell_concentration"]:
        df[f"{col}_log"] = np.log1p(df[col])

    hhi_threshold = 8000.0
    gini_threshold = 0.85

    df["extreme_hhi_flag"] = (df["hhi_index"] > hhi_threshold).astype(int)
    df["extreme_gini_flag"] = (df["gini_coefficient"] > gini_threshold).astype(int)

    # -------------------- Holder 피처 변환 --------------------
    df["holders_per_trade"] = df["total_holders"] / (
        df["total_buy_cnt"] + df["total_sell_cnt"] + 1
    )
    df["holders_per_activity"] = df["total_holders"] / (df["windows_with_activity"] + 1)

    df["high_holders_with_imbalance"] = (
        (df["total_holders"] > 100) &
        (df["imbalance_rate"] > 0.7)
    ).astype(float) * np.log1p(df["total_holders"])

    df["high_holders_active_trading"] = (
        (df["total_holders"] > 100) &
        (df["liquidity_ratio"] > 0.3) &
        (df["imbalance_rate"] < 0.5)
    ).astype(float) * (-1) * np.log1p(df["total_holders"])

    df["holder_data_reliability"] = 1.0

    mask1 = (df["total_holders"] < 5) & (df["extreme_hhi_flag"] == 1)
    df.loc[mask1, "holder_data_reliability"] = 0.5

    mask2 = (df["total_holders"] > 50) & (
        df["total_sell_cnt"] + df["total_buy_cnt"] < 10
    )
    df.loc[mask2, "holder_data_reliability"] = 0.3

    mask3 = (df["total_non_owner_sell_cnt"] == 0) & (df["total_owner_sell_cnt"] > 0)
    df.loc[mask3, "holder_data_reliability"] = 0.1

    mask4 = df["total_holders"] < 3
    df.loc[mask4, "holder_data_reliability"] = 0.4

    df["holder_concentration_risk"] = (
        df["hhi_index"] / 10000 *
        df["imbalance_rate"] *
        np.where(df["total_holders"] < 10, 2.0, 1.0)
    )

    df["holder_asymmetry_score"] = (
        df["gini_coefficient"] *
        (1 - df["holder_data_reliability"]) *
        np.log1p(df["total_holders"])
    )

    df["active_honeypot_score"] = (
        np.log1p(df["total_sell_cnt"])
        * df["imbalance_rate"]
        * df["max_sell_share"]
    )

    mint_threshold = 5
    df["excessive_minting_flag"] = (df["total_mint_events"] > mint_threshold).astype(int)

    df["sell_per_seller_ratio"] = df["total_sell_cnt"] / (df["unique_sellers"] + 1)
    df["high_sell_concentration_flag"] = (df["sell_per_seller_ratio"] > 5).astype(int)

    df["sell_cnt_log"] = np.log1p(df["total_sell_cnt"])
    df["buy_cnt_log"] = np.log1p(df["total_buy_cnt"])
    df["activity_log"] = np.log1p(df["windows_with_activity"])

    # -------------------- Dynamic Analyzer 통합 --------------------
    dynamic_cols = [c for c in Config.DYNAMIC_FEATURES if c in df.columns]
    if dynamic_cols:
        df["dynamic_risk_count"] = df[dynamic_cols].sum(axis=1)
        df["has_any_dynamic_risk"] = (df["dynamic_risk_count"] > 0).astype(int)
        df["has_multiple_risks"] = (df["dynamic_risk_count"] >= 2).astype(int)
        df["high_risk_code"] = (df["dynamic_risk_count"] >= 3).astype(int)

    if "buy_sell" in df.columns and "balance_manipulation" in df.columns:
        df["critical_combo_1"] = (
            (df["buy_sell"] == 1) & (df["balance_manipulation"] == 1)
        ).astype(int)

    if "blacklist_check" in df.columns and "trading_suspend_check" in df.columns:
        df["critical_combo_2"] = (
            (df["blacklist_check"] == 1) & (df["trading_suspend_check"] == 1)
        ).astype(int)

    if "unlimited_mint" in df.columns and "tax_manipulation" in df.columns:
        df["critical_combo_3"] = (
            (df["unlimited_mint"] == 1) & (df["tax_manipulation"] == 1)
        ).astype(int)

    if "buy_sell" in df.columns:
        df["buy_sell_x_sell_cnt"] = df["buy_sell"] * np.log1p(df["total_sell_cnt"])
        df["buy_sell_x_imbalance"] = df["buy_sell"] * df["imbalance_rate"]
        df["no_buy_sell_but_imbalanced"] = (
            (df["buy_sell"] == 0) & (df["imbalance_rate"] > 0.5)
        ).astype(int)

    if "balance_manipulation" in df.columns:
        df["balance_manip_x_concentration"] = (
            df["balance_manipulation"] * df["gini_coefficient"]
        )
        df["balance_manip_x_whale"] = (
            df["balance_manipulation"] * df["whale_domination_ratio"]
        )

    if "unlimited_mint" in df.columns:
        df["unlimited_mint_x_mint_events"] = (
            df["unlimited_mint"] * np.log1p(df["total_mint_events"])
        )
        df["mint_abuse_score"] = (
            df["unlimited_mint"] * df["excessive_minting_flag"]
        )

    if "tax_manipulation" in df.columns:
        df["tax_manip_x_vol_ratio"] = (
            df["tax_manipulation"] * df["sell_buy_vol_ratio"]
        )

    if "exterior_call_check" in df.columns:
        df["exterior_call_x_activity"] = (
            df["exterior_call_check"] * df["activity_intensity"]
        )

    if "buy_sell" in df.columns:
        df["buy_sell_x_holder_imbalance"] = (
            df["buy_sell"] * df["high_holders_with_imbalance"]
        )
        df["buy_sell_x_holders_per_trade"] = (
            df["buy_sell"] * (1 / (df["holders_per_trade"] + 0.01))
        )

    if "balance_manipulation" in df.columns:
        df["balance_manip_x_holder_conc"] = (
            df["balance_manipulation"] * df["holder_concentration_risk"]
        )
        df["balance_manip_x_asymmetry"] = (
            df["balance_manipulation"] * df["holder_asymmetry_score"]
        )

    if "existing_holders_check" in df.columns:
        df["holders_check_x_count"] = (
            df["existing_holders_check"] * np.log1p(df["total_holders"])
        )
        df["holders_check_x_concentration"] = (
            df["existing_holders_check"] * df["holder_concentration_risk"]
        )

    static_risk = (
        df["active_honeypot_score"] * 0.20
        + df["excessive_minting_flag"] * 0.12
        + df["high_sell_concentration_flag"] * 0.12
        + (1 - df["holder_data_reliability"]) * 0.12
        + df["extreme_hhi_flag"] * 0.08
        + df["extreme_gini_flag"] * 0.08
        + (df["imbalance_rate"] > 0.5).astype(int) * 0.08
        + df["holder_concentration_risk"] * 0.10
        + df["high_holders_with_imbalance"] * 0.05
        + df["high_holders_active_trading"] * 0.05
    )

    dynamic_risk = 0
    if "dynamic_risk_count" in df.columns:
        max_risk = len(dynamic_cols) if dynamic_cols else 1
        dynamic_risk = df["dynamic_risk_count"] / max_risk

    df["composite_risk_score_v2"] = static_risk * 0.6 + dynamic_risk * 0.4

    if "verified" in df.columns:
        df["verified_reliability_boost"] = df["verified"] * 0.2
        df["verified_but_risky"] = (
            df["verified"] * df["has_any_dynamic_risk"]
        ).astype(int)

    return df


# ======================================================================
# 전처리: inf / NaN / 이상값 처리
# ======================================================================

def clean_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    for col in X.columns:
        if X[col].isna().any():
            median_val = X[col].median()
            X[col].fillna(median_val, inplace=True)

    assert not np.any(np.isinf(X.values)), "Still contains inf values!"
    assert not np.any(np.isnan(X.values)), "Still contains nan values!"
    assert np.all(np.isfinite(X.values)), "Contains non-finite values!"

    return X


def classify_status(y_true, y_pred):
    if y_true == 1 and y_pred == 1:
        return "TP"
    if y_true == 0 and y_pred == 1:
        return "FP"
    if y_true == 1 and y_pred == 0:
        return "FN"
    if y_true == 0 and y_pred == 0:
        return "TN"
    return "UNKNOWN"

def run_v12_inference(df_raw: pd.DataFrame, compute_shap: bool = True) -> pd.DataFrame:
    """
    HoneypotProcessedData 형태의 raw DataFrame을 받아
    v12 모델로 예측 + top-5 feature 이름까지 계산해서 df_out 반환.

    df_out 컬럼:
      - token_addr_idx (있으면)
      - token_addr (있으면)
      - y_proba, y_pred
      - (옵션) top1_feat ~ top5_feat
      - (옵션) y_true, status
    """
    model, best_thr = load_model_and_threshold()

    if df_raw.empty:
        raise ValueError("df_raw is empty")

    # label 유무 체크
    has_label = "label" in df_raw.columns
    y_true = df_raw["label"].copy() if has_label else None

    # 학습 때 쓰지 않는 컬럼 제거
    drop_cols = [
        "id",
        "label",
        "token_addr_idx",
        "token_addr",
        "token_info",
        "token_info_id",
        "created_at",
        "updated_at",
        "insert_ts",
        "status",
    ]
    cols_to_drop_for_X = [c for c in drop_cols if c in df_raw.columns]
    X_base = df_raw.drop(columns=cols_to_drop_for_X)

    X_base = X_base.apply(pd.to_numeric, errors="coerce")

    # 피처 엔지니어링 + 정제
    X_eng = create_enhanced_features(X_base)
    X_clean = clean_features(X_eng)

    # 예측
    y_proba = model.predict_proba(X_clean)[:, 1]
    y_pred = (y_proba >= best_thr).astype(int)

    # 출력용 기본 컬럼 (token_addr_idx/token_addr은 adapter 쪽에서 df_raw에 넣어줄 것)
    token_ids = df_raw.get("token_addr_idx", pd.Series([None] * len(df_raw)))
    token_addr = df_raw.get("token_addr", pd.Series([None] * len(df_raw)))

    df_out = pd.DataFrame(
        {
            "token_addr_idx": token_ids.values,
            "token_addr": token_addr.values,
            "y_proba": y_proba,
            "y_pred": y_pred,
        }
    )

    # SHAP 기반 top-5 feature (원하면)
    if compute_shap:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_clean)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            shap_values = np.array(shap_values)
            feature_names = np.array(X_clean.columns)

            top_k = 5
            n_samples, _ = shap_values.shape
            top_feat_names = {f"top{i+1}_feat": [] for i in range(top_k)}

            for i in range(n_samples):
                sv_row = shap_values[i]
                idx_sorted = np.argsort(-np.abs(sv_row))[:top_k]
                for rank, idx in enumerate(idx_sorted):
                    name = feature_names[idx]
                    top_feat_names[f"top{rank+1}_feat"].append(name)

            for col_name, values in top_feat_names.items():
                df_out[col_name] = values

        except Exception as e:
            # SHAP 실패해도 예측은 계속
            print(f"  ⚠️ SHAP explanation computation failed: {e}")

    # label 있을 때만 status 붙이기
    if has_label:
        df_out["y_true"] = y_true.values
        df_out["status"] = [
            classify_status(t, p) for t, p in zip(df_out["y_true"], df_out["y_pred"])
        ]
    else:
        df_out["status"] = "PRED_ONLY"

    return df_out
