"""
Adapter implementations for module integration.
Each adapter wraps a module from the modules/ directory and provides
clear input/output specifications for pipeline orchestration.
"""

from typing import Dict, Any, List
from datetime import datetime
from django.utils import timezone


class DataCollectorAdapter:
    """
    Unified data collector for blockchain data.
    Collects all required data (token info, pair events, holders) in one go.

    Input: token_addr (str)
    Output: TokenInfo instance (with related PairEvent and HolderInfo records)
    Database: Inserts into token_info, pair_evt, holder_info tables
    """

    def __init__(self):
        """
        Initialize collector with settings from environment variables.

        Required environment variables:
            - ETHEREUM_RPC_URL: Web3 RPC endpoint (Alchemy)
            - ETHERSCAN_API_KEY: Etherscan API key
            - ETHERSCAN_API_URL: Etherscan V2 API URL
            - MORALIS_API_KEY: Moralis API key
        """
        from modules.data_collector import UnifiedDataCollector
        from django.conf import settings

        self.collector = UnifiedDataCollector(
            rpc_url=settings.ETHEREUM_RPC_URL,
            etherscan_api_key=settings.ETHERSCAN_API_KEY,
            etherscan_api_url=settings.ETHERSCAN_API_URL,
            moralis_api_key=settings.MORALIS_API_KEY,
            chainbase_api_key=settings.CHAINBASE_API_KEY
        )

    def collect_all(self, token_addr: str, days: int = 14) -> Dict[str, Any]:
        """
        Collect all blockchain data for a token.

        Args:
            token_addr: Token contract address (0x...)
            days: Number of days to collect pair events from creation (default: 14)

        Returns:
            Dictionary containing:
                token_info: {
                    - token_addr: str
                    - pair_addr: str
                    - token_create_ts: datetime
                    - lp_create_ts: datetime
                    - pair_idx: int (0 or 1)
                    - pair_type: str
                    - token_creator_addr: str
                }
                pair_events: [
                    {
                        - timestamp: datetime
                        - block_number: int
                        - tx_hash: str
                        - tx_from: str
                        - tx_to: str
                        - evt_idx: int
                        - evt_type: str (Mint, Burn, Swap, Sync)
                        - evt_log: dict (processed event args)
                        - lp_total_supply: str
                    },
                    ...
                ]
                holders: [
                    {
                        - holder_addr: str
                        - balance: str
                        - rel_to_total: str (percentage)
                    },
                    ...
                ]
        """
        return self.collector.collect_all(token_addr, days)

    def save_to_db(self, data: Dict[str, Any]) -> 'TokenInfo':
        """
        Save all collected data to database.

        Args:
            data: Dictionary from collect_all() method

        Returns:
            TokenInfo instance
        """
        from api.models import TokenInfo, PairEvent, HolderInfo

        # 1. Save TokenInfo
        token_info_data = data['token_info']
        token_info = TokenInfo.objects.create(
            token_addr=token_info_data['token_addr'],
            pair_addr=token_info_data['pair_addr'],
            pair_creator=token_info_data['pair_creator'],
            token_create_ts=token_info_data['token_create_ts'],
            lp_create_ts=token_info_data['lp_create_ts'],
            pair_idx=token_info_data['pair_idx'],
            pair_type=token_info_data['pair_type'],
            token_creator_addr=token_info_data['token_creator_addr'],
            symbol=token_info_data.get('symbol'),
            name=token_info_data.get('name'),
            holder_cnt=token_info_data.get('holder_cnt')
        )

        # 2. Save PairEvents (bulk)
        pair_events = [
            PairEvent(
                token_info=token_info,
                timestamp=event['timestamp'],
                block_number=event['block_number'],
                tx_hash=event['tx_hash'],
                tx_from=event['tx_from'],
                tx_to=event['tx_to'],
                evt_idx=event['evt_idx'],
                evt_type=event['evt_type'],
                evt_log=event['evt_log'],
                lp_total_supply=event['lp_total_supply']
            )
            for event in data['pair_events']
        ]
        PairEvent.objects.bulk_create(pair_events, batch_size=1000)

        # 3. Save HolderInfo (bulk)
        holders = [
            HolderInfo(
                token_info=token_info,
                holder_addr=holder['holder_addr'],
                balance=holder['balance'],
                rel_to_total=holder['rel_to_total']
            )
            for holder in data['holders']
        ]
        HolderInfo.objects.bulk_create(holders, batch_size=1000)

        return token_info


class PreprocessorAdapter:
    """
    Adapter for modules/preprocessor.
    Processes raw data into features for ML models.

    Input: TokenInfo instance (with related pair_events and holders)
    Output: HoneypotProcessedData and ExitProcessedData records
    Database: Inserts into honeypot_processed_data and exit_processed_data
    """

    def __init__(self):
        from modules.preprocessor.generate_features_honeypot import (
            compute_honeypot_features,
        )        
        self.compute_honeypot_features = compute_honeypot_features

    def process_for_honeypot(self, token_info: 'TokenInfo') -> Dict[str, Any]:
        from api.models import PairEvent, HolderInfo, HoneypotDaResult

        # 1) DB → raw records
        pair_events = PairEvent.objects.filter(token_info=token_info).values(
            "timestamp",
            "evt_type",
            "tx_from",
            "tx_to",
            "evt_log",
        )
        holders = HolderInfo.objects.filter(token_info=token_info).values(
            "holder_addr",
            "balance",
            "rel_to_total",
        )

        # 2) modules.generate_features_honeypot 호출
        features = self.compute_honeypot_features(
            token_addr=token_info.token_addr,
            owner_addr=token_info.token_creator_addr or "",
            pair_evt_records=pair_events,
            holder_records=holders,
        )

        # 🔐 token_addr 은 항상 TokenInfo 기준으로 강제 세팅
        features["token_addr"] = token_info.token_addr

        # 3) HoneypotDaResult → 동적 플래그 merge
        da = HoneypotDaResult.objects.filter(token_info=token_info).first()
        if da is not None:
            dyn_flags = {
                # === buy 시나리오 ===
                "buy_1": int(bool(da.buy_1)),
                "buy_2": int(bool(da.buy_2)),
                "buy_3": int(bool(da.buy_3)),

                # === sell 시나리오 결과 → sell_result_X ===
                "sell_result_1": int(bool(da.sell_1)),
                "sell_result_2": int(bool(da.sell_2)),
                "sell_result_3": int(bool(da.sell_3)),

                # === sell 실패 타입 (그대로 int 저장) ===
                "sell_fail_type_1": int(da.sell_fail_type_1 or 0),
                "sell_fail_type_2": int(da.sell_fail_type_2 or 0),
                "sell_fail_type_3": int(da.sell_fail_type_3 or 0),

                # === 기타 동적 분석 결과 ===
                "trading_suspend_check": int(bool(da.trading_suspend_result)),
                "exterior_call_check": int(bool(da.exterior_call_result)),
                "unlimited_mint": int(bool(da.unlimited_mint_result)),
                "balance_manipulation": int(bool(da.balance_manipulation_result)),
                "tax_manipulation": int(bool(da.tax_manipulation_result)),
                "existing_holders_check": int(bool(da.existing_holders_result)),
            }
            features.update(dyn_flags)


        return features

    def process_exit_instance(self, token_info: 'TokenInfo') -> int:
        """
        Generate exit instance-level features and save to DB.

        Returns: number of rows saved.
        """
        from modules.preprocessor.exit_instance import ExitInstancePreprocessor
        from api.models import ExitProcessedDataInstance

        pre = ExitInstancePreprocessor()
        rows = pre.compute_exit_features(token_info)
        if not rows:
            return 0

        ExitProcessedDataInstance.objects.filter(token_info=token_info).delete()

        objects = []
        for row in rows:
            objects.append(
                ExitProcessedDataInstance(
                    token_info=token_info,
                    event_time=row["event_time"],
                    tx_hash=row.get("tx_hash") or "",
                    delta_t_sec=row.get("delta_t_sec"),
                    is_swap_event=bool(row.get("is_swap_event")),
                    lp_total_supply=float(row["lp_total_supply"]) if row.get("lp_total_supply") is not None else None,
                    reserve_base_drop_frac=row.get("reserve_base_drop_frac"),
                    reserve_quote=float(row["reserve_quote"]) if row.get("reserve_quote") is not None else None,
                    reserve_quote_drop_frac=row.get("reserve_quote_drop_frac"),
                    price_ratio=row.get("price_ratio"),
                    time_since_last_mint_sec=row.get("time_since_last_mint_sec"),
                    lp_minted_amount_per_sec=float(row["lp_minted_amount_per_sec"]) if row.get("lp_minted_amount_per_sec") is not None else None,
                    lp_burned_amount_per_sec=float(row["lp_burned_amount_per_sec"]) if row.get("lp_burned_amount_per_sec") is not None else None,
                    recent_mint_ratio_last10=row.get("recent_mint_ratio_last10"),
                    recent_mint_ratio_last20=row.get("recent_mint_ratio_last20"),
                    recent_burn_ratio_last10=row.get("recent_burn_ratio_last10"),
                    recent_burn_ratio_last20=row.get("recent_burn_ratio_last20"),
                    reserve_quote_drawdown=row.get("reserve_quote_drawdown"),
                    lp_total_supply_mask=row.get("lp_total_supply_mask"),
                    reserve_quote_mask=row.get("reserve_quote_mask"),
                    price_ratio_mask=row.get("price_ratio_mask"),
                    time_since_last_mint_sec_mask=row.get("time_since_last_mint_sec_mask"),
                )
            )

        ExitProcessedDataInstance.objects.bulk_create(objects, batch_size=1000)
        return len(objects)

    def process_exit_static(self, token_info: 'TokenInfo') -> None:
        """
        Generate exit static features and save to DB.
        """
        from modules.preprocessor.exit_static import ExitStaticPreprocessor
        from api.models import ExitProcessedDataStatic

        pre = ExitStaticPreprocessor()
        data = pre.compute_static_features(token_info)
        ExitProcessedDataStatic.objects.update_or_create(
            token_info=token_info,
            defaults=data,
        )

    def save_honeypot_to_db(self, token_info: 'TokenInfo', data: Dict[str, Any]):
        """
        Save honeypot features to database.
        """
        from api.models import HoneypotProcessedData

        HoneypotProcessedData.objects.create(
            token_info=token_info,
            **data
        )

    ##### [모듈에서 이미 DB에 저장하고 있음]
    # def save_exit_to_db(self, token_info: 'TokenInfo', windows: List[Dict[str, Any]]) -> int:
    #     """
    #     Save exit scam features to database.
    #     """
    #     from api.models import ExitProcessedData

    #     records = [
    #         ExitProcessedData(
    #             token_info=token_info,
    #             **window
    #         )
    #         for window in windows
    #     ]

    #     ExitProcessedData.objects.bulk_create(records, batch_size=1000)
    #     return len(records)


class HoneypotDynamicAnalyzerAdapter:
    """
    Adapter for modules/honeypot_DA.
    Runs Brownie-based dynamic analysis via subprocess.

    Input: TokenInfo instance
    Output: Analysis results (dict)
    Database: Saves to HoneypotDaResult table
    """

    def __init__(self):
        from pathlib import Path
        self.module_path = Path(__file__).parent.parent / "modules" / "honeypot_DA"
        self.script_path = self.module_path / "scripts" / "scam_analyzer.py"

    def _run_analysis(self, token_addr_idx: int):
        """Run honeypot_DA script via subprocess."""
        import subprocess
        import sys

        cmd = [
            sys.executable,
            str(self.script_path),
            str(token_addr_idx),
        ]

        result = subprocess.run(
            cmd,
            cwd=str(self.module_path),
            capture_output=True,
            text=True,
            timeout=600
        )

        if result.returncode != 0:
            raise RuntimeError(f"honeypot_DA failed: {result.stderr}")

        return result.stdout

    def _parse_result_json(self, token_info: 'TokenInfo') -> Dict[str, Any]:
        """Parse result JSON file created by honeypot_DA."""
        import json

        result_file = self.module_path / "results" / f"{token_info.token_addr}.json"

        if not result_file.exists():
            raise FileNotFoundError(f"Result file not found: {result_file}")

        with open(result_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _save_to_db(self, token_info: 'TokenInfo', result_data: Dict[str, Any]):
        """Save analysis result to HoneypotDaResult table."""
        from api.models import HoneypotDaResult

        HoneypotDaResult.objects.update_or_create(
            token_info=token_info,
            defaults={
                'verified': result_data.get('verified', False),
                "buy_1": result_data.get('buy_1'),
                "buy_2": result_data.get('buy_2'),
                "buy_3": result_data.get('buy_3'),
                "sell_1": result_data.get('sell_1'),
                "sell_2": result_data.get('sell_2'),
                "sell_3": result_data.get('sell_3'),
                "sell_fail_type_1": result_data.get('sell_fail_type_1'),
                "sell_fail_type_2": result_data.get('sell_fail_type_2'),
                "sell_fail_type_3": result_data.get('sell_fail_type_3'),                   
                'trading_suspend_result': result_data.get('trading_suspend_check', {}).get('result', False),
                'exterior_call_result': result_data.get('exterior_call_check', {}).get('result', False),
                'unlimited_mint_result': result_data.get('unlimited_mint', {}).get('result', False),
                'balance_manipulation_result': result_data.get('balance_manipulation', {}).get('result', False),
                'tax_manipulation_result': result_data.get('tax_manipulation', {}).get('result', False),
                'existing_holders_result': result_data.get('existing_holders_check', {}).get('result', False),
            }
        )

    def analyze(self, token_info: 'TokenInfo') -> Dict[str, Any]:
        """
        Run dynamic analysis for honeypot detection.

        Args:
            token_info: TokenInfo instance
            processed_data: Not used (kept for interface compatibility)

        Returns:
            Dictionary containing analysis results
        """
        # Run analysis (script will load data from DB)
        self._run_analysis(token_info.id)

        # Parse result
        result_data = self._parse_result_json(token_info)

        # Save to database
        self._save_to_db(token_info, result_data)

        # Return result
        return result_data


class HoneypotMLAnalyzerAdapter:
    """
    Adapter for modules/honeypot_ML (v12).
    ML-based honeypot detection using XGBoost v12 model.

    Input: HoneypotProcessedData
    Output: HoneypotMlResult (DB 저장)
    """

    def __init__(self):
        """Initialize v12 model module."""
        from modules.honeypot_ML.predict_v12 import (
            run_v12_inference,
            load_model_and_threshold,
        )

        # v12 모듈 함수 보관
        self.run_v12_inference = run_v12_inference

        # 모델 & best threshold 한 번만 로드해서 threshold 저장
        _, best_thr = load_model_and_threshold()
        self.threshold = float(best_thr)

        # 리스크 레벨 기준 (MEDIUM = best_thr)
        self.threshold_levels = {
            "CRITICAL": 0.95,
            "HIGH": 0.85,
            "MEDIUM": self.threshold,
            "LOW": 0.40,
        }

    def _determine_risk_level(self, prob: float) -> str:
        """Determine risk level based on probability and thresholds."""
        thr_critical = self.threshold_levels.get("CRITICAL", 0.95)
        thr_high = self.threshold_levels.get("HIGH", 0.85)
        thr_medium = self.threshold_levels.get("MEDIUM", 0.5)
        thr_low = self.threshold_levels.get("LOW", 0.3)

        if prob >= thr_critical:
            return "CRITICAL"
        elif prob >= thr_high:
            return "HIGH"
        elif prob >= thr_medium:
            return "MEDIUM"
        elif prob >= thr_low:
            return "LOW"
        else:
            return "VERY_LOW"

    def predict(self, processed_data: "HoneypotProcessedData") -> Dict[str, Any]:
        """
        Run v12 ML model for honeypot prediction.

        Args:
            processed_data: HoneypotProcessedData instance

        Returns:
            dict:
                - is_honeypot: bool
                - probability: float
                - risk_level: str
                - threshold: float
                - top_feats: [top1_feat, ..., top5_feat]
                - status: str
        """
        import pandas as pd
        from django.forms.models import model_to_dict

        token_info = processed_data.token_info

        # 1) Django model → dict → DataFrame (v12 모듈 입력 형식으로 정리)
        row_dict = model_to_dict(processed_data)

        # v12 모듈이 기대하는 추가 컬럼 세팅
        # token_addr_idx는 따로 없으니 token_info.id를 사용 (모델엔 큰 영향 X)
        row_dict["token_addr_idx"] = token_info.id
        row_dict["token_addr"] = token_info.token_addr

        df_raw = pd.DataFrame([row_dict])

        # 2) v12 모듈 호출 (SHAP 기반 top5 feat 포함)
        df_out = self.run_v12_inference(df_raw, compute_shap=True)
        row = df_out.iloc[0]

        prob = float(row["y_proba"])
        pred = int(row["y_pred"])
        status = row.get("status", "PRED_ONLY")

        # 3) 리스크 레벨 계산
        risk_level = self._determine_risk_level(prob)

        # 4) top1_feat ~ top5_feat 추출
        top_feats: list[str | None] = []
        for i in range(1, 6):
            col = f"top{i}_feat"
            val = row.get(col) if col in df_out.columns else None
            # NaN이면 None으로 정리
            if pd.isna(val) if val is not None else False:
                val = None
            top_feats.append(val)

        result = {
            "is_honeypot": bool(pred == 1),
            "probability": prob,
            "risk_level": risk_level,
            "threshold": float(self.threshold),
            "top_feats": top_feats,
            "status": status,
        }

        # 5) DB 저장
        self._save_to_db(token_info, result)

        return result

    def _save_to_db(self, token_info: "TokenInfo", result: Dict[str, Any]):
        """Save honeypot ML results to HoneypotMlResult table."""
        from api.models import HoneypotMlResult

        top1, top2, top3, top4, top5 = (result["top_feats"] + [None] * 5)[:5]

        HoneypotMlResult.objects.update_or_create(
            token_info=token_info,
            defaults={
                "is_honeypot": result["is_honeypot"],
                "probability": result["probability"],
                "risk_level": result["risk_level"],
                "threshold": result["threshold"],
                "top1_feat": top1,
                "top2_feat": top2,
                "top3_feat": top3,
                "top4_feat": top4,
                "top5_feat": top5,
                "status": result["status"],
            },
        )


class ExitMLAnalyzerAdapter:
    """
    Adapter for modules/exit_ML.
    ML-based exit scam detection using attention-based MIL model.

    Input: TokenInfo instance
    Output: ML prediction results (dict)
    Database: Inserts into exit_ml_result, exit_ml_detect_transaction, exit_ml_detect_static
    """

    def __init__(self):
        pass

    def run(self, token_info: 'TokenInfo', save_to_db: bool = True) -> Dict[str, Any]:
        """
        Run MIL inference using DB-backed features.
        If save_to_db=True, results are persisted to ExitMlResult and related tables.
        """
        from modules.exit_ML.run_exit_detect import (
            run_exit_detection,
            INSTANCE_OUTPUT_FEATURES,
            STATIC_OUTPUT_FEATURES,
        )
        from api.models import ExitMlResult, ExitMlDetectTransaction, ExitMlDetectStatic
        from dateutil import parser as date_parser

        result = run_exit_detection(token_info.id)

        if not save_to_db:
            return result

        tx_cnt_val = int(result.get("tx_cnt") or 0)
        tx_ts_val = result.get("timestamp")
        tx_hash_val = result.get("tx_hash")
        feat_vals = {k: result.get(k) for k in INSTANCE_OUTPUT_FEATURES}
        static_vals = {k: result.get(k) for k in STATIC_OUTPUT_FEATURES}

        ts_parsed = None
        if tx_ts_val:
            try:
                ts_parsed = date_parser.isoparse(tx_ts_val)
            except Exception:
                ts_parsed = None

        exit_result, _ = ExitMlResult.objects.update_or_create(
            token_info=token_info,
            defaults={
                "probability": result["probability"],
                "tx_cnt": tx_cnt_val,
                "timestamp": ts_parsed,
                "tx_hash": tx_hash_val or "",
                "reserve_base_drop_frac": float(feat_vals.get("reserve_base_drop_frac") or 0.0),
                "reserve_quote": float(feat_vals.get("reserve_quote") or 0.0),
                "reserve_quote_drop_frac": float(feat_vals.get("reserve_quote_drop_frac") or 0.0),
                "price_ratio": float(feat_vals.get("price_ratio") or 0.0),
                "time_since_last_mint_sec": float(feat_vals.get("time_since_last_mint_sec") or 0.0),
                "liquidity_age_days": float(static_vals.get("liquidity_age_days") or 0.0),
                "reserve_quote_drawdown_global": float(static_vals.get("reserve_quote_drawdown_global") or 0.0),
            },
        )

        # Top transaction (rank 1)
        tx_data = {
            "timestamp": result.get("timestamp"),
            "tx_hash": result.get("tx_hash"),
            "feature_values": {k: result.get(k) for k in INSTANCE_OUTPUT_FEATURES},
        }

        ts_val = tx_data.get("timestamp")
        ts_parsed = None
        if ts_val:
            try:
                ts_parsed = date_parser.isoparse(ts_val)
            except Exception:
                ts_parsed = None

        ExitMlDetectTransaction.objects.update_or_create(
            exit_ml_result=exit_result,
            rank=1,
            defaults={
                "timestamp": ts_parsed,
                "tx_hash": tx_data.get("tx_hash"),
                "feature_values": tx_data.get("feature_values", {}),
            },
        )

        ExitMlDetectStatic.objects.update_or_create(
            exit_ml_result=exit_result,
            defaults={"feature_values": {k: result.get(k) for k in STATIC_OUTPUT_FEATURES}},
        )

        return result


class ResultAggregatorAdapter:
    """
    Aggregates all analysis results and computes final risk score.

    Input: All analysis results from DA and ML modules
    Output: Result record
    Database: Inserts into result table
    """

    def aggregate(
        self,
        token_info: 'TokenInfo',
        honeypot_da_result: Dict[str, Any],
        honeypot_ml_result: Dict[str, Any],
        exit_ml_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Aggregate all analysis results and compute final risk score.

        Args:
            token_info: TokenInfo instance
            honeypot_da_result: Results from HoneypotDynamicAnalyzerAdapter
            honeypot_ml_result: Results from HoneypotMLAnalyzerAdapter
            exit_ml_result: Results from ExitMLAnalyzerAdapter

        Returns:
            Dictionary containing:
                - risk_score: float (0-100)
                - scam_types: List[str] (e.g. ["Honeypot", "Exit Scam"])
                - victim_insights: List[str] (detailed findings)

        TODO: Implement aggregation logic based on requirements
        Example:
            risk_score = self._compute_risk_score(
                honeypot_da_result,
                honeypot_ml_result,
                exit_ml_result
            )
            scam_types = self._identify_scam_types(...)
            insights = self._generate_insights(...)
        """
        # Placeholder aggregation logic
        scam_types = []
        insights = []
        risk_score = 0.0

        # Aggregate honeypot results
        if honeypot_da_result.get('is_honeypot') or honeypot_ml_result.get('is_honeypot'):
            scam_types.append('Honeypot')
            risk_score += 40.0
            insights.extend(honeypot_da_result.get('indicators', []))

        # Aggregate exit scam results
        if exit_ml_result.get('is_exit_scam'):
            scam_types.append('Exit Scam')
            risk_score += 50.0
            insights.append(f"Exit scam probability: {exit_ml_result.get('probability', 0):.2%}")

        # Normalize risk score to 0-100
        risk_score = min(100.0, risk_score)

        return {
            'risk_score': risk_score,
            'scam_types': scam_types,
            'victim_insights': insights
        }

    def save_to_db(self, token_info: 'TokenInfo', aggregated_data: Dict[str, Any]):
        """
        Save final result to database.

        Args:
            token_info: TokenInfo instance
            aggregated_data: Dictionary from aggregate() method
        """
        from api.models import Result

        Result.objects.create(
            token_addr=token_info.token_addr,
            token_info=token_info,
            risk_score=aggregated_data['risk_score'],
            scam_types=aggregated_data['scam_types'],
            victim_insights=aggregated_data['victim_insights']
        )
