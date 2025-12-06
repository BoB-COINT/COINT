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
            - CHAINBASE_API_KEY: Chainbase API key
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
        """
        TODO: Import actual module when available.
        Example:
            from modules.preprocessor import DataPreprocessor
            self.preprocessor = DataPreprocessor()
        """
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

    def process_for_exit(self, token_info: 'TokenInfo') -> List[Dict[str, Any]]:
        """
        Generate exit scam detection features (per 5-second window).

        Args:
            token_info: TokenInfo with related pair_events and holders

        Returns:
            List of dictionaries, each representing one 5-second window
            with all 52 exit detection features as specified in DB schema

        TODO: Replace with actual module call
        Example:
            return self.preprocessor.compute_exit_features(
                token_addr_idx=token_info.id,
                pair_events=token_info.pair_events.all(),
                holders=token_info.holders.all()
            )
        """
        raise NotImplementedError("Module not integrated yet")

    def save_honeypot_to_db(self, token_info: 'TokenInfo', data: Dict[str, Any]):
        """
        Save honeypot features to database.
        """
        from api.models import HoneypotProcessedData

        HoneypotProcessedData.objects.create(
            token_info=token_info,
            **data
        )

    def save_exit_to_db(self, token_info: 'TokenInfo', windows: List[Dict[str, Any]]) -> int:
        """
        Save exit scam features to database.
        """
        from api.models import ExitProcessedData

        records = [
            ExitProcessedData(
                token_info=token_info,
                **window
            )
            for window in windows
        ]

        ExitProcessedData.objects.bulk_create(records, batch_size=1000)
        return len(records)


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
            encoding="utf-8",   # 🔹 명시적으로 UTF-8 사용
            errors="ignore",    # 🔹 디코딩 안 되는 바이트는 버리기
            timeout=600,
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

        def to_bool(value):
            """
            value 형태:
              - 0/1, True/False
              - "0"/"1"/"true"/"false"
              - {"result": 0/1, ...} 또는 {"flag": ...}
            을 모두 Bool로 통일.
            """
            if isinstance(value, dict):
                if "result" in value:
                    value = value["result"]
                elif "flag" in value:
                    value = value["flag"]
                elif "value" in value:
                    value = value["value"]

            # 숫자/문자 케이스 처리
            if isinstance(value, (int, float, bool)):
                return bool(value)
            if isinstance(value, str):
                s = value.strip().lower()
                if s in ("1", "true", "yes", "y"):
                    return True
                if s in ("0", "false", "no", "n", ""):
                    return False
            return bool(value)

        def to_int(value):
            """
            value 형태:
              - 정수/실수
              - "1", "2"
              - {"code": 1}, {"value": 1}
            을 모두 int로 통일.
            """
            if isinstance(value, dict):
                for key in ("code", "value", "result"):
                    if key in value:
                        value = value[key]
                        break
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0

        HoneypotDaResult.objects.update_or_create(
            token_info=token_info,
            defaults={
                # 기본 verified
                "verified": to_bool(result_data.get("verified", False)),

                # buy 시나리오 (동적 CSV의 buy_1,2,3과 동일하게)
                "buy_1": to_bool(result_data.get("buy_1")),
                "buy_2": to_bool(result_data.get("buy_2")),
                "buy_3": to_bool(result_data.get("buy_3")),

                # sell 시나리오 (sell_result_1,2,3 → DB의 sell_1,2,3)
                "sell_1": to_bool(
                    result_data.get("sell_result_1", result_data.get("sell_1"))
                ),
                "sell_2": to_bool(
                    result_data.get("sell_result_2", result_data.get("sell_2"))
                ),
                "sell_3": to_bool(
                    result_data.get("sell_result_3", result_data.get("sell_3"))
                ),

                # sell 실패 타입
                "sell_fail_type_1": to_int(result_data.get("sell_fail_type_1")),
                "sell_fail_type_2": to_int(result_data.get("sell_fail_type_2")),
                "sell_fail_type_3": to_int(result_data.get("sell_fail_type_3")),

                # 나머지 체크 플래그들 (동적 CSV의 *_check, unlimited_mint 등과 동일)
                "trading_suspend_result": to_bool(
                    result_data.get("trading_suspend_check",
                                    result_data.get("trading_suspend_result"))
                ),
                "exterior_call_result": to_bool(
                    result_data.get("exterior_call_check",
                                    result_data.get("exterior_call_result"))
                ),
                "unlimited_mint_result": to_bool(
                    result_data.get("unlimited_mint",
                                    result_data.get("unlimited_mint_result"))
                ),
                "balance_manipulation_result": to_bool(
                    result_data.get("balance_manipulation",
                                    result_data.get("balance_manipulation_result"))
                ),
                "tax_manipulation_result": to_bool(
                    result_data.get("tax_manipulation",
                                    result_data.get("tax_manipulation_result"))
                ),
                "existing_holders_result": to_bool(
                    result_data.get("existing_holders_check",
                                    result_data.get("existing_holders_result"))
                ),
            },
        )

    def analyze(self, token_info: 'TokenInfo', processed_data: 'HoneypotProcessedData' = None) -> Dict[str, Any]:
        """
        Run dynamic analysis for honeypot detection.

        Args:
            token_info: TokenInfo instance
            processed_data: (optional) HoneypotProcessedData, not used for now

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

    Input: TokenInfo instance (token_addr)
    Output: ML prediction results (dict) + DB storage
    Database: Inserts into exit_ml_result, exit_ml_detect_transaction, exit_ml_detect_static
    """

    def __init__(self):
        """Initialize adapter and set up paths."""
        from pathlib import Path

        self.module_path = Path(__file__).parent.parent / "modules" / "exit_ML"
        self.artifact_dir = self.module_path
        self.feature_csv = self.module_path / "features_exit_mil_new2.csv"
        self.static_csv = self.module_path / "features_exit_static3.csv"
        self.token_info_csv = self.module_path / "token_information3.csv"

    def predict(self, token_info: 'TokenInfo') -> Dict[str, Any]:
        """
        Run ML model for exit scam prediction.

        Args:
            token_info: TokenInfo instance

        Returns:
            Dictionary containing:
                - token_addr: str
                - probability: float (0-1)
                - detect_tx_1: Dict (timestamp, tx_hash, feature_values)
                - detect_tx_2: Dict (or None)
                - detect_tx_3: Dict (or None)
                - detect_static: Dict (static feature values)
        """
        import sys
        sys.path.insert(0, str(self.module_path))

        try:
            from run_mil_inference import run_inference
        except ImportError as e:
            raise RuntimeError(f"Failed to import run_mil_inference: {e}")

        # Run inference using the module's function
        result = run_inference(
            token_addr=token_info.token_addr,
            feature_path=self.feature_csv,
            static_path=self.static_csv,
            token_info_path=self.token_info_csv,
            artifact_dir=self.artifact_dir,
            output_path=None  # Don't write to file
        )

        return result

    def save_to_db(self, token_info: 'TokenInfo', result: Dict[str, Any]):
        """
        Save exit ML results to database.

        Args:
            token_info: TokenInfo instance
            result: Dictionary from predict() method
        """
        from api.models import ExitMlResult, ExitMlDetectTransaction, ExitMlDetectStatic
        from datetime import datetime

        probability = float(result.get('probability', 0.0))

        # Load threshold from training.json
        import json
        training_json_path = self.module_path / "attention_mil_training.json"
        threshold = 0.6047  # default
        if training_json_path.exists():
            with open(training_json_path) as f:
                training_data = json.load(f)
                threshold = training_data.get('best_threshold', threshold)

        is_exit_scam = probability >= threshold

        # Create or update ExitMlResult
        exit_ml_result, created = ExitMlResult.objects.update_or_create(
            token_info=token_info,
            defaults={
                'probability': probability,
                'threshold': threshold,
                'is_exit_scam': is_exit_scam
            }
        )

        # Save detect transactions (top 3)
        for rank in [1, 2, 3]:
            detect_key = f"detect_tx_{rank}"
            detect_data = result.get(detect_key)

            if detect_data is None:
                continue

            # Parse timestamp
            timestamp = None
            if detect_data.get('timestamp'):
                try:
                    from dateutil import parser as date_parser
                    timestamp = date_parser.isoparse(detect_data['timestamp'])
                except:
                    pass

            # Extract feature values (exclude timestamp and tx_hash)
            feature_values = {
                k: v for k, v in detect_data.items()
                if k not in ['timestamp', 'tx_hash']
            }

            ExitMlDetectTransaction.objects.update_or_create(
                exit_ml_result=exit_ml_result,
                rank=rank,
                defaults={
                    'timestamp': timestamp,
                    'tx_hash': detect_data.get('tx_hash'),
                    'feature_values': feature_values
                }
            )

        # Save detect static features
        detect_static = result.get('detect_static', {})
        if detect_static:
            ExitMlDetectStatic.objects.update_or_create(
                exit_ml_result=exit_ml_result,
                defaults={
                    'feature_values': detect_static
                }
            )


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