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
            moralis_api_key=settings.MORALIS_API_KEY
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
            name=token_info_data.get('name')
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
        self.preprocessor = None

    def process_for_honeypot(self, token_info: 'TokenInfo') -> Dict[str, Any]:
        """
        Generate honeypot detection features.

        Args:
            token_info: TokenInfo with related pair_events and holders

        Returns:
            Dictionary with all 23 honeypot features as specified in DB schema

        TODO: Replace with actual module call
        Example:
            return self.preprocessor.compute_honeypot_features(
                token_addr_idx=token_info.id,
                pair_events=token_info.pair_events.all(),
                holders=token_info.holders.all()
            )
        """
        raise NotImplementedError("Module not integrated yet")

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
    Adapter for modules/honeypot_ML.
    ML-based honeypot detection using XGBoost.

    Input: HoneypotProcessedData
    Output: ML prediction results (dict)
    Database: Results stored in memory, aggregated later (TODO)
    """

    def __init__(self):
        """Initialize and load XGBoost model."""
        from pathlib import Path
        import xgboost as xgb
        import pandas as pd

        self.module_path = Path(__file__).parent.parent / "modules" / "honeypot_ML"

        # Load model
        model_path = self.module_path / "input" / "model_v8_addZero.json"
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(model_path))

        # Load metadata
        meta_path = self.module_path / "input" / "metadata_v8_addZero.csv"
        meta_row = pd.read_csv(meta_path).iloc[0]

        # v8 metadata uses 'best_threshold_val' instead of 'threshold'
        self.threshold = float(meta_row.get('best_threshold_val', 0.5))

        # v8 metadata doesn't have removed_features or threshold_dynamic_levels
        self.removed_features = []

        # Use default threshold levels for v8
        self.threshold_levels = {
            'CRITICAL': 0.95,
            'HIGH': 0.85,
            'MEDIUM': 0.64,  # Using v8 threshold
            'LOW': 0.4
        }

        # Features to remove (BASE_REMOVE)
        self.BASE_REMOVE = [
            'whale_total_pct', 'small_holders_pct', 'holder_balance_std',
            'holder_balance_cv', 'hhi_index', 'whale_count'
        ]

    def _parse_removed_features(self, x) -> list:
        """Parse removed_features from metadata."""
        import json
        import ast
        import pandas as pd

        if pd.isna(x):
            return []
        s = str(x).strip()
        if s in ("", "[]"):
            return []
        try:
            return list(ast.literal_eval(s))
        except Exception:
            try:
                return json.loads(s)
            except Exception:
                return []

    def _parse_threshold_levels(self, x) -> dict:
        """Parse threshold_dynamic_levels from metadata."""
        import pandas as pd

        if pd.isna(x):
            return {}
        s = str(x).strip()
        try:
            parsed = eval(s)
            return {k: float(v) for k, v in parsed.items()}
        except Exception:
            return {}

    def _determine_risk_level(self, prob: float) -> str:
        """Determine risk level based on probability and thresholds."""
        if not self.threshold_levels:
            if prob >= 0.9:
                return 'CRITICAL'
            elif prob >= 0.7:
                return 'HIGH'
            elif prob >= 0.5:
                return 'MEDIUM'
            elif prob >= 0.3:
                return 'LOW'
            else:
                return 'VERY_LOW'

        thr_critical = self.threshold_levels.get('CRITICAL', 0.999)
        thr_high = self.threshold_levels.get('HIGH', 0.997)
        thr_medium = self.threshold_levels.get('MEDIUM', 0.6)
        thr_low = self.threshold_levels.get('LOW', 0.3)

        if prob >= thr_critical:
            return 'CRITICAL'
        elif prob >= thr_high:
            return 'HIGH'
        elif prob >= thr_medium:
            return 'MEDIUM'
        elif prob >= thr_low:
            return 'LOW'
        else:
            return 'VERY_LOW'

    def _calculate_holder_statistics(self, token_info: 'TokenInfo') -> dict:
        """Calculate holder statistics from HolderInfo table (only gini and total_holders)."""
        from api.models import HolderInfo
        import numpy as np

        holders = HolderInfo.objects.filter(token_info=token_info).order_by('-balance')

        if not holders.exists():
            return {
                'gini_coefficient': 0.0,
                'total_holders': 0
            }

        total_holders = holders.count()
        balances = [float(h.balance) for h in holders]
        total_supply = sum(balances)

        if total_supply == 0:
            return {
                'gini_coefficient': 0.0,
                'total_holders': total_holders
            }

        # Gini coefficient
        sorted_balances = sorted(balances)
        n = len(sorted_balances)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_balances)) / (n * np.sum(sorted_balances)) - (n + 1) / n

        return {
            'gini_coefficient': gini,
            'total_holders': total_holders
        }

    def _create_advanced_features(self, df):
        """Feature engineering (same as training)."""
        import pandas as pd
        import numpy as np

        df = df.copy()

        # Interaction features
        df['sell_vol_per_cnt'] = df['total_sell_vol'] / (df['total_sell_cnt'] + 1)
        df['buy_vol_per_cnt'] = df['total_buy_vol'] / (df['total_buy_cnt'] + 1)
        df['sell_buy_cnt_ratio'] = df['total_sell_cnt'] / (df['total_buy_cnt'] + 1)
        df['sell_buy_vol_ratio'] = df['total_sell_vol'] / (df['total_buy_vol'] + 1)

        df['owner_sell_ratio'] = df['total_owner_sell_cnt'] / (df['total_sell_cnt'] + 1)
        df['non_owner_sell_ratio'] = df['total_non_owner_sell_cnt'] / (df['total_sell_cnt'] + 1)

        df['seller_buyer_ratio'] = df['unique_sellers'] / (df['unique_buyers'] + 1)
        df['avg_sell_per_seller'] = df['total_sell_cnt'] / (df['unique_sellers'] + 1)
        df['avg_buy_per_buyer'] = df['total_buy_cnt'] / (df['unique_buyers'] + 1)
        df['trade_balance'] = (df['total_buy_cnt'] - df['total_sell_cnt']) / (df['total_buy_cnt'] + df['total_sell_cnt'] + 1)

        # Statistical features
        df['liquidity_ratio'] = df['windows_with_activity'] / (df['total_windows'] + 1)
        df['sell_concentration'] = df['max_sell_share'] * df['total_sell_cnt']
        df['activity_intensity'] = (df['windows_with_activity'] / (df['total_windows'] + 1)) * (df['total_sell_cnt'] + df['total_buy_cnt'])
        df['vol_log_diff'] = df['total_sell_vol_log'] - df['total_buy_vol_log']
        df['block_window_ratio'] = df['total_sell_block_windows'] / (df['consecutive_sell_block_windows'] + 1)

        # Domain features
        df['high_concentration'] = (df['max_sell_share'] > 0.5).astype(int)

        # Log transformations
        for col in ['sell_vol_per_cnt', 'buy_vol_per_cnt', 'sell_concentration']:
            df[f'{col}_log'] = np.log1p(df[col])

        return df

    def _clean_data(self, df):
        """Clean data (handle inf, nan, clip values)."""
        import numpy as np

        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        float32_min, float32_max = np.finfo(np.float32).min, np.finfo(np.float32).max
        num_cols = df.select_dtypes(include=np.number).columns
        df[num_cols] = df[num_cols].clip(lower=float32_min, upper=float32_max)

        return df

    def _compute_feature_contributions(self, X, feature_names):
        """
        Compute feature contributions using XGBoost pred_contribs.

        Args:
            X: numpy array (1 sample)
            feature_names: list of feature names

        Returns:
            list of top-5 contributing features
        """
        import xgboost as xgb

        # XGBoost DMatrix
        dmatrix = xgb.DMatrix(X, feature_names=feature_names)

        # Get contributions (shape: n_samples x n_features+1)
        # Last column is bias
        contribs = self.model.get_booster().predict(dmatrix, pred_contribs=True)

        # Single sample
        sample_contribs = contribs[0]
        bias = sample_contribs[-1]
        feature_contribs = sample_contribs[:-1]

        # Build contribution list
        contrib_list = []
        for j, (fname, contrib) in enumerate(zip(feature_names, feature_contribs)):
            contrib_list.append({
                'feature': fname,
                'contribution': float(contrib),
                'abs_contribution': float(abs(contrib)),
                'direction': 'increases_risk' if contrib > 0 else 'decreases_risk',
                'feature_value': float(X[0, j])
            })

        # Sort by absolute contribution
        contrib_list.sort(key=lambda x: x['abs_contribution'], reverse=True)

        # Return top 5
        return contrib_list[:5]

    def predict(self, processed_data: 'HoneypotProcessedData') -> Dict[str, Any]:
        """
        Run ML model for honeypot prediction with feature contributions.

        Args:
            processed_data: HoneypotProcessedData instance

        Returns:
            Dictionary containing:
                - is_honeypot: bool
                - probability: float (0-1)
                - risk_level: str (CRITICAL/HIGH/MEDIUM/LOW/VERY_LOW)
                - threshold: float
                - top_contributing_features: list (top 5)
        """
        import pandas as pd

        token_info = processed_data.token_info

        # 1. Calculate holder statistics
        holder_stats = self._calculate_holder_statistics(token_info)

        # 2. Build DataFrame with base features
        data = {
            'total_buy_cnt': processed_data.total_buy_cnt,
            'total_sell_cnt': processed_data.total_sell_cnt,
            'total_owner_sell_cnt': processed_data.total_owner_sell_cnt,
            'total_non_owner_sell_cnt': processed_data.total_non_owner_sell_cnt,
            'imbalance_rate': processed_data.imbalance_rate,
            'total_windows': processed_data.total_windows,
            'windows_with_activity': processed_data.windows_with_activity,
            'total_burn_events': processed_data.total_burn_events,
            'total_mint_events': processed_data.total_mint_events,
            's_owner_count': processed_data.s_owner_count,
            'total_sell_vol': float(processed_data.total_sell_vol),
            'total_buy_vol': float(processed_data.total_buy_vol),
            'total_owner_sell_vol': float(processed_data.total_owner_sell_vol),
            'total_sell_vol_log': processed_data.total_sell_vol_log,
            'total_buy_vol_log': processed_data.total_buy_vol_log,
            'total_owner_sell_vol_log': processed_data.total_owner_sell_vol_log,
            'liquidity_event_mask': processed_data.liquidity_event_mask,
            'max_sell_share': processed_data.max_sell_share,
            'unique_sellers': processed_data.unique_sellers,
            'unique_buyers': processed_data.unique_buyers,
            'consecutive_sell_block_windows': processed_data.consecutive_sell_block_windows,
            'total_sell_block_windows': processed_data.total_sell_block_windows,
            **holder_stats
        }

        df = pd.DataFrame([data])

        # 3. Remove BASE_REMOVE features
        df = df[[c for c in df.columns if c not in self.BASE_REMOVE]]

        # 4. Feature engineering
        df = self._create_advanced_features(df)
        df = self._clean_data(df)

        # 5. Remove metadata removed_features
        if self.removed_features:
            df = df[[c for c in df.columns if c not in self.removed_features]]

        # 6. Align features with model
        if hasattr(self.model, "feature_names_in_"):
            need = list(self.model.feature_names_in_)
            for c in need:
                if c not in df.columns:
                    df[c] = 0.0
            df = df[need]

        # 7. Predict
        X = df.values
        feature_names = df.columns.tolist()

        proba = self.model.predict_proba(X)[:, 1][0]
        pred = int(proba >= self.threshold)
        risk_level = self._determine_risk_level(proba)

        # 8. Compute feature contributions
        top_features = self._compute_feature_contributions(X, feature_names)

        result = {
            'is_honeypot': bool(pred == 1),
            'probability': float(proba),
            'risk_level': risk_level,
            'threshold': float(self.threshold),
            'top_contributing_features': top_features
        }

        # Save to DB
        self._save_to_db(token_info, result)

        return result

    def _save_to_db(self, token_info: 'TokenInfo', result: Dict[str, Any]):
        """Save honeypot ML results to database."""
        from api.models import HoneypotMlResult

        HoneypotMlResult.objects.update_or_create(
            token_info=token_info,
            defaults={
                'is_honeypot': result['is_honeypot'],
                'probability': result['probability'],
                'risk_level': result['risk_level'],
                'threshold': result['threshold'],
                'top_contributing_features': result['top_contributing_features']
            }
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
