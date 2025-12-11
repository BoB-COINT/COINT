"""
Pipeline orchestrator for token scam detection workflow.
Coordinates all stages from data collection to final result aggregation.
"""

import logging
from typing import Optional, Dict, Any
from django.utils import timezone
from django.db import transaction

logger = logging.getLogger(__name__)

class PipelineOrchestrator:
    
    def _reset_pipeline_tables(self) -> None:
        """
        Result 테이블을 제외한 중간 테이블 전체 초기화.
        호출 시점: execute() 맨 앞에서 1번 호출.
        """
        from api.models import (
            TokenInfo,
            PairEvent,
            HolderInfo,
            HoneypotDaResult,
            HoneypotProcessedData,
            HoneypotMlResult,
            ExitProcessedDataInstance,
            ExitProcessedDataStatic,
            ExitMlResult,
        )

        logger.info("Resetting pipeline tables (except Result) ...")

        # 수집 단계
        TokenInfo.objects.all().delete()
        PairEvent.objects.all().delete()
        HolderInfo.objects.all().delete()

        # Honeypot 파이프라인
        HoneypotDaResult.objects.all().delete()
        HoneypotProcessedData.objects.all().delete()
        HoneypotMlResult.objects.all().delete()

        # Exit 파이프라인
        ExitProcessedDataInstance.objects.all().delete()
        ExitProcessedDataStatic.objects.all().delete()
        ExitMlResult.objects.all().delete()

        logger.info("Pipeline tables reset completed.")
    
    def _reset_all_tables(self,token_addr) -> None:
        from api.models import (
            TokenInfo,
            PairEvent,
            HolderInfo,
            HoneypotDaResult,
            HoneypotProcessedData,
            HoneypotMlResult,
            ExitProcessedDataInstance,
            ExitProcessedDataStatic,
            ExitMlResult,
            Result
        )

        logger.info("Resetting pipeline tables (except Result) ...")

        token_info = TokenInfo.objects.get(token_addr=token_addr)
        #수집 단계
        PairEvent.objects.filter(token_info=token_info).delete()
        HolderInfo.objects.filter(token_info=token_info).delete()

        # Honeypot 파이프라인
        HoneypotDaResult.objects.filter(token_info=token_info).delete()
        HoneypotProcessedData.objects.filter(token_info=token_info).delete()
        HoneypotMlResult.objects.filter(token_info=token_info).delete()

        # Exit 파이프라인
        ExitProcessedDataInstance.objects.filter(token_info=token_info).delete()
        ExitProcessedDataStatic.objects.filter(token_info=token_info).delete()
        ExitMlResult.objects.filter(token_info=token_info).delete()
        TokenInfo.objects.filter(token_addr=token_addr).delete()
        Result.objects.filter(token_addr=token_addr).delete()

        logger.info("Pipeline tables reset completed.!!!!!!^_^")

    """
    Orchestrates the complete analysis pipeline according to workflow design.

    Workflow:
        1. Check if token already analyzed (result table)
        2. Collect token info (TokenInfo)
        3. Collect pair events (PairEvent)
        4. Collect holder info (HolderInfo)
        5. Preprocess data (HoneypotProcessedData, ExitProcessedData)
        6. Run honeypot dynamic analysis
        7. Run honeypot ML analysis
        8. Run exit scam ML analysis
        9. Aggregate results and save to result table
    """

    def __init__(self):
        """
        Initialize all pipeline adapters.
        """
        from .adapters import (
            DataCollectorAdapter,
            DetectUnformedLpAdapter,
            PreprocessorAdapter,
            HoneypotDynamicAnalyzerAdapter,
            HoneypotMLAnalyzerAdapter,
            ExitMLAnalyzerAdapter,
            ResultAggregatorAdapter,
        )

        self.token_collector = DataCollectorAdapter()
        self.unformedlp = DetectUnformedLpAdapter()
        self.preprocessor = PreprocessorAdapter()
        self.honeypot_da = HoneypotDynamicAnalyzerAdapter()
        self.honeypot_ml = HoneypotMLAnalyzerAdapter()
        self.exit_ml = ExitMLAnalyzerAdapter()
        self.aggregator = ResultAggregatorAdapter()

    def check_existing_result(self, token_addr: str) -> Optional['Result']:
        """
        Check if token has already been analyzed.

        Args:
            token_addr: Token contract address

        Returns:
            Result instance if exists, None otherwise
        """
        from api.models import Result

        try:
            return Result.objects.get(token_addr__iexact=token_addr)
        except Result.DoesNotExist:
            return None

    def execute(self, token_addr: str, reset,days: int | None = None,) -> bool:
        """
        Execute complete pipeline for given token.
        """
        # from api.models import AnalysisJob  # 아직 안 쓰면 주석으로만 두기

        logger.info(f"Starting pipeline for token {token_addr}")

        try:
            # 1) 이미 분석된 토큰인지 확인
            existing_result = self.check_existing_result(token_addr)
            if existing_result and reset == 0:
                logger.info(f"Token {token_addr} already analyzed, using cached result")
                return True
            
            # # 0) Result 제외 나머지 테이블 초기화
            # self._reset_pipeline_tables()

            if reset == 1:
                print("result reset!!!")
                self._reset_all_tables(token_addr=token_addr)

            # 2) 토큰 메타데이터 수집
            token_info = self._collect_token_info(token_addr, days)

            # 3) Honeypot DA
            honeypot_da_result = self._run_honeypot_da(token_info)

            # 4) Unformed LP 검사
            is_unformed = self._run_unformed_lp(token_info)
            
            # 5) 전처리
            self._preprocess_data(token_info, is_unformed)

            # 6) Honeypot ML

            # 7) Exit ML (Unformed LP면 dummy)
            if not is_unformed:
                exit_ml_result = self._run_exit_ml(token_info)
                honeypot_ml_result = self._run_honeypot_ml(token_info)
            else:
                exit_ml_result = {
                    "token_addr": token_addr,
                    "probability": None,
                    "tx_cnt": None,
                    "timestamp": None,
                    "tx_hash": None,
                    "reserve_base_drop_frac": None,
                    "reserve_quote": None,
                    "reserve_quote_drop_frac": None,
                    "price_ratio": None,
                    "time_since_last_mint_sec": None,
                    "liquidity_age_days": None,
                    "reserve_quote_drawdown_global": None,
                }
                honeypot_ml_result = {
                    "is_honeypot": None,
                    "probability": None,
                    "risk_level": None,
                    "threshold": None,
                    "top_feats": [None,None,None,None,None],
                    "top_feat_values": [None,None,None,None,None],  # 🔹 새로 추가
                    "status": None
                }
                self.honeypot_ml._save_to_db(token_info,honeypot_ml_result)

            # 8) 결과 집계 & 저장
            self._aggregate_and_save_results(
                token_info,
                is_unformed,
                honeypot_da_result,
                honeypot_ml_result,
                exit_ml_result,
            )

            # 9) 처리 완료 플래그 해제
            token_info.is_processing = False
            token_info.save()

            return True

        except Exception as e:
            logger.error(f"Pipeline failed for token {token_addr}: {e}")
            # 에러 발생 시에도 플래그 해제
            from api.models import TokenInfo
            try:
                token_obj = TokenInfo.objects.filter(token_addr__iexact=token_addr).first()
                if token_obj:
                    token_obj.is_processing = False
                    token_obj.save()
            except:
                pass
            return False

    def _run_unformed_lp(self,token_info):
        result = self.unformedlp.run(token_info)

        is_unformed = bool(result.get("is_unformed_lp"))

        return is_unformed
    
    def _collect_token_info(self, token_addr, days) -> 'TokenInfo':
        logger.info(f"Collecting token info for {token_addr}")

        try:
            token_data = self.token_collector.collect_all(token_addr, days)
            token_info = self.token_collector.save_to_db(token_data)
            logger.info(f"Token info collected, token_addr_idx={token_info.id}")
            return token_info

        except KeyError as e:
            # LP가 없을 때 'pair_creator' KeyError 나오는 케이스 방어
            if str(e) == "'pair_creator'":
                logger.error(f"No pair found for token {token_addr} (missing pair_creator)")
            else:
                logger.error(f"Failed to collect token info (KeyError): {e}")
            raise

        except Exception as e:
            logger.error(f"Failed to collect token info: {e}")
            raise


    # def _collect_pair_events(self, job: 'AnalysisJob', token_info: 'TokenInfo'):
    #     """
    #     Step 3: Collect pair event history.
    #     """
    #     job.status = 'collecting_pair'
    #     job.current_step = 'Collecting pair event history'
    #     job.save()

    #     logger.info(f"Collecting pair events for token {token_info.id}")

    #     try:
    #         # Call module to collect events
    #         events = self.pair_collector.collect(token_info)

    #         # Save to database
    #         count = self.pair_collector.save_to_db(token_info, events)

    #         logger.info(f"Collected {count} pair events")

    #     except Exception as e:
    #         job.error_step = 'collecting_pair'
    #         logger.error(f"Failed to collect pair events: {e}")
    #         raise

    # def _collect_holder_info(self, job: 'AnalysisJob', token_info: 'TokenInfo'):
    #     """
    #     Step 4: Collect token holder information.
    #     """
    #     job.status = 'collecting_holder'
    #     job.current_step = 'Collecting token holder information'
    #     job.save()

    #     logger.info(f"Collecting holder info for token {token_info.id}")

    #     try:
    #         # Call module to collect holders
    #         holders = self.holder_collector.collect(token_info)

    #         # Save to database
    #         count = self.holder_collector.save_to_db(token_info, holders)

    #         logger.info(f"Collected {count} holders")

    #     except Exception as e:
    #         job.error_step = 'collecting_holder'
    #         logger.error(f"Failed to collect holder info: {e}")
    #         raise

    def _preprocess_data(self, token_info: 'TokenInfo', is_unformed):
        """
        Step 5: Preprocess data for ML models.
        """
        # job.status = 'preprocessing'
        # job.current_step = 'Preprocessing data for analysis'
        # job.save()

        logger.info(f"Preprocessing data for token {token_info.id}")

        try:
            # Generate honeypot features
            honeypot_features = self.preprocessor.process_for_honeypot(token_info)
            self.preprocessor.save_honeypot_to_db(token_info, honeypot_features)
            logger.info(f"Honeypot features generated")

            # Generate exit scam features
            if not is_unformed:
                exit_windows = self.preprocessor.process_exit_instance(token_info)
                self.preprocessor.process_exit_static(token_info)
                logger.info(f"Exit scam features generated for {exit_windows} windows")

        except Exception as e:
            # job.error_step = 'preprocessing'
            logger.error(f"Failed to preprocess data: {e}")
            raise

    def _run_honeypot_da(self, token_info: 'TokenInfo') -> Dict[str, Any]:
        """
        Step 6: Run honeypot dynamic analysis.
        """
        # job.status = 'analyzing_honeypot_da'
        # job.current_step = 'Running honeypot dynamic analysis'
        # job.save()

        logger.info(f"Running honeypot DA for token {token_info.id}")

        try:
            result = self.honeypot_da.analyze(token_info)
            logger.info(f"Honeypot DA completed: {result.get('is_honeypot', False)}")
            return result

        except Exception as e:
            # job.error_step = 'analyzing_honeypot_da'
            logger.error(f"Failed to run honeypot DA: {e}")
            raise

    def _run_honeypot_ml(self, token_info: 'TokenInfo') -> Dict[str, Any]:
        """
        Step 7: Run honeypot ML analysis for *only this token*.
        """
        from api.models import HoneypotProcessedData

        logger.info(
            f"Running honeypot ML for token_id={token_info.id} "
            f"(token_addr={token_info.token_addr})"
        )

        try:
            # ✅ 현재 token에 해당하는 전처리 레코드 1건만 사용
            hp = HoneypotProcessedData.objects.select_related("token_info").get(
                token_info=token_info
            )

            result = self.honeypot_ml.predict(hp)
            logger.info(
                f"Honeypot ML completed for token_id={token_info.id}, "
                f"is_honeypot={result.get('is_honeypot', False)}"
            )
            return result

        except HoneypotProcessedData.DoesNotExist:
            logger.error(
                f"No HoneypotProcessedData found for token_id={token_info.id} "
                f"(token_addr={token_info.token_addr})"
            )
            raise

        except Exception as e:
            logger.error(f"Failed to run honeypot ML: {e}")
            raise


    def _run_exit_ml(self, token_info: 'TokenInfo') -> Dict[str, Any]:
        """
        Step 8: Run exit scam ML analysis.
        """
        # job.status = 'analyzing_exit_ml'
        # job.current_step = 'Running exit scam ML analysis'
        # job.save()

        logger.info(f"Running exit ML for token {token_info.id}")

        try:
            result = self.exit_ml.run(token_info,save_to_db=True)
            logger.info(f"Exit ML completed: {result.get('is_exit_scam', False)}")
            return result

        except Exception as e:
            # job.error_step = 'analyzing_exit_ml'
            logger.error(f"Failed to run exit ML: {e}")
            raise

    def _aggregate_and_save_results(
        self,
        token_info: 'TokenInfo',
        is_unformed,
        honeypot_da_result: Dict[str, Any],
        honeypot_ml_result: Dict[str, Any],
        exit_ml_result: Dict[str, Any]
    ):
        """
        Step 9: Aggregate all results and save to result table.
        """
        # job.status = 'aggregating'
        # job.current_step = 'Aggregating analysis results'
        # job.save()

        logger.info(f"Aggregating results for token {token_info.id}")

        try:
            # Aggregate results
            aggregated_data = self.aggregator.aggregate(
                token_info,
                is_unformed,
                honeypot_da_result,
                honeypot_ml_result,
                exit_ml_result
            )

            # Save to result table
            self.aggregator.save_to_db(token_info,is_unformed,aggregated_data)

            logger.info(f"Results aggregated and saved, risk_score={aggregated_data['risk_score']}")

        except Exception as e:
            # job.error_step = 'aggregating'
            logger.error(f"Failed to aggregate results: {e}")
            raise

    @classmethod
    def execute_async(cls, token_addr: str, days: int | None = None):
        """
        Execute pipeline asynchronously.
        TODO: Integrate with Celery for true async execution.

        Args:
            job_id: ID of AnalysisJob to process
        """
        orchestrator = cls()
        return orchestrator.execute(token_addr, days)