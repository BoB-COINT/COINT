"""
API views for token scam detection system.
Provides endpoints for job submission and result retrieval.
"""

from rest_framework import viewsets, status
from rest_framework.decorators import action, api_view
from rest_framework.response import Response
from django.shortcuts import get_object_or_404

from .models import AnalysisJob, Result

import ast
from django.http import JsonResponse
from django.views.decorators.http import require_GET
from django.shortcuts import get_object_or_404

from pipeline.orchestrator import PipelineOrchestrator


class AnalysisJobViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet for viewing analysis jobs.
    """
    queryset = AnalysisJob.objects.all()
    serializer_class = None  # TODO: Create serializer

    @action(detail=True, methods=['get'])
    def status(self, request, pk=None):
        """
        Get current status of analysis job.
        """
        job = self.get_object()
        return Response({
            'job_id': job.id,
            'status': job.status,
            'current_step': job.current_step,
            'error_message': job.error_message
        })

@require_GET
def result_detail(request, token_addr: str):
    result = get_object_or_404(Result, token_addr__iexact=token_addr)

    # 1) scam_types 문자열 → 리스트로
    scam_types_raw = result.scam_types
    try:
        scam_types = ast.literal_eval(scam_types_raw) if isinstance(scam_types_raw, str) else scam_types_raw
    except Exception:
        scam_types = []

    # 2) honeypotMlInsight 문자열 → 리스트로
    ml_insight_raw = result.honeypotMlInsight
    try:
        ml_insight = ast.literal_eval(ml_insight_raw) if isinstance(ml_insight_raw, str) else ml_insight_raw
    except Exception:
        ml_insight = []

    data = {
        "token_addr": result.token_addr,
        "riskScore": result.risk_score,
        "scam_types": scam_types,
        "tokenSnapshot": result.token_snapshot,
        "holderSnapshot": result.holder_snapshot,
        "exitInsight": result.exitInsight,
        "honeypotMlInsight": ml_insight,
        "honeypotDaInsight": result.honeypotDaInsight,
        "created_at": result.created_at
    }
    return JsonResponse(data)

@api_view(['POST'])
def analyze_token(request):
    """
    새 token_addr를 받아서 전체 파이프라인 실행 후 Result 반환.
    """
    token_addr = request.data.get("token_addr")
    reset = request.data.get("reset")
    if not token_addr:
        return Response({"detail": "token_addr is required"}, status=status.HTTP_400_BAD_REQUEST)
    
    # 1) 이미 결과 있으면 재사용
    existing = Result.objects.filter(token_addr__iexact=token_addr).first()
    if existing and reset == 0:
        # 바로 기존 결과 JSON 리턴 (result_detail과 동일 포맷)
        return Response({
            "token_addr": existing.token_addr,
            "riskScore": existing.risk_score,
            "scam_types": existing.scam_types,
            "tokenSnapshot": existing.token_snapshot,
            "holderSnapshot": existing.holder_snapshot,
            "exitInsight": existing.exitInsight,
            "honeypotMlInsight": existing.honeypotMlInsight,
            "honeypotDaInsight": existing.honeypotDaInsight,
        })            

    # 2) 없으면 오케스트레이터 실행
    orch = PipelineOrchestrator()
    ok = orch.execute(token_addr,reset)
    if not ok:
        return Response({"detail": "pipeline failed"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # 3) 실행 후 Result 다시 조회해서 리턴
    result = Result.objects.get(token_addr__iexact=token_addr)
    return Response({
        "token_addr": result.token_addr,
        "riskScore": result.risk_score,
        "scam_types": result.scam_types,
        "tokenSnapshot": result.token_snapshot,
        "holderSnapshot": result.holder_snapshot,
        "exitInsight": result.exitInsight,
        "honeypotMlInsight": result.honeypotMlInsight,
        "honeypotDaInsight": result.honeypotDaInsight,
    })