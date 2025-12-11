"""
API views for token scam detection system.
Provides endpoints for job submission and result retrieval.
"""

from rest_framework import viewsets, status
from rest_framework.decorators import action, api_view
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from django.db import IntegrityError

from .models import AnalysisJob, Result

import ast
from django.http import JsonResponse
from django.views.decorators.http import require_GET

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
    동시 요청 시 IntegrityError 처리.
    """
    token_addr = request.data.get("token_addr")
    reset = request.data.get("reset", 0)

    if not token_addr:
        return Response({"detail": "token_addr is required"}, status=status.HTTP_400_BAD_REQUEST)

    token_addr = token_addr.lower().strip()

    # 1) 이미 결과 있으면 재사용
    existing = Result.objects.filter(token_addr__iexact=token_addr).first()
    if existing and reset == 0:
        return Response({
            "status": "cached",
            "token_addr": existing.token_addr,
            "riskScore": existing.risk_score,
            "scam_types": existing.scam_types,
            "tokenSnapshot": existing.token_snapshot,
            "holderSnapshot": existing.holder_snapshot,
            "exitInsight": existing.exitInsight,
            "honeypotMlInsight": existing.honeypotMlInsight,
            "honeypotDaInsight": existing.honeypotDaInsight,
            "created_at": existing.created_at
        })

    # 2) 새 분석 실행
    try:
        orch = PipelineOrchestrator()
        ok = orch.execute(token_addr, reset)

        if not ok:
            return Response({"detail": "pipeline failed"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        result = Result.objects.get(token_addr__iexact=token_addr)
        return Response({
            "status": "success",
            "token_addr": result.token_addr,
            "riskScore": result.risk_score,
            "scam_types": result.scam_types,
            "tokenSnapshot": result.token_snapshot,
            "holderSnapshot": result.holder_snapshot,
            "exitInsight": result.exitInsight,
            "honeypotMlInsight": result.honeypotMlInsight,
            "honeypotDaInsight": result.honeypotDaInsight,
            "created_at": result.created_at
        })

    except IntegrityError as e:
        error_msg = str(e).lower()
        if 'unique constraint' in error_msg or 'token_addr' in error_msg:
            return Response({
                "status": "processing",
                "message": "This token is currently being analyzed by another request. Please try again in a few minutes.",
                "token_addr": token_addr
            }, status=status.HTTP_202_ACCEPTED)
        raise

    except Exception as e:
        return Response({"detail": f"Unexpected error: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)