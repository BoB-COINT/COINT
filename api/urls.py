from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import AnalysisJobViewSet, result_detail, analyze_token_view

router = DefaultRouter()
router.register(r'jobs', AnalysisJobViewSet, basename='job')

urlpatterns = [
    path('analyze/', analyze_token_view, name='analyze-token'),
    path('results/<str:token_addr>/', result_detail, name='result-detail'),
    path('', include(router.urls)),
]
