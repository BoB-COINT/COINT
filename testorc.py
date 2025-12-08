from pipeline.orchestrator import PipelineOrchestrator
import os
import django


os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

orch = PipelineOrchestrator()
success = orch.execute()

print("Success:", success)