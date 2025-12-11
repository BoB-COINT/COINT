from pipeline.orchestrator import PipelineOrchestrator
import os
import django
import sys


os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

orch = PipelineOrchestrator()
token_addr = sys.argv[1]
reset = True
success = orch.execute(token_addr,reset)

print("Success:", success)