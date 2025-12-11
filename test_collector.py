import os
import django
import time

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from pipeline.adapters import DataCollectorAdapter,HoneypotDynamicAnalyzerAdapter
from api.models import TokenInfo, PairEvent, HolderInfo, ExitProcessedDataInstance, ExitProcessedDataStatic

# 기존 데이터 삭제
print("🗑️  기존 데이터 삭제 중...")
PairEvent.objects.all().delete()
HolderInfo.objects.all().delete()
TokenInfo.objects.all().delete()
ExitProcessedDataInstance.objects.all().delete()
ExitProcessedDataStatic.objects.all().delete()

print("   삭제 완료!")

# 데이터 수집
collector = DataCollectorAdapter()
analyzer = HoneypotDynamicAnalyzerAdapter()
token_addr = "0x0A434553C582cdD165AC7083CddB959d413e08F0"

print(f"\n🔍 수집 시작: {token_addr}")
start_time = time.time()
data = collector.collect_all(token_addr,None)
collection_time = time.time() - start_time

print(f"\n📊 수집된 데이터:")
print(f"   Token Info: {data['token_info']['token_addr']}")
print(f"   Pair Addr: {data['token_info']['pair_addr']}")
print(f"   Pair Events: {len(data['pair_events'])}개")
print(f"   Holders: {data['token_info']['holder_cnt']}개")
print(f"   수집 시간: {collection_time:.2f}초")

start_time = time.time()
token_info = collector.save_to_db(data)
save_time = time.time() - start_time

print(f"\n✅ DB 저장 완료!")
print(f"   Token ID: {token_info.id}")
print(f"   Pair Events: {token_info.pair_events.count()}")
print(f"   Holders: {token_info.holders.count()}")
print(f"   저장 시간: {save_time:.2f}초")

token_info2= TokenInfo.objects.get(id=token_info.id)
result_da = analyzer.analyze(token_info2)
print(f"\n✅ 동적분석기 완료!")