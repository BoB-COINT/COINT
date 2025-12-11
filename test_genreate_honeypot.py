# test_generate_honeypot.py
import os
import django
import time

# ✅ 1. Django 설정 로드
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from pipeline.adapters import PreprocessorAdapter
from api.models import TokenInfo, HoneypotProcessedData


def main():
    print("============================================================")
    print("🧪 Honeypot 전처리 테스트 (PreprocessorAdapter → modules)")
    print("============================================================")

    # ✅ 2. Adapter 생성
    preprocessor = PreprocessorAdapter()

    # ⚠ 필요하면 기존 데이터 날리고 새로 테스트
    print("🗑️  기존 HoneypotProcessedData 삭제 중...")
    deleted_count, _ = HoneypotProcessedData.objects.all().delete()
    print(f"   삭제 완료! (삭제된 행: {deleted_count})")

    # ✅ 3. 전처리 대상 TokenInfo 선택
    token_qs = TokenInfo.objects.all().order_by("id")
    total_tokens = token_qs.count()
    print(f"\n📊 전처리 대상 TokenInfo 개수: {total_tokens}")

    if total_tokens == 0:
        print("⚠️ TokenInfo 테이블에 데이터가 없습니다. collector 먼저 돌려주세요.")
        return

    start_time = time.time()
    success = 0
    failed = 0

    # ✅ 4. 토큰별로 전처리 수행
    for idx, token in enumerate(token_qs.iterator(), start=1):
        print("\n------------------------------------------------------------")
        print(f"[{idx}/{total_tokens}] Token ID={token.id}")
        print(f"   Token Addr: {token.token_addr}")

        try:
            # 4-1) Adapter → modules.generate_features_honeypot 호출
            features_data = preprocessor.process_for_honeypot(token)

            # 4-2) HoneypotProcessedData DB 저장
            preprocessor.save_honeypot_to_db(token, features_data)

            success += 1
            print("   ✅ 전처리 + DB 저장 완료")

        except Exception as e:
            failed += 1
            print("   ❌ 전처리 실패:", repr(e))

    elapsed = time.time() - start_time
    print("\n============================================================")
    print("🏁 전처리 테스트 종료")
    print(f"   성공: {success}개, 실패: {failed}개")
    print(f"   소요 시간: {elapsed:.2f}초")
    print("============================================================")


if __name__ == "__main__":
    main()
