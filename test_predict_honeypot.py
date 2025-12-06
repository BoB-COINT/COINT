# test_predict_honeypot.py
import os
import time

import django

# 1) Django 설정 초기화
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from api.models import HoneypotProcessedData, HoneypotMlResult  # noqa: E402
from pipeline.adapters import HoneypotMLAnalyzerAdapter          # noqa: E402


def main():
    print("=" * 60)
    print("🧪 Honeypot v12 예측 테스트 (Adapter → modules/honeypot_ML/predict_v12)")
    print("=" * 60)

    # 2) 어댑터 생성 (여기서 modules/predict_v12.py를 내부에서 import)
    ml_adapter = HoneypotMLAnalyzerAdapter()

    # 3) 기존 ML 결과 삭제 (테스트용)
    print("🗑️  기존 HoneypotMlResult 삭제 중...")
    deleted, _ = HoneypotMlResult.objects.all().delete()
    print(f"   삭제 완료! (삭제된 행: {deleted})")

    # 4) 예측 대상 로드
    qs = HoneypotProcessedData.objects.select_related("token_info").order_by("token_info_id")
    total = qs.count()
    print(f"\n📊 대상 HoneypotProcessedData 개수: {total}")

    if total == 0:
        print("⚠️ HoneypotProcessedData 에 데이터가 없습니다. 전처리 먼저 돌려주세요.")
        return

    start = time.time()
    success = 0
    failed = 0

    # 5) 한 건씩 adapter.predict() 호출
    for idx, hp in enumerate(qs.iterator(), start=1):
        ti = hp.token_info
        print("\n------------------------------------------------------------")
        print(f"[{idx}/{total}] TokenInfo ID={ti.id}")
        print(f"   Token Addr: {ti.token_addr}")

        try:
            result = ml_adapter.predict(hp)
            print(
                f"   ✅ 예측 완료: "
                f"prob={result['probability']:.4f}, "
                f"risk={result['risk_level']}, "
                f"is_honeypot={result['is_honeypot']}, "
                f"status={result.get('status')}"
            )
            success += 1
        except Exception as e:
            print("   ❌ 예측 실패:", repr(e))
            failed += 1

    elapsed = time.time() - start
    print("\n============================================================")
    print("🏁 Honeypot v12 예측 테스트 종료")
    print(f"   성공: {success}개, 실패: {failed}개")
    print(f"   소요 시간: {elapsed:.2f}초")
    print("============================================================")


if __name__ == "__main__":
    main()
