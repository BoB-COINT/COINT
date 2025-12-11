# 예: modules/honeypot_DA/utils/tx_helpers.py 같은 데에
from brownie.exceptions import TransactionError

def safe_transact(fn, *args, retries=2, **kwargs):
    """
    Brownie 트랜잭션용 안전 래퍼.
    - Tx dropped면 지정 횟수만큼 재시도
    - 끝까지 안되면 TransactionError 그대로 다시 올림
    """
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return fn(*args, **kwargs)
        except TransactionError as e:
            print(f"[safe_transact] attempt {attempt+1} failed: {e}")
            last_exc = e
    # 여기까지 왔다는 건 retries번 다 실패
    raise last_exc
