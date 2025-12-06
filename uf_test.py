"""
Test driver for Unformed LP detection (DB-backed, single token).

Usage:
    python uf_test.py
"""

from __future__ import annotations

from pprint import pprint

from modules.preprocessor.exit_instance import setup_django
from pipeline.adapters import DetectUnformedLpAdapter


def main() -> None:
    setup_django()

    adapter = DetectUnformedLpAdapter()
    result = adapter.run()

    token_addr = result.get("token_addr")
    is_unformed = bool(result.get("is_unformed_lp"))

    print("============================================")
    print("🧪 Unformed LP Detection Test")
    print("============================================")
    print(f"Token Addr: {token_addr}")
    print(f"Unformed LP?: {'YES' if is_unformed else 'NO'}")

    if is_unformed:
        print("→ Marked ExitMlResult.is_unformed_lp = 1 (others set to NULL)")
    else:
        print("→ No DB update performed for ExitMlResult.")


if __name__ == "__main__":
    main()
