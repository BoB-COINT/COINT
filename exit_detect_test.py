"""
Test driver for DB-based exit MIL inference.

Usage:
    python exit_detect_test.py
"""

from __future__ import annotations

import json

from pipeline.adapters import ExitMLAnalyzerAdapter


def _resolve_token_id() -> int:
    from modules.preprocessor.exit_instance import setup_django

    setup_django()
    from api.models import TokenInfo

    qs = TokenInfo.objects.order_by("id")
    count = qs.count()
    if count == 0:
        raise SystemExit("No TokenInfo rows found.")
    if count > 1:
        raise SystemExit("Multiple TokenInfo rows found; this helper assumes a single TokenInfo.")
    return qs.first().id


def main() -> None:
    token_info_id = _resolve_token_id()
    from api.models import TokenInfo

    adapter = ExitMLAnalyzerAdapter()
    token = TokenInfo.objects.get(id=token_info_id)
    result = adapter.run(token, save_to_db=True)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
