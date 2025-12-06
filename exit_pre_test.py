"""
Run both exit preprocessors (instance + static) for the single TokenInfo in DB.
Assumes there is exactly one TokenInfo row.
"""

from __future__ import annotations

from pprint import pprint

from modules.preprocessor.exit_instance import setup_django
from pipeline.adapters import PreprocessorAdapter


def _resolve_token():
    from api.models import TokenInfo

    qs = TokenInfo.objects.order_by("id")
    count = qs.count()
    if count == 0:
        raise SystemExit("No TokenInfo rows found.")
    if count > 1:
        raise SystemExit("Multiple TokenInfo rows found; this helper assumes a single TokenInfo.")
    return qs.first()


def main() -> None:
    setup_django()

    token = _resolve_token()
    adapter = PreprocessorAdapter()

    saved = adapter.process_exit_instance(token)
    print(f"[instance] generated and saved {saved} rows for token {token.token_addr}")

    adapter.process_exit_static(token)
    print(f"[static] computed and saved static features for token {token.token_addr}")


if __name__ == "__main__":
    main()
