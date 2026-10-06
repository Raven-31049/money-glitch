"""Smoke test: the package imports and the toolchain runs.

The suite is empty by design at Phase 0, but it must execute green from day one
so a regression in the tooling itself is visible immediately rather than being
confused with a real failure later.
"""


def test_package_importable() -> None:
    import bet_engine

    assert bet_engine.__name__ == "bet_engine"
