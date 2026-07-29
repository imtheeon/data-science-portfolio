# tests/conftest.py
import pytest

import api.main as main_module


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """slowapi's in-memory limiter storage otherwise persists across tests
    within the same process, which can cause unrelated tests to fail (or
    pass only by accident of ordering) if pytest ever reorders execution."""
    main_module.limiter.reset()
    yield
    main_module.limiter.reset()
