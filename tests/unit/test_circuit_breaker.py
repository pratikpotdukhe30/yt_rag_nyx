import pytest

from app.resilience.circuit_breaker import CircuitBreaker, CircuitOpenError, State


@pytest.fixture
def clock(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("app.resilience.circuit_breaker.time.monotonic", lambda: now[0])
    return now


def test_opens_after_threshold(clock):
    b = CircuitBreaker("t", failure_threshold=3, recovery_seconds=30)
    for _ in range(3):
        b.before_call()
        b.record_failure()
    assert b.state is State.OPEN
    with pytest.raises(CircuitOpenError):
        b.before_call()                       # fails fast, never touches the dependency


def test_half_open_probe_success_closes(clock):
    b = CircuitBreaker("t", failure_threshold=1, recovery_seconds=30)
    b.record_failure()
    clock[0] = 31                             # cooldown passes instantly
    b.before_call()
    assert b.state is State.HALF_OPEN
    b.record_success()
    assert b.state is State.CLOSED


def test_half_open_probe_failure_reopens(clock):
    b = CircuitBreaker("t", failure_threshold=1, recovery_seconds=30)
    b.record_failure()
    clock[0] = 31
    b.before_call()
    b.record_failure()
    assert b.state is State.OPEN