"""
Tests for DailyRequestCounter in app/rate_limit.py.

Save this as: backend/tests/test_daily_counter.py
"""
from datetime import datetime, timedelta
from app.rate_limit import DailyRequestCounter


def test_allows_requests_under_limit():
    counter = DailyRequestCounter(daily_limit=3)
    assert counter.increment_and_check() is True
    assert counter.increment_and_check() is True
    assert counter.increment_and_check() is True


def test_blocks_requests_over_limit():
    counter = DailyRequestCounter(daily_limit=2)
    counter.increment_and_check()
    counter.increment_and_check()
    assert counter.increment_and_check() is False


def test_remaining_counts_down_correctly():
    counter = DailyRequestCounter(daily_limit=5)
    assert counter.remaining() == 5
    counter.increment_and_check()
    assert counter.remaining() == 4
    counter.increment_and_check()
    assert counter.remaining() == 3


def test_remaining_never_goes_negative():
    counter = DailyRequestCounter(daily_limit=1)
    counter.increment_and_check()
    counter.increment_and_check()  # over limit, doesn't increment further
    assert counter.remaining() == 0


def test_resets_on_new_day():
    """_reset_if_new_day() compares against datetime.utcnow().date(), not
    local date — so the rollover must be simulated in UTC too. Using local
    date.today() here would make this test flaky near midnight IST
    (UTC+5:30), where local "today" and UTC "today" can briefly disagree."""
    counter = DailyRequestCounter(daily_limit=1)
    counter.increment_and_check()
    assert counter.remaining() == 0

    # Simulate a day rollover by manually rewinding current_day (in UTC)
    counter.current_day = datetime.utcnow().date() - timedelta(days=1)
    assert counter.remaining() == 1  # should have reset
    assert counter.increment_and_check() is True
