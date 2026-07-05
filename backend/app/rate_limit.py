"""
Shared rate-limiting state — kept in its own module so both main.py and
predict.py can import from it without a circular import (main.py includes
the predict router; predict.py needs the limiter instance).

Save this as: app/rate_limit.py
"""
import os
import datetime
from slowapi import Limiter
from slowapi.util import get_remote_address

# ── Per-IP limiter ────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)


# ── Global daily cap ──────────────────────────────────────
# Protects the shared Groq LLM API budget specifically — every
# /api/predict call triggers a paid LLM request regardless of which IP
# sends it, so this is tracked separately from the per-IP limiter above.
class DailyRequestCounter:
    def __init__(self, daily_limit: int):
        self.daily_limit = daily_limit
        self.count = 0
        self.current_day = None

    def _reset_if_new_day(self):
        today = datetime.datetime.utcnow().date()
        if self.current_day != today:
            self.current_day = today
            self.count = 0

    def increment_and_check(self) -> bool:
        """Returns True if under the limit (and increments), False if limit reached."""
        self._reset_if_new_day()
        if self.count >= self.daily_limit:
            return False
        self.count += 1
        return True

    def remaining(self) -> int:
        self._reset_if_new_day()
        return max(0, self.daily_limit - self.count)


DAILY_REQUEST_LIMIT = int(os.getenv('DAILY_REQUEST_LIMIT', '200'))
daily_counter = DailyRequestCounter(daily_limit=DAILY_REQUEST_LIMIT)
