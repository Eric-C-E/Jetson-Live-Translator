# util/timing_helpers.py
import time

class RateLimiter:
    def __init__(self, hz: float):
        self.period = 1.0 / max(hz, 1e-6)
        self.t_next = 0.0

    def allow(self) -> bool:
        now = time.time()
        if now >= self.t_next:
            self.t_next = now + self.period
            return True
        return False