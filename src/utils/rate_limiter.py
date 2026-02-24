"""
Rate Limiter - Protection against API quota exhaustion.

Provides rate limiting with exponential backoff for LLM API calls.
"""

import asyncio
import functools
import logging
import random
import re
import time
from collections import deque
from contextlib import contextmanager
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def extract_retry_delay(error_message: str) -> float | None:
    """
    Extract retryDelay from Google API 429 error message.

    Google returns errors like:
        'retryDelay': '38s'
        'retryDelay': '927.819932ms'
        'Please retry in 38.057921181s.'

    Returns:
        Delay in seconds, or None if not found
    """
    # Pattern 1: 'retryDelay': 'Xs' or 'retryDelay': 'Xms'
    match = re.search(r"'retryDelay':\s*'([\d.]+)(s|ms)'", error_message)
    if match:
        value = float(match.group(1))
        unit = match.group(2)
        return value / 1000 if unit == "ms" else value

    # Pattern 2: 'Please retry in Xs.'
    match = re.search(r"Please retry in ([\d.]+)s", error_message)
    if match:
        return float(match.group(1))

    return None


class RateLimiter:
    """
    Rate limiter with sliding window and exponential backoff.

    Designed to respect API rate limits (e.g., Gemini free tier: 15 req/min).

    Usage:
        limiter = RateLimiter(requests_per_minute=15)

        for item in items:
            with limiter.throttle():
                result = call_api(item)
    """

    def __init__(
        self,
        requests_per_minute: int = 15,
        requests_per_day: int = 1500,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: bool = True,
        state_path: str | None = None,
    ):
        """
        Initialize the rate limiter.

        Args:
            requests_per_minute: Maximum requests allowed per minute
            requests_per_day: Maximum requests allowed per day
            max_retries: Maximum retry attempts on rate limit errors
            base_delay: Base delay in seconds for exponential backoff
            max_delay: Maximum delay in seconds
            jitter: Whether to add random jitter to delays
            state_path: Path to JSON file for persisting state
        """
        self.requests_per_minute = requests_per_minute
        self.requests_per_day = requests_per_day
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self.state_path = state_path

        # Sliding windows
        self._timestamps: deque[float] = deque()
        self._window_seconds = 60.0

        # Daily tracking
        self._daily_timestamps: deque[float] = deque()
        self._day_seconds = 24 * 3600.0

        # Stats - session vs historical
        self.session_requests = 0  # Requests in this session only
        self.total_requests = 0  # Total including loaded from state
        self.total_waits = 0
        self.total_wait_time = 0.0
        self._loaded_daily_count = 0  # Daily requests loaded from state

        # Load existing state if provided
        if self.state_path:
            self._load_state()

    def _clean_old_timestamps(self) -> None:
        """Remove timestamps outside the sliding windows."""
        now = time.time()

        # Minute window
        cutoff_min = now - self._window_seconds
        while self._timestamps and self._timestamps[0] < cutoff_min:
            self._timestamps.popleft()

        # Daily window
        cutoff_day = now - self._day_seconds
        while self._daily_timestamps and self._daily_timestamps[0] < cutoff_day:
            self._daily_timestamps.popleft()

    def _calculate_wait_time(self) -> float:
        """Calculate how long to wait before next request."""
        self._clean_old_timestamps()

        if len(self._timestamps) < self.requests_per_minute:
            return 0.0

        # Wait until oldest request exits the window
        oldest = self._timestamps[0]
        wait_until = oldest + self._window_seconds
        wait_time = max(0.0, wait_until - time.time())

        return wait_time

    def _calculate_per_day_wait(self) -> float:
        """Calculate wait time for daily limit."""
        if len(self._daily_timestamps) < self.requests_per_day:
            return 0.0

        # This is rare but possible in Free Tiers
        oldest = self._daily_timestamps[0]
        wait_until = oldest + self._day_seconds
        return max(0.0, wait_until - time.time())

    def wait_if_needed(self) -> float:
        """
        Block if rate limit would be exceeded.

        Returns:
            Time waited in seconds

        Raises:
            DailyQuotaExhaustedError: If daily quota is exhausted (wait > 1 hour)
        """
        # Check minute limit
        wait_time = self._calculate_wait_time()

        # Check daily limit if minute limit is OK
        if wait_time <= 0:
            day_wait = self._calculate_per_day_wait()
            if day_wait > 0:
                # If wait time > 1 hour, it's daily quota - stop immediately
                if day_wait > 3600:
                    from src.llm.provider import DailyQuotaExhaustedError

                    hours_to_wait = day_wait / 3600
                    logger.error("\n" + "=" * 60)
                    logger.error("DAILY QUOTA EXHAUSTED - Cannot continue")
                    logger.error(f"Would need to wait {hours_to_wait:.1f} hours")
                    logger.error("Please wait until tomorrow or upgrade your plan.")
                    logger.error("=" * 60 + "\n")
                    raise DailyQuotaExhaustedError(
                        f"Daily quota exhausted. Would need to wait {hours_to_wait:.1f} hours."
                    )
                logger.warning("DAILY QUOTA REACHED: waiting %.2fs before next request", day_wait)
                wait_time = day_wait

        if wait_time > 0:
            logger.debug(
                "Rate limit: waiting %.2fs (current: %d/%d req/min, %d/%d req/day)",
                wait_time,
                len(self._timestamps),
                self.requests_per_minute,
                len(self._daily_timestamps),
                self.requests_per_day,
            )
            self.total_waits += 1
            self.total_wait_time += wait_time
            time.sleep(wait_time)

        # Record this request
        now = time.time()
        self._timestamps.append(now)
        self._daily_timestamps.append(now)
        self.session_requests += 1
        self.total_requests += 1

        # Persist state
        if self.state_path:
            self._save_state()

        return wait_time

    def _save_state(self) -> None:
        """Save current timestamps and quota info to disk."""
        import json
        from pathlib import Path
        from datetime import datetime

        try:
            # Calculate current quota usage
            self._clean_old_timestamps()
            minute_used = len(self._timestamps)
            daily_used = len(self._daily_timestamps)

            state = {
                "minute_timestamps": list(self._timestamps),
                "daily_timestamps": list(self._daily_timestamps),
                "total_requests": self.total_requests,
                "quota_info": {
                    "minute_limit": self.requests_per_minute,
                    "minute_used": minute_used,
                    "minute_remaining": max(0, self.requests_per_minute - minute_used),
                    "daily_limit": self.requests_per_day,
                    "daily_used": daily_used,
                    "daily_remaining": max(0, self.requests_per_day - daily_used),
                },
                "session_requests": self.session_requests,
                "last_updated": datetime.now().isoformat(),
            }
            path = Path(self.state_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save rate limiter state: {e}")

    def _load_state(self) -> None:
        """Load timestamps from disk."""
        import json
        from pathlib import Path

        path = Path(self.state_path)
        if not path.exists():
            return

        try:
            with open(path, "r") as f:
                state = json.load(f)

            # Restore and clean (remove expired)
            self._timestamps = deque(state.get("minute_timestamps", []))
            self._daily_timestamps = deque(state.get("daily_timestamps", []))

            # Clean expired timestamps immediately
            self._clean_old_timestamps()

            # Track loaded daily count for stats
            self._loaded_daily_count = len(self._daily_timestamps)

            logger.info(
                "Rate limiter state loaded: %d requests in daily window, %d in minute window",
                self._loaded_daily_count,
                len(self._timestamps),
            )
        except Exception as e:
            logger.error(f"Failed to load rate limiter state: {e}")
            # Don't fail execution, just start fresh

    @contextmanager
    def throttle(self):
        """
        Context manager for rate-limited operations.

        Usage:
            with limiter.throttle():
                result = api_call()
        """
        self.wait_if_needed()
        yield

    def handle_rate_limit_error(self, error: Exception) -> float:
        """
        Handle a rate limit (429) error by extracting and using Google's suggested delay.

        Args:
            error: The exception (typically ClientError with status 429)

        Returns:
            The delay time waited in seconds
        """
        error_str = str(error)

        # Try to extract Google's suggested delay
        suggested_delay = extract_retry_delay(error_str)

        if suggested_delay:
            # Use Google's suggested delay + small buffer
            delay = suggested_delay + 1.0
            logger.warning(
                "Rate limit hit. Using Google's suggested delay: %.2fs (+ 1s buffer)",
                suggested_delay,
            )
        else:
            # Fallback to exponential backoff
            delay = self.calculate_backoff(self.total_waits)
            logger.warning("Rate limit hit. Using exponential backoff: %.2fs", delay)

        self.total_waits += 1
        self.total_wait_time += delay
        time.sleep(delay)

        return delay

    def calculate_backoff(self, attempt: int) -> float:
        """
        Calculate exponential backoff delay.

        Args:
            attempt: Retry attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        delay = self.base_delay * (2**attempt)
        delay = min(delay, self.max_delay)

        if self.jitter:
            # Add 0-50% random jitter
            delay *= 1 + random.random() * 0.5

        return delay

    def with_retry(
        self,
        func: Callable[..., T],
        *args: Any,
        retry_on: tuple[type[Exception], ...] = (Exception,),
        **kwargs: Any,
    ) -> T:
        """
        Execute function with rate limiting and retry logic.

        Args:
            func: Function to execute
            *args: Positional arguments for func
            retry_on: Exception types to retry on
            **kwargs: Keyword arguments for func

        Returns:
            Result from func

        Raises:
            Last exception if all retries fail
        """
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                self.wait_if_needed()
                return func(*args, **kwargs)
            except retry_on as e:
                last_exception = e

                if attempt < self.max_retries:
                    delay = self.calculate_backoff(attempt)
                    logger.warning(
                        "Request failed (attempt %d/%d): %s. Retrying in %.2fs",
                        attempt + 1,
                        self.max_retries + 1,
                        str(e),
                        delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "Request failed after %d attempts: %s",
                        self.max_retries + 1,
                        str(e),
                    )

        raise last_exception  # type: ignore

    async def with_retry_async(
        self,
        func: Callable[..., T],
        *args: Any,
        retry_on: tuple[type[Exception], ...] = (Exception,),
        **kwargs: Any,
    ) -> T:
        """
        Async version of with_retry.
        """
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                # Async wait
                wait_time = self._calculate_wait_time()
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                self._timestamps.append(time.time())
                self.total_requests += 1

                # Call function (handle both sync and async)
                result = func(*args, **kwargs)
                if asyncio.iscoroutine(result):
                    return await result
                return result

            except retry_on as e:
                last_exception = e

                if attempt < self.max_retries:
                    delay = self.calculate_backoff(attempt)
                    logger.warning(
                        "Async request failed (attempt %d/%d): %s. Retrying in %.2fs",
                        attempt + 1,
                        self.max_retries + 1,
                        str(e),
                        delay,
                    )
                    await asyncio.sleep(delay)

        raise last_exception  # type: ignore

    def get_stats(self) -> dict[str, Any]:
        """Get rate limiter statistics."""
        self._clean_old_timestamps()  # Ensure fresh counts
        return {
            "session_requests": self.session_requests,
            "total_requests": self.total_requests,
            "total_waits": self.total_waits,
            "total_wait_time": round(self.total_wait_time, 2),
            "average_wait": round(self.total_wait_time / max(1, self.total_waits), 2),
            "current_minute_requests": len(self._timestamps),
            "current_daily_requests": len(self._daily_timestamps),
            "requests_per_minute_limit": self.requests_per_minute,
            "requests_per_day_limit": self.requests_per_day,
        }

    def get_quota_info(self) -> dict[str, Any]:
        """
        Get current quota usage information.

        Returns dict with:
            minute_used: requests in current minute window
            minute_limit: max requests per minute
            minute_remaining: remaining requests this minute
            daily_used: requests in current 24h window
            daily_limit: max requests per day
            daily_remaining: remaining requests today
        """
        self._clean_old_timestamps()

        minute_used = len(self._timestamps)
        daily_used = len(self._daily_timestamps)

        return {
            "minute_used": minute_used,
            "minute_limit": self.requests_per_minute,
            "minute_remaining": max(0, self.requests_per_minute - minute_used),
            "daily_used": daily_used,
            "daily_limit": self.requests_per_day,
            "daily_remaining": max(0, self.requests_per_day - daily_used),
        }

    def reset(self) -> None:
        """Reset the rate limiter state."""
        self._timestamps.clear()
        self._daily_timestamps.clear()
        self.session_requests = 0
        self.total_requests = 0
        self.total_waits = 0
        self.total_wait_time = 0.0
        self._loaded_daily_count = 0

        # Clear persisted state if exists
        if self.state_path:
            self._save_state()


def rate_limit(
    requests_per_minute: int = 15,
    max_retries: int = 3,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for rate-limited functions.

    Usage:
        @rate_limit(requests_per_minute=15)
        def call_api(data):
            return api.request(data)
    """
    limiter = RateLimiter(
        requests_per_minute=requests_per_minute,
        max_retries=max_retries,
    )

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return limiter.with_retry(func, *args, **kwargs)

        # Attach limiter for inspection
        wrapper._rate_limiter = limiter  # type: ignore
        return wrapper

    return decorator
