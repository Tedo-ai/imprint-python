"""Span representation for distributed tracing."""

import os
import secrets
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .client import Client

SDK_NAME = "imprint-python"
SDK_VERSION = "0.1.0"
SDK_LANGUAGE = "python"


def generate_trace_id() -> str:
    """Generate a 32-character hex trace ID."""
    return secrets.token_hex(16)


def generate_span_id() -> str:
    """Generate a 16-character hex span ID."""
    return secrets.token_hex(8)


class Span:
    """A single span in a distributed trace."""

    def __init__(
        self,
        trace_id: str,
        span_id: str,
        name: str,
        namespace: str,
        kind: str = "internal",
        parent_id: Optional[str] = None,
        client: Optional["Client"] = None,
        sampled: bool = True,
        start_time_ns: Optional[int] = None,
    ):
        self.trace_id = trace_id
        self.span_id = span_id
        self.parent_id = parent_id
        self.namespace = namespace
        self.name = name
        self.kind = kind  # "server", "client", "internal", "consumer", "event"

        # Support explicit start time for reconstructing spans (e.g., from job metadata)
        if start_time_ns is not None:
            self.start_time = datetime.fromtimestamp(start_time_ns / 1_000_000_000, tz=timezone.utc)
            self._start_monotonic = None  # No monotonic reference when using explicit time
        else:
            self.start_time = datetime.now(timezone.utc)
            self._start_monotonic = time.perf_counter()
        self.duration_ns: Optional[int] = None

        self.status_code: int = 0
        self.error_data: Optional[str] = None
        self.attributes: Dict[str, str] = {}

        self._client = client
        self._sampled = sampled
        self._promoted = False
        self._ended = False
        self._lock = threading.Lock()

    def set_attribute(self, key: str, value: Any) -> "Span":
        """Set an attribute on this span."""
        with self._lock:
            self.attributes[key] = str(value)
        return self

    def set_status(self, code: int) -> "Span":
        """Set the status code."""
        with self._lock:
            self.status_code = code
        return self

    def record_error(self, error: Optional[Exception] = None, message: Optional[str] = None) -> "Span":
        """Record an error on this span."""
        with self._lock:
            if error:
                self.error_data = f"{type(error).__name__}: {str(error)}"
            elif message:
                self.error_data = message

            # Promote unsampled spans with errors (tail-based sampling)
            if not self._sampled:
                self._promoted = True
                self.attributes["imprint.sampling.promoted"] = "true"
        return self

    def finish(self, end_time_ns: Optional[int] = None) -> None:
        """End the span and queue it for sending.

        Args:
            end_time_ns: Optional explicit end time in nanoseconds since epoch.
                         If not provided, uses current time (via monotonic clock if available).
        """
        with self._lock:
            if self._ended:
                return
            self._ended = True

            if end_time_ns is not None:
                # Explicit end time - calculate duration from start_time
                start_ns = int(self.start_time.timestamp() * 1_000_000_000)
                self.duration_ns = end_time_ns - start_ns
            elif self._start_monotonic is not None:
                # Calculate duration using monotonic clock (normal case)
                elapsed = time.perf_counter() - self._start_monotonic
                self.duration_ns = int(elapsed * 1_000_000_000)
            else:
                # Fallback: use wall clock (less accurate)
                self.duration_ns = 0

        # Queue span for sending
        if self._client and (self._sampled or self._promoted):
            self._client._queue_span(self)

    def end(self, end_time_ns: Optional[int] = None) -> None:
        """Alias for finish() for more intuitive API."""
        self.finish(end_time_ns=end_time_ns)

    def __enter__(self) -> "Span":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_val:
            self.record_error(exc_val)
            if self.status_code == 0:
                self.status_code = 500
        self.finish()

    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary for JSON serialization."""
        # Merge SDK metadata into attributes
        attrs = {
            **self.attributes,
            "telemetry.sdk.name": SDK_NAME,
            "telemetry.sdk.version": SDK_VERSION,
            "telemetry.sdk.language": SDK_LANGUAGE,
            "service.instance.id": os.getenv("HOSTNAME", os.getenv("HOST", "unknown")),
        }

        data = {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "namespace": self.namespace,
            "name": self.name,
            "kind": self.kind,
            "start_time": self.start_time.isoformat().replace("+00:00", "Z"),
            "duration_ns": self.duration_ns or 0,
            "status_code": self.status_code,
            "attributes": attrs,
        }

        if self.parent_id:
            data["parent_id"] = self.parent_id
        if self.error_data:
            data["error_data"] = self.error_data

        return data

    def traceparent(self) -> str:
        """Generate W3C traceparent header value."""
        sampled = "01" if self._sampled else "00"
        return f"00-{self.trace_id}-{self.span_id}-{sampled}"
