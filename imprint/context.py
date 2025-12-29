"""Context propagation for distributed tracing."""

import re
from contextvars import ContextVar
from typing import Dict, Optional, Tuple

from .span import Span

# Context variable for current span
_current_span: ContextVar[Optional[Span]] = ContextVar("current_span", default=None)

# W3C traceparent pattern: 00-{32 hex}-{16 hex}-{2 hex}
TRACEPARENT_PATTERN = re.compile(r"^00-([a-f0-9]{32})-([a-f0-9]{16})-([a-f0-9]{2})$")


def get_current_span() -> Optional[Span]:
    """Get the current span from context."""
    return _current_span.get()


def set_current_span(span: Optional[Span]) -> None:
    """Set the current span in context."""
    _current_span.set(span)


class SpanContext:
    """Context manager for setting current span and finishing span on exit."""

    def __init__(self, span: Span):
        self.span = span
        self._token = None

    def __enter__(self) -> Span:
        self._token = _current_span.set(self.span)
        return self.span

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Record exception if one occurred
        if exc_val:
            self.span.record_error(exc_val)
            if self.span.status_code == 0:
                self.span.status_code = 500

        # Finish the span
        self.span.finish()

        # Reset context
        if self._token:
            _current_span.reset(self._token)


def extract_traceparent(header: Optional[str]) -> Optional[Tuple[str, str, bool]]:
    """
    Extract trace context from W3C traceparent header.

    Returns: (trace_id, parent_span_id, sampled) or None if invalid.
    """
    if not header:
        return None

    match = TRACEPARENT_PATTERN.match(header.lower())
    if not match:
        return None

    trace_id = match.group(1)
    span_id = match.group(2)
    flags = match.group(3)
    sampled = (int(flags, 16) & 0x01) == 1

    # Validate trace_id and span_id are not all zeros
    if trace_id == "0" * 32 or span_id == "0" * 16:
        return None

    return trace_id, span_id, sampled


def inject_traceparent(span: Span) -> str:
    """Generate W3C traceparent header value from span."""
    return span.traceparent()


def extract_from_headers(headers: Dict[str, str]) -> Optional[Tuple[str, str, bool]]:
    """Extract trace context from HTTP headers dict."""
    # Try common header name formats
    for key in ["traceparent", "Traceparent", "TRACEPARENT", "HTTP_TRACEPARENT"]:
        if key in headers:
            return extract_traceparent(headers[key])
    return None


def inject_to_headers(span: Span, headers: Dict[str, str]) -> Dict[str, str]:
    """Inject trace context into HTTP headers dict."""
    headers["traceparent"] = inject_traceparent(span)
    return headers
