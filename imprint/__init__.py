"""
Imprint Python SDK for distributed tracing.

Usage:
    import imprint

    # Initialize
    imprint.init(api_key="imp_live_...", service_name="my-service")

    # Create spans
    with imprint.start_span("operation") as span:
        span.set_attribute("key", "value")
        # ... do work ...

    # Shutdown
    imprint.shutdown()
"""

from .config import Config
from .context import (
    SpanContext,
    extract_from_headers,
    extract_traceparent,
    get_current_span,
    inject_to_headers,
    inject_traceparent,
    set_current_span,
)
from .client import Client, get_client, init, shutdown
from .sampler import AlwaysSampler, NeverSampler, RateSampler, Sampler, create_sampler
from .span import Span, generate_span_id, generate_trace_id

__version__ = "0.1.0"

__all__ = [
    # Config
    "Config",
    # Client
    "Client",
    "init",
    "shutdown",
    "get_client",
    # Span
    "Span",
    "generate_trace_id",
    "generate_span_id",
    # Context
    "SpanContext",
    "get_current_span",
    "set_current_span",
    "extract_traceparent",
    "inject_traceparent",
    "extract_from_headers",
    "inject_to_headers",
    # Sampler
    "Sampler",
    "AlwaysSampler",
    "NeverSampler",
    "RateSampler",
    "create_sampler",
]


def start_span(name: str, **kwargs):
    """
    Start a new span using the global client.

    Convenience function that returns just the SpanContext for use with `with`.
    """
    client = get_client()
    if client is None:
        raise RuntimeError("Imprint not initialized. Call imprint.init() first.")
    ctx, span = client.start_span(name, **kwargs)
    return ctx


def current_trace_id() -> str | None:
    """Get the current trace ID."""
    span = get_current_span()
    return span.trace_id if span else None


def current_span_id() -> str | None:
    """Get the current span ID."""
    span = get_current_span()
    return span.span_id if span else None
