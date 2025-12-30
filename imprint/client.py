"""Imprint client for sending spans to the ingest service."""

import atexit
import json
import logging
import threading
import time
from queue import Empty, Full, Queue
from typing import Any, Dict, Optional, Tuple
from urllib.request import Request, urlopen
from urllib.error import URLError

from .config import Config
from .context import SpanContext, extract_from_headers, get_current_span
from .sampler import Sampler, create_sampler
from .span import Span, generate_span_id, generate_trace_id

logger = logging.getLogger("imprint")


class Client:
    """Imprint client for distributed tracing."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._sampler = create_sampler(self.config.sampling_rate)

        # Span buffer
        self._buffer: Queue[Span] = Queue(maxsize=self.config.buffer_size)
        self._shutdown = threading.Event()

        # Start background worker
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

        # Register shutdown handler
        atexit.register(self.shutdown)

    def set_sampler(self, sampler: Sampler) -> None:
        """Set a custom sampler."""
        self._sampler = sampler

    def start_span(
        self,
        name: str,
        kind: str = "internal",
        parent: Optional[Span] = None,
        trace_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        start_time: Optional[int] = None,
    ) -> Tuple[SpanContext, Span]:
        """
        Start a new span.

        Args:
            name: Operation name
            kind: Span kind (server, client, internal, consumer, event)
            parent: Parent span (if not provided, uses current span from context)
            trace_id: Explicit trace ID (for continuing a trace from headers)
            parent_id: Explicit parent span ID
            attributes: Initial attributes
            start_time: Explicit start time in nanoseconds since epoch (for reconstructing spans)

        Returns:
            Tuple of (SpanContext for with statement, Span)
        """
        # Determine trace context
        if parent is None:
            parent = get_current_span()

        if trace_id:
            # Continuing from extracted trace context
            sampled = self._sampler.should_sample(trace_id)
        elif parent:
            # Child span inherits from parent
            trace_id = parent.trace_id
            parent_id = parent_id or parent.span_id
            sampled = parent._sampled
        else:
            # New root span
            trace_id = generate_trace_id()
            sampled = self._sampler.should_sample(trace_id)

        span = Span(
            trace_id=trace_id,
            span_id=generate_span_id(),
            name=name,
            namespace=self.config.service_name,
            kind=kind,
            parent_id=parent_id,
            client=self,
            sampled=sampled,
            start_time_ns=start_time,
        )

        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)

        return SpanContext(span), span

    def start_span_from_headers(
        self,
        name: str,
        headers: Dict[str, str],
        kind: str = "server",
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Tuple[SpanContext, Span]:
        """
        Start a span from incoming HTTP headers (W3C traceparent).

        Args:
            name: Operation name
            headers: HTTP headers dict
            kind: Span kind
            attributes: Initial attributes

        Returns:
            Tuple of (SpanContext, Span)
        """
        context = extract_from_headers(headers)

        if context:
            trace_id, parent_id, _ = context
            return self.start_span(
                name=name,
                kind=kind,
                trace_id=trace_id,
                parent_id=parent_id,
                attributes=attributes,
            )
        else:
            return self.start_span(name=name, kind=kind, attributes=attributes)

    def record_event(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a point-in-time event as a zero-duration span."""
        ctx, span = self.start_span(name, kind="event", attributes=attributes)
        with ctx:
            span.duration_ns = 0

    def _queue_span(self, span: Span) -> None:
        """Queue a span for sending (non-blocking)."""
        if not self.config.enabled:
            return

        try:
            self._buffer.put_nowait(span)
        except Full:
            if self.config.debug:
                logger.warning("Span buffer full, dropping span")

    def _worker(self) -> None:
        """Background worker that flushes spans periodically."""
        last_flush = time.time()

        while not self._shutdown.is_set():
            try:
                # Check if we should flush
                now = time.time()
                buffer_size = self._buffer.qsize()
                time_elapsed = now - last_flush >= self.config.flush_interval
                size_exceeded = buffer_size >= self.config.batch_size

                if (time_elapsed or size_exceeded) and buffer_size > 0:
                    self._flush()
                    last_flush = time.time()

                # Short sleep to avoid busy-waiting
                time.sleep(0.1)

            except Exception as e:
                if self.config.debug:
                    logger.exception("Error in worker thread")

    def _flush(self) -> None:
        """Flush buffered spans to the ingest service."""
        spans = []

        # Drain buffer up to batch_size
        while len(spans) < self.config.batch_size:
            try:
                span = self._buffer.get_nowait()
                spans.append(span)
            except Empty:
                break

        if not spans:
            return

        self._send_batch(spans)

    def _send_batch(self, spans: list) -> None:
        """Send a batch of spans to the ingest service."""
        if not self.config.api_key:
            if self.config.debug:
                logger.warning("No API key configured, dropping spans")
            return

        try:
            data = json.dumps([s.to_dict() for s in spans]).encode("utf-8")

            request = Request(
                self.config.ingest_url,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.config.api_key}",
                },
                method="POST",
            )

            with urlopen(request, timeout=5) as response:
                if self.config.debug:
                    logger.debug(f"Sent {len(spans)} spans, status: {response.status}")

        except URLError as e:
            if self.config.debug:
                logger.warning(f"Failed to send spans: {e}")
        except Exception as e:
            if self.config.debug:
                logger.exception("Error sending spans")

    def shutdown(self, timeout: float = 5.0) -> None:
        """Shutdown the client and flush remaining spans."""
        self._shutdown.set()

        # Wait for worker to finish
        self._worker_thread.join(timeout=timeout)

        # Final flush
        self._flush()


# Global client instance
_client: Optional[Client] = None


def get_client() -> Optional[Client]:
    """Get the global client instance."""
    return _client


def init(config: Optional[Config] = None, **kwargs) -> Client:
    """
    Initialize the global Imprint client.

    Args:
        config: Config instance, or pass individual config options as kwargs

    Returns:
        The initialized Client
    """
    global _client

    if config is None:
        config = Config(**kwargs)

    _client = Client(config)
    return _client


def shutdown(timeout: float = 5.0) -> None:
    """Shutdown the global client."""
    global _client
    if _client:
        _client.shutdown(timeout)
        _client = None
