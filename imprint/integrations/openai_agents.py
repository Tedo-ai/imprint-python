"""
Imprint integration for OpenAI Agents SDK.

Provides automatic tracing of OpenAI agent runs, including:
- Agent execution spans
- LLM calls with token usage and cost tracking
- Tool/function calls
- Handoffs between agents

Usage:
    import imprint
    from imprint.integrations.openai_agents import setup_tracing

    # Initialize Imprint
    imprint.init(api_key="imp_live_xxx", service_name="my-agent")

    # Enable OpenAI agents tracing
    setup_tracing()

    # Run your agents as normal
    result = await Runner.run(agent, "Hello!")
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import imprint
from imprint import Span as ImprintSpan
from imprint.llm import (
    LLMSpan,
    MODEL_PRICING,
    get_provider,
    calculate_cost,
    ATTR_LLM_MODEL,
    ATTR_LLM_SYSTEM,
)


class ImprintTracingProcessor:
    """
    TracingProcessor that sends OpenAI agent traces to Imprint.

    Maps OpenAI SDK traces/spans to Imprint's span model with full
    LLM observability including tokens, costs, and caching metrics.
    """

    def __init__(self):
        self._trace_spans: Dict[str, ImprintSpan] = {}
        self._span_map: Dict[str, ImprintSpan] = {}

    def on_trace_start(self, trace) -> None:
        """Called when an OpenAI agent trace starts."""
        ctx, span = imprint.get_client().start_span(
            name=f"trace:{trace.name}",
            kind="internal",
            trace_id=self._normalize_trace_id(trace.trace_id),
            attributes={
                "openai.trace_id": trace.trace_id,
                "openai.workflow_name": trace.name,
            },
        )
        self._trace_spans[trace.trace_id] = span

    def on_trace_end(self, trace) -> None:
        """Called when an OpenAI agent trace ends."""
        span = self._trace_spans.pop(trace.trace_id, None)
        if span:
            span.finish()

    def on_span_start(self, span) -> None:
        """Called when an OpenAI agent span starts."""
        parent_span = None
        if span.parent_id:
            parent_span = self._span_map.get(span.parent_id)
        if not parent_span:
            parent_span = self._trace_spans.get(span.trace_id)

        span_name = self._get_span_name(span)
        span_kind = self._get_span_kind(span)

        ctx, imprint_span = imprint.get_client().start_span(
            name=span_name,
            kind=span_kind,
            trace_id=self._normalize_trace_id(span.trace_id),
            parent_id=parent_span.span_id if parent_span else None,
            attributes=self._extract_start_attributes(span),
        )
        self._span_map[span.span_id] = imprint_span

    def on_span_end(self, span) -> None:
        """Called when an OpenAI agent span ends."""
        imprint_span = self._span_map.pop(span.span_id, None)
        if imprint_span:
            if span.error:
                # Handle both object and dict error formats
                if hasattr(span.error, 'message'):
                    imprint_span.record_error(message=span.error.message)
                elif isinstance(span.error, dict) and 'message' in span.error:
                    imprint_span.record_error(message=span.error['message'])
                else:
                    imprint_span.record_error(message=str(span.error))

            # Extract LLM usage (token counts only available at end)
            self._record_llm_usage(span, imprint_span)

            imprint_span.finish()

    def shutdown(self) -> None:
        """Cleanup when application terminates."""
        pass  # Imprint client handles its own shutdown

    def force_flush(self) -> None:
        """Force immediate processing of queued spans."""
        pass

    def _normalize_trace_id(self, openai_trace_id: str) -> str:
        """Normalize OpenAI trace ID to 32-char hex format."""
        clean_id = openai_trace_id.replace("trace_", "")
        if len(clean_id) < 32:
            clean_id = clean_id.ljust(32, "0")
        return clean_id[:32]

    def _get_span_name(self, span) -> str:
        """Extract a meaningful span name from OpenAI span data."""
        span_data = span.span_data
        if span_data is None:
            return "unknown"

        data_type = type(span_data).__name__

        if data_type == "AgentSpanData":
            return f"agent:{getattr(span_data, 'name', 'unknown')}"
        elif data_type == "GenerationSpanData":
            return f"llm:{getattr(span_data, 'model', 'unknown')}"
        elif data_type == "ResponseSpanData":
            return "llm:response"
        elif data_type == "FunctionSpanData":
            return f"tool:{getattr(span_data, 'name', 'unknown')}"
        elif data_type == "HandoffSpanData":
            from_agent = getattr(span_data, 'from_agent', 'unknown')
            to_agent = getattr(span_data, 'to_agent', 'unknown')
            return f"handoff:{from_agent}->{to_agent}"
        elif data_type == "GuardrailSpanData":
            return f"guardrail:{getattr(span_data, 'name', 'unknown')}"
        elif data_type == "MCPListToolsSpanData":
            return "mcp:list_tools"
        elif data_type == "MCPCallToolSpanData":
            return f"mcp:call:{getattr(span_data, 'tool', 'unknown')}"
        else:
            return f"span:{data_type}"

    def _get_span_kind(self, span) -> str:
        """Determine Imprint span kind from OpenAI span type."""
        span_data = span.span_data
        if span_data is None:
            return "internal"

        data_type = type(span_data).__name__

        if data_type in ("GenerationSpanData", "ResponseSpanData"):
            return "client"
        elif data_type.startswith("MCP"):
            return "client"
        else:
            return "internal"

    def _extract_start_attributes(self, span) -> Dict[str, Any]:
        """Extract attributes available at span start."""
        attrs = {
            "openai.span_id": span.span_id,
            "openai.span_type": type(span.span_data).__name__ if span.span_data else "unknown",
        }

        if span.parent_id:
            attrs["openai.parent_id"] = span.parent_id

        span_data = span.span_data
        if span_data is None:
            return attrs

        data_type = type(span_data).__name__

        if data_type == "AgentSpanData":
            attrs["agent.name"] = getattr(span_data, 'name', '')
        elif data_type == "FunctionSpanData":
            attrs["tool.name"] = getattr(span_data, 'name', '')
        elif data_type == "MCPCallToolSpanData":
            attrs["mcp.tool"] = getattr(span_data, 'tool', '')
            attrs["mcp.server"] = getattr(span_data, 'server', '')

        return attrs

    def _record_llm_usage(self, span, imprint_span: ImprintSpan) -> None:
        """Record LLM usage metrics using the core imprint.llm module."""
        span_data = span.span_data
        if span_data is None:
            return

        data_type = type(span_data).__name__
        if data_type not in ("GenerationSpanData", "ResponseSpanData"):
            return

        # Extract model name
        model = getattr(span_data, 'model', None)

        # Extract token counts from various possible locations
        input_tokens = 0
        output_tokens = 0
        cached_tokens = 0
        reasoning_tokens = 0

        # Check for direct attributes
        if hasattr(span_data, 'input_tokens') and span_data.input_tokens:
            input_tokens = span_data.input_tokens
        if hasattr(span_data, 'output_tokens') and span_data.output_tokens:
            output_tokens = span_data.output_tokens

        # Check for usage object
        if hasattr(span_data, 'usage') and span_data.usage:
            usage = span_data.usage
            input_tokens = getattr(usage, 'input_tokens', 0) or getattr(usage, 'prompt_tokens', 0) or input_tokens
            output_tokens = getattr(usage, 'output_tokens', 0) or getattr(usage, 'completion_tokens', 0) or output_tokens

            input_details = getattr(usage, 'input_tokens_details', None)
            if input_details:
                cached_tokens = getattr(input_details, 'cached_tokens', 0) or 0

            output_details = getattr(usage, 'output_tokens_details', None)
            if output_details:
                reasoning_tokens = getattr(output_details, 'reasoning_tokens', 0) or 0

        # Check for response.usage (OpenAI Responses API)
        if hasattr(span_data, 'response') and span_data.response:
            response = span_data.response
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                input_tokens = getattr(usage, 'input_tokens', 0) or input_tokens
                output_tokens = getattr(usage, 'output_tokens', 0) or output_tokens

                input_details = getattr(usage, 'input_tokens_details', None)
                if input_details:
                    cached_tokens = getattr(input_details, 'cached_tokens', 0) or 0

                output_details = getattr(usage, 'output_tokens_details', None)
                if output_details:
                    reasoning_tokens = getattr(output_details, 'reasoning_tokens', 0) or 0

            if hasattr(response, 'model') and response.model:
                model = response.model

        # Use the core LLMSpan helper to record all metrics
        if model or input_tokens > 0 or output_tokens > 0:
            llm_span = LLMSpan(imprint_span)
            if model:
                llm_span.set_model(model)
            llm_span.set_tokens(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_tokens=cached_tokens,
                reasoning_tokens=reasoning_tokens,
            )


def setup_tracing() -> ImprintTracingProcessor:
    """
    Enable Imprint tracing for OpenAI Agents SDK.

    Must call imprint.init() first.

    Returns:
        The ImprintTracingProcessor instance.

    Example:
        import imprint
        from imprint.integrations.openai_agents import setup_tracing

        imprint.init(api_key="imp_live_xxx")
        setup_tracing()

        # Now run your agents
        result = await Runner.run(agent, "Hello!")
    """
    try:
        from agents.tracing import add_trace_processor
    except ImportError:
        raise ImportError(
            "openai-agents package not installed. Install with: pip install openai-agents"
        )

    if imprint.get_client() is None:
        raise RuntimeError("Imprint not initialized. Call imprint.init() first.")

    processor = ImprintTracingProcessor()
    add_trace_processor(processor)
    return processor
