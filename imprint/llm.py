"""
LLM observability schema and utilities for Imprint.

This module defines the standard schema for LLM tracing in Imprint,
including attribute names, model pricing, and cost calculation.

Usage:
    from imprint.llm import LLMSpan, MODEL_PRICING

    # Record LLM metrics on a span
    llm_span = LLMSpan(span)
    llm_span.set_model("gpt-5-mini", provider="openai")
    llm_span.set_tokens(input_tokens=150, output_tokens=50, cached_tokens=100)

    # Or use the convenience method
    span.set_llm_usage(
        model="gpt-5-mini",
        input_tokens=150,
        output_tokens=50,
        cached_tokens=100,
    )
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .span import Span


# Pattern for date suffixes in model names (e.g., -2025-08-07 or -20241022)
_DATE_SUFFIX_PATTERN = re.compile(r"-\d{4}-\d{2}-\d{2}$|-\d{8}$")


# =============================================================================
# LLM Attribute Schema
# =============================================================================

# Provider identification
ATTR_LLM_SYSTEM = "llm.system"  # Provider: openai, anthropic, google, etc.
ATTR_LLM_MODEL = "llm.model"  # Model name: gpt-5-mini, claude-3-opus, etc.

# Token counts (Imprint native format)
ATTR_LLM_TOKENS_INPUT = "llm.tokens_input"
ATTR_LLM_TOKENS_OUTPUT = "llm.tokens_output"
ATTR_LLM_TOKENS_TOTAL = "llm.tokens_total"
ATTR_LLM_TOKENS_CACHED = "llm.tokens_cached"
ATTR_LLM_TOKENS_REASONING = "llm.tokens_reasoning"

# Cost tracking
ATTR_LLM_COST_USD = "llm.cost_usd"
ATTR_LLM_CACHE_SAVINGS_USD = "llm.cache_savings_usd"
ATTR_LLM_CACHE_HIT_RATE = "llm.cache_hit_rate"

# OpenTelemetry GenAI semantic conventions (for compatibility)
ATTR_GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
ATTR_GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
ATTR_GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
ATTR_GEN_AI_USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"
ATTR_GEN_AI_USAGE_CACHED_TOKENS = "gen_ai.usage.cached_tokens"
ATTR_GEN_AI_USAGE_REASONING_TOKENS = "gen_ai.usage.reasoning_tokens"
ATTR_GEN_AI_USAGE_COST = "gen_ai.usage.cost"


# =============================================================================
# Model Pricing (per 1M tokens in USD)
# =============================================================================

MODEL_PRICING: Dict[str, Dict[str, float]] = {
    # OpenAI GPT-5 series
    "gpt-5-mini": {
        "input": 0.25,
        "cached_input": 0.025,
        "output": 2.00,
    },
    "gpt-5": {
        "input": 2.50,
        "cached_input": 0.25,
        "output": 10.00,
    },
    # OpenAI GPT-4.1 series
    "gpt-4.1": {
        "input": 2.00,
        "cached_input": 0.50,
        "output": 8.00,
    },
    "gpt-4.1-mini": {
        "input": 0.40,
        "cached_input": 0.10,
        "output": 1.60,
    },
    "gpt-4.1-nano": {
        "input": 0.10,
        "cached_input": 0.025,
        "output": 0.40,
    },
    # OpenAI reasoning models
    "o3": {
        "input": 10.00,
        "cached_input": 2.50,
        "output": 40.00,
    },
    "o3-mini": {
        "input": 1.10,
        "cached_input": 0.275,
        "output": 4.40,
    },
    "o1": {
        "input": 15.00,
        "cached_input": 7.50,
        "output": 60.00,
    },
    "o1-mini": {
        "input": 1.10,
        "cached_input": 0.55,
        "output": 4.40,
    },
    # Anthropic Claude models
    "claude-3-5-sonnet-20241022": {
        "input": 3.00,
        "cached_input": 0.30,
        "output": 15.00,
    },
    "claude-3-5-haiku-20241022": {
        "input": 0.80,
        "cached_input": 0.08,
        "output": 4.00,
    },
    "claude-3-opus-20240229": {
        "input": 15.00,
        "cached_input": 1.50,
        "output": 75.00,
    },
    "claude-opus-4-5-20251101": {
        "input": 15.00,
        "cached_input": 1.50,
        "output": 75.00,
    },
    "claude-sonnet-4-20250514": {
        "input": 3.00,
        "cached_input": 0.30,
        "output": 15.00,
    },
}


# =============================================================================
# Provider Detection
# =============================================================================

def get_provider(model: str) -> str:
    """
    Determine the provider from a model name.

    Args:
        model: Model name (e.g., "gpt-5-mini", "claude-3-opus")

    Returns:
        Provider name: openai, anthropic, google, meta, mistral, cohere
    """
    model_lower = model.lower()

    if model_lower.startswith(("gpt-", "o1", "o3", "davinci", "curie", "babbage", "ada")):
        return "openai"
    elif model_lower.startswith("claude"):
        return "anthropic"
    elif model_lower.startswith(("gemini", "palm")):
        return "google"
    elif model_lower.startswith("llama"):
        return "meta"
    elif model_lower.startswith("mistral"):
        return "mistral"
    elif model_lower.startswith("command"):
        return "cohere"

    return "unknown"


# =============================================================================
# Cost Calculation
# =============================================================================

def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
) -> Optional[float]:
    """
    Calculate the cost in USD for an LLM call.

    Args:
        model: Model name
        input_tokens: Total input tokens (including cached)
        output_tokens: Output tokens
        cached_tokens: Number of tokens served from cache

    Returns:
        Cost in USD, or None if model pricing is unknown
    """
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        # Try base model name (strip date suffix like -2025-08-07 or -20241022)
        base_model = _DATE_SUFFIX_PATTERN.sub("", model)
        pricing = MODEL_PRICING.get(base_model)

    if not pricing:
        return None

    # Cached tokens are charged at reduced rate
    regular_input = max(0, input_tokens - cached_tokens)
    input_cost = (regular_input / 1_000_000) * pricing["input"]
    cached_cost = (cached_tokens / 1_000_000) * pricing["cached_input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]

    return input_cost + cached_cost + output_cost


# =============================================================================
# LLMSpan Helper Class
# =============================================================================

class LLMSpan:
    """
    Helper class for recording LLM metrics on an Imprint span.

    Provides a fluent interface for setting all LLM-related attributes
    in both Imprint native and OpenTelemetry formats.
    """

    def __init__(self, span: "Span"):
        """
        Wrap an Imprint span with LLM metric helpers.

        Args:
            span: The Imprint span to add LLM attributes to
        """
        self._span = span
        self._model: Optional[str] = None
        self._input_tokens: int = 0
        self._output_tokens: int = 0
        self._cached_tokens: int = 0
        self._reasoning_tokens: int = 0

    def set_model(self, model: str, provider: Optional[str] = None) -> "LLMSpan":
        """
        Set the model name and provider.

        Args:
            model: Model name (e.g., "gpt-5-mini")
            provider: Provider name (auto-detected if not specified)
        """
        self._model = model
        provider = provider or get_provider(model)

        self._span.set_attribute(ATTR_LLM_MODEL, model)
        self._span.set_attribute(ATTR_LLM_SYSTEM, provider)
        self._span.set_attribute(ATTR_GEN_AI_REQUEST_MODEL, model)

        return self

    def set_tokens(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_tokens: int = 0,
        reasoning_tokens: int = 0,
    ) -> "LLMSpan":
        """
        Set token counts and calculate costs.

        Args:
            input_tokens: Total input tokens (including cached)
            output_tokens: Output tokens
            cached_tokens: Tokens served from prompt cache
            reasoning_tokens: Tokens used for reasoning (o1, o3 models)
        """
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._cached_tokens = cached_tokens
        self._reasoning_tokens = reasoning_tokens

        # Set Imprint native attributes
        if input_tokens > 0:
            self._span.set_attribute(ATTR_LLM_TOKENS_INPUT, input_tokens)
            self._span.set_attribute(ATTR_GEN_AI_USAGE_INPUT_TOKENS, input_tokens)

        if output_tokens > 0:
            self._span.set_attribute(ATTR_LLM_TOKENS_OUTPUT, output_tokens)
            self._span.set_attribute(ATTR_GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens)

        total_tokens = input_tokens + output_tokens
        if total_tokens > 0:
            self._span.set_attribute(ATTR_LLM_TOKENS_TOTAL, total_tokens)
            self._span.set_attribute(ATTR_GEN_AI_USAGE_TOTAL_TOKENS, total_tokens)

        if cached_tokens > 0:
            self._span.set_attribute(ATTR_LLM_TOKENS_CACHED, cached_tokens)
            self._span.set_attribute(ATTR_GEN_AI_USAGE_CACHED_TOKENS, cached_tokens)

            # Calculate cache hit rate
            if input_tokens > 0:
                hit_rate = (cached_tokens / input_tokens) * 100
                self._span.set_attribute(ATTR_LLM_CACHE_HIT_RATE, f"{hit_rate:.1f}%")

        if reasoning_tokens > 0:
            self._span.set_attribute(ATTR_LLM_TOKENS_REASONING, reasoning_tokens)
            self._span.set_attribute(ATTR_GEN_AI_USAGE_REASONING_TOKENS, reasoning_tokens)

        # Calculate cost if model is set
        if self._model and (input_tokens > 0 or output_tokens > 0):
            cost = calculate_cost(
                self._model,
                input_tokens,
                output_tokens,
                cached_tokens,
            )
            if cost is not None:
                self._span.set_attribute(ATTR_LLM_COST_USD, f"{cost:.6f}")
                self._span.set_attribute(ATTR_GEN_AI_USAGE_COST, f"{cost:.6f}")

                # Calculate cache savings
                if cached_tokens > 0:
                    cost_without_cache = calculate_cost(
                        self._model,
                        input_tokens,
                        output_tokens,
                        0,
                    )
                    if cost_without_cache is not None:
                        savings = cost_without_cache - cost
                        self._span.set_attribute(ATTR_LLM_CACHE_SAVINGS_USD, f"{savings:.6f}")

        return self

    def record_usage(
        self,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_tokens: int = 0,
        reasoning_tokens: int = 0,
        provider: Optional[str] = None,
    ) -> "LLMSpan":
        """
        Convenience method to set model and tokens in one call.

        Args:
            model: Model name
            input_tokens: Total input tokens
            output_tokens: Output tokens
            cached_tokens: Cached input tokens
            reasoning_tokens: Reasoning tokens
            provider: Provider name (auto-detected if not specified)
        """
        self.set_model(model, provider)
        self.set_tokens(input_tokens, output_tokens, cached_tokens, reasoning_tokens)
        return self


def add_pricing(model: str, input: float, cached_input: float, output: float) -> None:
    """
    Add custom model pricing to the pricing table.

    Args:
        model: Model name
        input: Price per 1M input tokens (USD)
        cached_input: Price per 1M cached input tokens (USD)
        output: Price per 1M output tokens (USD)
    """
    MODEL_PRICING[model] = {
        "input": input,
        "cached_input": cached_input,
        "output": output,
    }
