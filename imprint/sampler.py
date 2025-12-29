"""Sampling logic for distributed tracing."""

from abc import ABC, abstractmethod


class Sampler(ABC):
    """Base class for samplers."""

    @abstractmethod
    def should_sample(self, trace_id: str) -> bool:
        """Determine if a trace should be sampled."""
        pass


class AlwaysSampler(Sampler):
    """Always sample all traces."""

    def should_sample(self, trace_id: str) -> bool:
        return True


class NeverSampler(Sampler):
    """Never sample any traces."""

    def should_sample(self, trace_id: str) -> bool:
        return False


class RateSampler(Sampler):
    """
    Sample traces at a given rate using consistent hashing.

    The same trace_id will always produce the same sampling decision,
    ensuring all spans in a trace are either all sampled or all dropped.
    """

    def __init__(self, rate: float):
        """
        Initialize rate sampler.

        Args:
            rate: Sampling rate between 0.0 and 1.0
        """
        self.rate = max(0.0, min(1.0, rate))
        # Calculate threshold for FNV-1a hash comparison
        self._threshold = int(self.rate * (2**64 - 1))

    def should_sample(self, trace_id: str) -> bool:
        if self.rate >= 1.0:
            return True
        if self.rate <= 0.0:
            return False

        # FNV-1a hash for consistent sampling
        h = self._fnv1a_hash(trace_id)
        return h < self._threshold

    @staticmethod
    def _fnv1a_hash(s: str) -> int:
        """FNV-1a 64-bit hash."""
        FNV_OFFSET = 14695981039346656037
        FNV_PRIME = 1099511628211
        MASK = 2**64 - 1

        h = FNV_OFFSET
        for byte in s.encode("utf-8"):
            h ^= byte
            h = (h * FNV_PRIME) & MASK
        return h


def create_sampler(rate: float) -> Sampler:
    """Create a sampler for the given rate."""
    if rate >= 1.0:
        return AlwaysSampler()
    elif rate <= 0.0:
        return NeverSampler()
    else:
        return RateSampler(rate)
