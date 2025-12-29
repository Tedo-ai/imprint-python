"""Configuration for Imprint."""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    """Imprint client configuration."""

    api_key: str = field(default_factory=lambda: os.getenv("IMPRINT_API_KEY", ""))
    service_name: str = field(default_factory=lambda: os.getenv("IMPRINT_SERVICE_NAME", "python-app"))
    ingest_url: str = field(default_factory=lambda: os.getenv("IMPRINT_INGEST_URL", "http://localhost:17080/v1/spans"))

    # Request filtering
    ignore_paths: List[str] = field(default_factory=list)
    ignore_prefixes: List[str] = field(default_factory=list)
    ignore_extensions: List[str] = field(default_factory=lambda: [
        ".css", ".js", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".woff", ".woff2", ".ttf", ".map"
    ])

    # Batching
    batch_size: int = 100
    flush_interval: float = 5.0  # seconds
    buffer_size: int = 1000

    # Sampling
    sampling_rate: float = 1.0  # 0.0 to 1.0

    # Debug
    debug: bool = field(default_factory=lambda: os.getenv("IMPRINT_DEBUG", "").lower() == "true")
    enabled: bool = field(default_factory=lambda: os.getenv("IMPRINT_ENABLED", "true").lower() == "true")

    def should_ignore(self, path: str) -> bool:
        """Check if a path should be ignored."""
        if path in self.ignore_paths:
            return True
        for prefix in self.ignore_prefixes:
            if path.startswith(prefix):
                return True
        for ext in self.ignore_extensions:
            if path.endswith(ext):
                return True
        return False
