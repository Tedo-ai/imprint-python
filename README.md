# Imprint Python SDK

Python SDK for distributed tracing with Imprint.

## Installation

```bash
pip install imprint-python
```

## Usage

```python
import imprint

# Initialize
imprint.init(
    api_key="imp_live_...",
    service_name="my-service",
    ingest_url="http://localhost:17080/v1/spans",
)

# Create spans
with imprint.start_span("operation") as span:
    span.set_attribute("key", "value")
    # ... do work ...

# Manual span management
ctx, span = imprint.get_client().start_span("operation")
try:
    span.set_attribute("key", "value")
    # ... do work ...
finally:
    span.finish()

# Shutdown (flushes remaining spans)
imprint.shutdown()
```

## Configuration

Configuration can be passed to `init()` or set via environment variables:

| Option | Environment Variable | Default |
|--------|---------------------|---------|
| `api_key` | `IMPRINT_API_KEY` | (required) |
| `service_name` | `IMPRINT_SERVICE_NAME` | `python-app` |
| `ingest_url` | `IMPRINT_INGEST_URL` | `http://localhost:17080/v1/spans` |
| `sampling_rate` | - | `1.0` |
| `debug` | `IMPRINT_DEBUG` | `false` |
| `enabled` | `IMPRINT_ENABLED` | `true` |

## Framework Integrations

- **Django**: Use [imprint-django](../imprint-django)
