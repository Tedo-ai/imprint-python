"""
Imprint integrations for third-party libraries.
"""

# Lazy imports to avoid requiring all dependencies
def openai_agents():
    """Get the OpenAI Agents SDK integration module."""
    from . import openai_agents as module
    return module
