"""Small dependency helpers shared by the optional OpenAI integrations."""

from importlib.metadata import PackageNotFoundError, version


def openai_sdk_version() -> str | None:
    """Return the installed OpenAI distribution version without importing it."""
    try:
        return version("openai")
    except PackageNotFoundError:
        return None
