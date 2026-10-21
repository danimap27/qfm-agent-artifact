"""Smoke tests for the package scaffold."""

from code import __version__


def test_package_imports() -> None:
    """The package imports and exposes a version string."""
    assert isinstance(__version__, str)


def test_version_is_semver() -> None:
    """The version follows a major.minor.patch shape."""
    parts = __version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)
