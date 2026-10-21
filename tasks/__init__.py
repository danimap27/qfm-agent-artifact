"""Task registry and loaders for the QFM-lite substrate."""

from tasks.loaders import TaskData
from tasks.registry import SPECS, get_tasks

__all__ = ["SPECS", "TaskData", "get_tasks"]
