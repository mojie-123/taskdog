"""Task implementations for custom environments.

Auto-discovers and imports all task configurations so that
gym.register() calls in each task's __init__.py are executed.
"""

from isaaclab_tasks.utils import import_packages

_BLACKLIST_PKGS = ["utils"]

# Import all configs in sub-packages (this triggers gym.register() calls)
import_packages(__name__, _BLACKLIST_PKGS)
