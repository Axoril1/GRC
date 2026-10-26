"""GRC: Turn a public GitHub repository URL into a narrated demo video."""

from __future__ import annotations

import importlib
import sys
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec
from typing import Any

__version__ = "0.1.0"


class _Repo2VideoRedirectFinder(MetaPathFinder):
    """Redirect legacy repo2video imports to GRC transparently."""

    def find_spec(
        self,
        fullname: str,
        path: Any = None,
        target: Any = None,
    ) -> ModuleSpec | None:
        if fullname == "repo2video" or fullname.startswith("repo2video."):
            grc_name = "GRC" + fullname[len("repo2video") :]
            mod = importlib.import_module(grc_name)
            sys.modules[fullname] = mod
            spec = getattr(mod, "__spec__", None)
            return spec
        return None


if not any(isinstance(finder, _Repo2VideoRedirectFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _Repo2VideoRedirectFinder())

sys.modules.setdefault("repo2video", sys.modules[__name__])
