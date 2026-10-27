"""Affect-source settings, persisted like ChronoSoul's `chronosoul_providers.json`.

Settings live in `motion_providers.json` (gitignored, written with 0600
permissions). No secret is hard-coded. A token for a ChronoSoul chamber behind an
authenticating proxy can come from the file or from `CHRONOSOUL_TOKEN`; it is
never returned by `public()`.

    {
      "affect_source": "simulated" | "manual" | "chronosoul" | "chronosoul_http",
      "chronosoul_url": "http://127.0.0.1:47321/state",
      "chronosoul_repo": "/path/to/continuous-emotional-substrate",
      "chronosoul_token": ""
    }
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from chronosoul_bridge import (
    AffectSource,
    ChronoSoulAffectSource,
    ChronoSoulHTTPSource,
    SimulatedAffectSource,
    StaticAffectSource,
    load_chronosoul_substrate,
)

SOURCES = ("simulated", "manual", "chronosoul", "chronosoul_http")
DEFAULTS: dict[str, Any] = {
    "affect_source": "simulated",
    "chronosoul_url": "http://127.0.0.1:47321/state",
    "chronosoul_repo": "",
    "chronosoul_token": "",
}
TOKEN_ENV = "CHRONOSOUL_TOKEN"


class ProviderError(ValueError):
    pass


class ProviderSettings:
    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else None
        self.data: dict[str, Any] = dict(DEFAULTS)
        if self.path and self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self.data.update({k: v for k, v in loaded.items() if k in DEFAULTS})
            except (OSError, ValueError):
                pass
        env_source = os.environ.get("MOTION_AFFECT_SOURCE")
        if env_source:
            self.data["affect_source"] = env_source
        if self.data["affect_source"] not in SOURCES:
            self.data["affect_source"] = "simulated"

    def token(self) -> str:
        return os.environ.get(TOKEN_ENV) or str(self.data.get("chronosoul_token") or "")

    def public(self) -> dict[str, Any]:
        """Settings safe to return over HTTP: the token is reported only as present or absent."""
        out = {k: v for k, v in self.data.items() if k != "chronosoul_token"}
        out["has_token"] = bool(self.token())
        out["sources"] = list(SOURCES)
        return out

    def update(self, **changes: Any) -> None:
        for key, value in changes.items():
            if key not in DEFAULTS:
                raise ProviderError(f"unknown setting: {key}")
            if key == "affect_source" and value not in SOURCES:
                raise ProviderError(f"affect_source must be one of {', '.join(SOURCES)}")
            if key == "chronosoul_token" and value and (len(value) > 500 or any(ch in value for ch in "\r\n")):
                raise ProviderError("token looks invalid")
            self.data[key] = value
        self.save()

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(self.data, ensure_ascii=False, indent=2).encode("utf-8")
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_bytes(raw)
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        tmp.replace(self.path)

    def build_source(self) -> AffectSource:
        """Construct the configured source. Anything unavailable falls back to simulated."""
        kind = self.data["affect_source"]
        if kind == "manual":
            return StaticAffectSource()
        if kind == "chronosoul_http":
            source = ChronoSoulHTTPSource(self.data["chronosoul_url"], token=self.token() or None)
            source.start()
            return source
        if kind == "chronosoul" and self.data.get("chronosoul_repo"):
            try:
                return ChronoSoulAffectSource(load_chronosoul_substrate(self.data["chronosoul_repo"]))
            except Exception as exc:  # missing checkout or its heavy deps
                print(f"[MotionSubstrate] ChronoSoul unavailable ({exc}); using simulated affect")
        return SimulatedAffectSource()
