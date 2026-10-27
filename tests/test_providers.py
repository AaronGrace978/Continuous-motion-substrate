from __future__ import annotations

import json
import os
import stat

import pytest

from chronosoul_bridge import ChronoSoulHTTPSource, SimulatedAffectSource, StaticAffectSource
from providers import ProviderError, ProviderSettings


def test_defaults_are_offline(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("MOTION_AFFECT_SOURCE", raising=False)
    settings = ProviderSettings(tmp_path / "motion_providers.json")
    assert settings.data["affect_source"] == "simulated"
    assert isinstance(settings.build_source(), SimulatedAffectSource)


def test_saved_with_0600_and_token_never_public(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("CHRONOSOUL_TOKEN", raising=False)
    path = tmp_path / "motion_providers.json"
    settings = ProviderSettings(path)
    settings.update(affect_source="chronosoul_http", chronosoul_token="s3cret")
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    public = settings.public()
    assert "s3cret" not in json.dumps(public)
    assert public["has_token"] is True
    assert json.loads(path.read_text())["chronosoul_token"] == "s3cret"


def test_env_token_wins(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CHRONOSOUL_TOKEN", "from-env")
    assert ProviderSettings(tmp_path / "p.json").token() == "from-env"


def test_rejects_bad_values(tmp_path) -> None:
    settings = ProviderSettings(tmp_path / "p.json")
    with pytest.raises(ProviderError):
        settings.update(affect_source="telepathy")
    with pytest.raises(ProviderError):
        settings.update(api_key="x")
    with pytest.raises(ProviderError):
        settings.update(chronosoul_token="a\nb")


def test_corrupt_or_unknown_falls_back(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("MOTION_AFFECT_SOURCE", raising=False)
    path = tmp_path / "p.json"
    path.write_text("{not json")
    assert ProviderSettings(path).data["affect_source"] == "simulated"
    path.write_text(json.dumps({"affect_source": "nope"}))
    assert ProviderSettings(path).data["affect_source"] == "simulated"


def test_builds_each_source(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("MOTION_AFFECT_SOURCE", raising=False)
    settings = ProviderSettings(tmp_path / "p.json")
    settings.update(affect_source="manual")
    assert isinstance(settings.build_source(), StaticAffectSource)
    settings.update(affect_source="chronosoul_http", chronosoul_url="http://127.0.0.1:9/state")
    source = settings.build_source()
    assert isinstance(source, ChronoSoulHTTPSource)
    source.stop()
    settings.update(affect_source="chronosoul", chronosoul_repo=str(tmp_path / "missing"))
    assert isinstance(settings.build_source(), SimulatedAffectSource)
