"""P11: fetch_models dry run lists everything, makes no network calls, downloads nothing."""

from __future__ import annotations

import socket

from scripts import fetch_models


def _forbid_network(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("network access attempted during dry run")

    monkeypatch.setattr(socket, "socket", boom)
    monkeypatch.setattr(socket, "create_connection", boom)


def test_p11_dry_run_lists_every_model_without_network(monkeypatch, capsys):
    _forbid_network(monkeypatch)

    def forbid_download(*args, **kwargs):
        raise AssertionError("download attempted without --yes")

    monkeypatch.setattr(fetch_models, "_download_hf", forbid_download)
    monkeypatch.setattr(fetch_models, "_download_ultralytics", forbid_download)

    assert fetch_models.main(["--dry-run"]) == 0
    output = capsys.readouterr().out
    for model in fetch_models.MODELS:
        assert model["id"] in output
    assert "DRY RUN" in output

    # Default mode is also a dry run and must not download.
    assert fetch_models.main([]) == 0
    output = capsys.readouterr().out
    for model in fetch_models.MODELS:
        assert model["id"] in output
    assert "DRY RUN" in output
