from __future__ import annotations

import asyncio
from pathlib import Path

from bdh_graph_harness.api.watcher import VaultWatcher
from bdh_graph_harness.graph.sources import ExternalMarkdownSource, VaultMarkdownSource


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_watcher_tracks_filtered_sources_and_coalesces_changes(tmp_path):
    vault = tmp_path / "vault"
    projects = tmp_path / "projects"
    _write(vault / "note.md", "# Vault")
    _write(projects / "demo/README.md", "# Demo")
    _write(projects / "ignored/README.md", "# Ignored")

    calls = []

    async def update():
        calls.append("update")

    sources = [
        VaultMarkdownSource(str(vault)),
        ExternalMarkdownSource(
            str(projects),
            source_id="projects",
            include=["demo/**/*.md"],
        ),
    ]
    watcher = VaultWatcher(
        str(vault),
        update,
        sources=sources,
        debounce_seconds=0,
    )
    watcher._signatures = watcher._scan_signatures()

    assert set(watcher._signatures) == {
        str(vault / "note.md"),
        str(projects / "demo/README.md"),
    }

    async def exercise():
        # Ignored external files are not part of the snapshot.
        _write(projects / "ignored/README.md", "# Ignored changed")
        await watcher._check_changes()
        await watcher._check_changes()
        assert calls == []

        # A burst of add + modify before the quiet poll is one update.
        _write(projects / "demo/new.md", "# New")
        _write(projects / "demo/README.md", "# Demo changed")
        await watcher._check_changes()
        assert calls == []
        await watcher._check_changes()
        assert calls == ["update"]

        # Delete is observed and dispatched once.
        (projects / "demo/new.md").unlink()
        await watcher._check_changes()
        await watcher._check_changes()
        assert calls == ["update", "update"]

    asyncio.run(exercise())
