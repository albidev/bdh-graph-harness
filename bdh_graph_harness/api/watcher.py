"""Filesystem watcher for configured Markdown sources.

Monitors the primary vault and any configured external Markdown sources using
source include/exclude rules. Polling is intentionally boring and portable;
short bursts of editor writes are coalesced before the graph update callback
runs. The watcher never writes to a source directory.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Callable, Coroutine, Iterable, Optional

logger = logging.getLogger("bdh-watcher")


class VaultWatcher:
    """Watch one or more configured Markdown sources for filesystem changes."""

    def __init__(
        self,
        vault_path: str,
        update_fn: Callable[[], Coroutine],
        poll_interval: float = 2.0,
        *,
        sources: Iterable[Any] | None = None,
        debounce_seconds: float = 1.0,
    ):
        self.vault_path = vault_path
        self.update_fn = update_fn
        self.poll_interval = poll_interval
        self.sources = tuple(sources or ())
        self.debounce_seconds = max(0.0, debounce_seconds)
        self._signatures: dict[str, tuple[int, int]] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._task: Optional[asyncio.Task] = None
        self._pending_since: float | None = None
        self._pending_counts = (0, 0, 0)

    def start(self, loop: asyncio.AbstractEventLoop):
        """Start watching the configured source directories."""
        self._loop = loop
        self._signatures = self._scan_signatures()
        print(f"👁️  Vault watcher: scanned {len(self._signatures)} files", flush=True)
        self._task = loop.create_task(self._poll_loop())
        print(f"👁️  Vault watcher: poll task created: {self._task}", flush=True)

    def stop(self):
        if self._task:
            self._task.cancel()
            self._task = None

    def _scan_signatures(self) -> dict[str, tuple[int, int]]:
        """Return stable file signatures for all matching Markdown paths."""
        if self.sources:
            paths = (
                absolute
                for source in self.sources
                for _relative, absolute in source.iter_paths()
            )
        else:
            paths = self._legacy_paths()

        signatures: dict[str, tuple[int, int]] = {}
        for path in paths:
            try:
                stat = os.stat(path)
            except OSError:
                continue
            signatures[path] = (stat.st_mtime_ns, stat.st_size)
        return signatures

    def _legacy_paths(self):
        """Keep the old constructor behavior for non-federated callers."""
        for root, dirs, files in os.walk(self.vault_path):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for filename in files:
                if filename.startswith('.') or not filename.lower().endswith('.md'):
                    continue
                yield os.path.join(root, filename)

    async def _poll_loop(self):
        print("👁️  Vault watcher: poll loop started", flush=True)
        while True:
            await asyncio.sleep(self.poll_interval)
            try:
                await self._check_changes()
            except asyncio.CancelledError:
                print("👁️  Vault watcher: poll loop cancelled", flush=True)
                break
            except Exception as exc:
                print(f"👁️  Vault watcher error: {exc}", flush=True)

    async def _check_changes(self):
        new_signatures = self._scan_signatures()
        old_signatures = self._signatures
        changed = [
            path
            for path, signature in new_signatures.items()
            if path in old_signatures and signature != old_signatures[path]
        ]
        added = [path for path in new_signatures if path not in old_signatures]
        deleted = [path for path in old_signatures if path not in new_signatures]
        self._signatures = new_signatures

        if added or changed or deleted:
            self._pending_since = time.monotonic()
            self._pending_counts = (len(added), len(changed), len(deleted))
            logger.info(
                "Watcher change burst: %s new, %s changed, %s deleted",
                len(added), len(changed), len(deleted),
            )
            return

        if self._pending_since is None:
            return
        if time.monotonic() - self._pending_since < self.debounce_seconds:
            return

        added_count, changed_count, deleted_count = self._pending_counts
        self._pending_since = None
        self._pending_counts = (0, 0, 0)
        print(
            f"👁️  Vault watcher: {added_count} new, "
            f"{changed_count} changed, {deleted_count} deleted — triggering update",
            flush=True,
        )
        try:
            print("👁️  Vault watcher: calling update_fn...", flush=True)
            await self.update_fn()
            print("👁️  Vault watcher: update_fn completed", flush=True)
        except Exception as exc:
            import traceback
            print(f"👁️  Vault watcher update FAILED: {exc}", flush=True)
            traceback.print_exc()
