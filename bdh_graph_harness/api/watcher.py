"""Filesystem watcher for vault changes.

Monitors the vault directory for .md file changes by comparing mtimes.
Polls every N seconds, triggers graph rebuild + WS broadcast when changes detected.
"""

import asyncio
import logging
import os
import time
from typing import Callable, Coroutine, Optional

logger = logging.getLogger("bdh-watcher")


class VaultWatcher:
    """Watches vault directory for .md changes and triggers graph updates."""

    def __init__(
        self,
        vault_path: str,
        update_fn: Callable[[], Coroutine],
        poll_interval: float = 2.0,
    ):
        self.vault_path = vault_path
        self.update_fn = update_fn
        self.poll_interval = poll_interval
        self._mtimes: dict[str, float] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._task: Optional[asyncio.Task] = None

    def start(self, loop: asyncio.AbstractEventLoop):
        """Start watching the vault directory."""
        self._loop = loop
        self._mtimes = self._scan_mtimes()
        print(f"👁️  Vault watcher: scanned {len(self._mtimes)} files", flush=True)
        self._task = loop.create_task(self._poll_loop())
        print(f"👁️  Vault watcher: poll task created: {self._task}", flush=True)

    def stop(self):
        if self._task:
            self._task.cancel()

    def _scan_mtimes(self) -> dict[str, float]:
        mtimes = {}
        for root, dirs, files in os.walk(self.vault_path):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for f in files:
                if not f.endswith('.md') or f.startswith('.'):
                    continue
                path = os.path.join(root, f)
                try:
                    mtimes[path] = os.path.getmtime(path)
                except OSError:
                    continue
        return mtimes

    async def _poll_loop(self):
        print(f"👁️  Vault watcher: poll loop started", flush=True)
        while True:
            await asyncio.sleep(self.poll_interval)
            try:
                await self._check_changes()
            except asyncio.CancelledError:
                print(f"👁️  Vault watcher: poll loop cancelled", flush=True)
                break
            except Exception as e:
                print(f"👁️  Vault watcher error: {e}", flush=True)

    async def _check_changes(self):
        new_mtimes = self._scan_mtimes()

        changed = []
        for path, mtime in new_mtimes.items():
            old_mtime = self._mtimes.get(path, 0)
            if mtime > old_mtime:
                changed.append(path)

        added = [p for p in new_mtimes if p not in self._mtimes]
        deleted = [p for p in self._mtimes if p not in new_mtimes]

        if not added and not changed and not deleted:
            return

        print(f"👁️  Vault watcher: {len(added)} new, {len(changed)} changed, {len(deleted)} deleted", flush=True)
        self._mtimes = new_mtimes
        logger.info(f"Vault watcher: {len(added)} new, {len(changed)} changed, {len(deleted)} deleted — triggering update")

        try:
            print(f"👁️  Vault watcher: calling update_fn...", flush=True)
            await self.update_fn()
            print(f"👁️  Vault watcher: update_fn completed", flush=True)
        except Exception as e:
            import traceback
            print(f"👁️  Vault watcher update FAILED: {e}", flush=True)
            traceback.print_exc()
