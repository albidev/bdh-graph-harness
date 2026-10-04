# Alpha adoption and release checklist

The demo is an onboarding boundary, not evidence that the entire system is production-ready. Keep this checklist paired with [compatibility](compatibility.md) and [operations](operations.md).

## Evidence before publishing a tag

- [ ] Fresh dependency install on the declared platform/Python; record exact dependency versions.
- [ ] Run [quickstart](quickstart.md) with a copied fictional vault and no operator credentials/config.
- [ ] Real embedding retrieval and read-only smoke pass; verify learning artifacts, not only HTTP 200.
- [ ] Actual selected local completion model generates a response; a mocked or error response is not success.
- [ ] Explicit Hebbian learning changes state/counter, not demo Markdown; stop/backup/restore recovers the original state.
- [ ] Suite passes, or every failure is reproduced on the unchanged baseline and publicly disclosed. Resolve the query-rewrite golden-set/issue-19 contract discrepancy before declaring the full suite green.
- [ ] Relative doc links, code fences, source scan links and effective config values checked.
- [ ] Optional bridge is tested separately through a real isolated Hermes registry dispatch; no production chat needed.
- [ ] No private vaults, config backups, policy maps, transcripts, credentials, benchmark inputs or caches included in the release payload.
- [ ] Independent review covers changed artifacts and actual runtime contracts; reject unsupported reviewer assertions.
- [ ] Define release version/tag, upgrade notes and rollback instructions; get explicit approval before commits/push/tag publication.

## What alpha does not promise

No stable API/support SLA, all-provider compatibility, full network hardening, exactly-once writes, automatic retention purge or independent scientific validation of the architecture is implied. Learning and synthesis remain operator-governed experiments. A read-only onboarding check cannot certify watcher, curation or production traffic behavior.

For the optional bridge, the direct-timeout retry limitation is a **write-path release blocker** until fixed and verified against all timeout shapes. Disabling rewrite/synthesis flags is not a universal automatic-write disable. See the bridge [operations](https://github.com/albidev/bdh-hermes-bridge/blob/main/docs/operations.md) and [alpha checklist](https://github.com/albidev/bdh-hermes-bridge/blob/main/docs/alpha-checklist.md).
