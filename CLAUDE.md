# Agent rules for this repo

- Before any task, read `CODEBASE_CONTEXT.md` (repo root). It is the source of truth for structure, pipeline, engines, I/O, UI, keys and current state. Then open only the files your task needs (its §17 task index).
- Mandatory: any change to code, data, tests, config or requirements must update the affected sections of `CODEBASE_CONTEXT.md` in the same commit, plus its §0 SYNC_STAMP and §18 SYNC_LOG. A change without this update is incomplete.
- Only grounded facts go in that file: verify against code, mark anything unverified as UNVERIFIED, never guess.
- If `CODEBASE_CONTEXT.md` disagrees with the code, the code wins: fix the file as part of your change.
