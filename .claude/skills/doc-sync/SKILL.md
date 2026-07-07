---
name: doc-sync
description: >
  Map a fireEngine code change to the docs that must be updated in the same branch. Use after any
  code change, before committing, or when the user says "update docs", "doc sync", "which docs",
  "keep docs in sync".
---

# Doc sync

Docs are part of the change, updated in the **same branch**. Maintainer docs live in `docs/`;
`README.md`, `CLAUDE.md` / `AGENTS.md` stay at the repo root. Cross-references are relative links: a
`docs/` file → a sibling is bare (`collision.md`), → a root file is `../README.md`; a root file → a
`docs/` file is `docs/collision.md`.

## By change type

| Change | Update |
|---|---|
| Rendering feature | `README.md` + `docs/onboarding.md` + `docs/review-order.md` + `docs/roadmap.md` |
| Physics feature | the above + `docs/collision.md` |
| LOD / simplifier | `docs/lod.md` (+ README / onboarding if user-visible) |
| New / renamed / split source file | `docs/review-order.md` (tier entry) + `docs/onboarding.md` if it's a subsystem or a common task |
| New cross-file invariant | `docs/onboarding.md` § Cross-File Invariants + `CLAUDE.md` § Architecture |
| User-visible CLI flag / dependency / build step | `README.md` |
| Convention / workflow / invariant rule | `CLAUDE.md` (the `AGENTS.md` symlink mirrors it) |
| Roadmap item landed | `docs/roadmap.md` status |
| Assets / scenes / flags changed | `docs/acceptance-testing.md` |

## Check
Sweep the names of the files you touched for stale references — a constant you moved, a binding you
renamed, a file you split. The authoritative index is CLAUDE.md § Documentation.
