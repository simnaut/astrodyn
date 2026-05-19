# wiki-stubs/

Staging area for content moved out of [`CLAUDE.md`](../CLAUDE.md) that should
live on the [GitHub wiki](https://github.com/simnaut/astrodyn/wiki). Each file
here is named for the wiki page it should populate.

## Workflow

1. **Promote**: for each file in this directory, create the matching page on
   the wiki (e.g., `Architecture.md` → `https://github.com/simnaut/astrodyn/wiki/Architecture`)
   and paste the contents.
2. **Verify**: open `CLAUDE.md` and click each `wiki/<Page>` link — they should
   all resolve to live pages.
3. **Delete**: once every stub has been promoted, delete this directory in a
   single follow-up commit. There is no reason to keep the staged content in
   the repo after the wiki is populated; the wiki is the source of truth for
   non-Rust-crate docs (per `CLAUDE.md`'s documentation convention).

## Inventory

| Stub | Wiki page (after promotion) | Source of original content |
|---|---|---|
| [Architecture.md](Architecture.md) | `wiki/Architecture` | Old `CLAUDE.md` §"Three-Layer Architecture" |
| [CI.md](CI.md) | `wiki/CI` | Old `CLAUDE.md` §"Build and Test" + §"Test tiers and CI" |
| [Common-Pitfalls.md](Common-Pitfalls.md) | `wiki/Common-Pitfalls` | Old `CLAUDE.md` §"Common Pitfalls" |
| [Environment.md](Environment.md) | `wiki/Environment` | Old `CLAUDE.md` §"Environment Setup" |
| [JEOD-ECS-Mapping.md](JEOD-ECS-Mapping.md) | `wiki/JEOD-ECS-Mapping` | Old `CLAUDE.md` §"JEOD Integration Loop" + §"JEOD Key Classes -> ECS Mapping" |
| [JEOD-Invariant-Workflow.md](JEOD-Invariant-Workflow.md) | `wiki/JEOD-Invariant-Workflow` | Old `CLAUDE.md` §"JEOD Invariant Tracking" |

The git diff that introduced these stubs (and the corresponding lean rewrite of
`CLAUDE.md`) is the canonical authority on which content went where — `git log
-p CLAUDE.md` from this commit's parent shows the full migration.
