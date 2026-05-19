# JEOD invariant workflow

`CLAUDE.md` carries the one-paragraph rule: every JEOD invariant we encounter
goes in `docs/JEOD_invariants.md`, every enforcement site carries a
`// JEOD_INV: XX.YY` source tag, and `tests/invariant_coverage.rs` enforces
the bidirectional consistency. This page documents the full process,
including the negative-test convention and the section conventions.

## How it works

1. **Catalog** (`docs/JEOD_invariants.md`): one row per invariant with a
   `Section.Tag` ID (e.g., `GV.04`), description, JEOD enforcement mechanism,
   and our status (`enforced`, `partial`, `deferred`, `n/a`, `structural`).

2. **Source tags**: every enforcement site in our Rust code has a comment
   like `// JEOD_INV: GV.04 — degree <= source degree`. The tag text should
   accurately describe what the code does and note any divergence from JEOD.

3. **CI coverage** (`tests/invariant_coverage.rs`, at the workspace root):
   bidirectional test — every `enforced` / `partial` / `structural` invariant
   in the catalog must have at least one source tag, and every source tag
   must reference a catalog entry.

4. **Negative-test coverage** (same file,
   `enforced_rows_have_negative_tests_or_warn`): every `enforced` catalog row
   *should* have a `#[should_panic(expected = "…")]` test that drives the
   misconfiguration and pins the panic message. Recognised by a
   `// JEOD_INV: XX.YY` tag inside the test function or in the few lines
   immediately above the `#[should_panic]` attribute. **The bidirectional
   tag-↔-catalog check only proves the panic site *exists* — the negative
   test proves it actually fires**; without it a future refactor could
   neuter an `assert!` to a silent `if … return` and CI would stay green.
   The gate is informational (warning report, exit 0) in this round and is
   promoted to a hard assertion in a follow-up PR once the catalog body of
   negative tests lands per section (DB / RF / IG / GV / TM / …). To add
   one: write `#[should_panic(expected = "<substring>")]` near the
   enforcement site, drive the failure with a minimal misconfiguration, and
   tag the test with `// JEOD_INV: XX.YY`.

## When you encounter an unrecorded invariant

When reading JEOD source and you find a `MessageHandler::fail()`,
`MessageHandler::error()`, assert, or structural guarantee that is not
already in the catalog:

1. **Add a row** to `docs/JEOD_invariants.md` with the next available tag in
   the appropriate section (e.g., `DB.28`, `GV.19`).
2. **Add `// JEOD_INV: XX.YY`** at our enforcement site (or note `deferred` /
   `n/a` in the catalog if we don't enforce it yet).
3. **Run** `cargo test --test invariant_coverage` to verify consistency.

## When you encounter an untagged enforcement site

If our code enforces a JEOD invariant but the enforcement site lacks a
`// JEOD_INV` tag, add the tag. If the invariant isn't in the catalog yet,
add it there too.

## Tag accuracy

Tags must describe what the code **actually does**, not what JEOD does. When
our implementation diverges from JEOD (e.g., we divide by mass at runtime
instead of precomputing `inverse_mass`), the tag should note the divergence.
Never copy JEOD's description verbatim if our code works differently.
