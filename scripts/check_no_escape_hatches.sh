#!/usr/bin/env bash
# CI guard: no escape-hatch APIs may leak into the Bevy adapter.
#
# Two categories of escape hatch are policed:
#
# 1. **Marker-based** (`#[doc(hidden)]`, `tag_as_inertial!`):
#    Items deliberately removed from the public-API surface but still
#    callable. The `tag_as_inertial!` macro doesn't currently exist —
#    grepped defensively to keep the door closed against re-introduction.
#    Banned across `crates/` and `src/` except where annotated with
#    `// allowed: <reason>`.
#
# 2. **Typed-quantity bypass constructors** (`from_untyped_unchecked`,
#    `from_dmat3_unchecked`, `from_raw_si`):
#    The Phase-8 typed-quantity facade promises that frame mismatches
#    are compile errors. These constructors mint a typed value from raw
#    storage without any check that the caller's frame phantom matches
#    reality.
#
#    They are legitimately part of the typed-sibling boundary inside
#    `crates/jeod_*/src/` — every typed sibling (`TranslationalStateTyped`,
#    `MassPropertiesTyped`, `GravityAccelerationTyped`, …) implements a
#    `from_untyped_unchecked` / `from_raw_si` bridge by definition. So
#    `crates/**` is allowed for category (2).
#
#    What's banned for category (2): the **Bevy adapter** (`src/**`).
#    The audit (issue #172, finding H1) identified per-step `from_raw_si`
#    lifts in `src/systems.rs` as the load-bearing failure mode of the
#    typed-quantity facade — every system was extracting raw `DVec3`
#    from a component, hand-tagging it `Inertial`, and dropping back to
#    raw on exit, so the phantom never crossed the ECS boundary. The
#    fix is to make Bevy components wrap the typed siblings directly;
#    this guard prevents regression by refusing the bypass APIs in
#    `src/`.
#
#    `// allowed: <reason>` annotations exempt individual `src/` lines
#    when there is no choice (typically a JEOD-CSV-style boundary).
#    Use sparingly and document each exemption in the PR.
set -euo pipefail

# ── Category 1: marker-based (banned across crates/ + src/) ──
marker_matches=$(grep -rEn '#\[doc\(hidden\)\]|tag_as_inertial!' crates/ src/ \
  | grep -v '// allowed:' || true)

# ── Category 2: typed-quantity bypass constructors (banned in src/systems.rs only) ──
# `crates/**` is fully exempt — the typed siblings and their internal
# `_unchecked` bridges all live there by construction.
# `src/components.rs` is exempt — Bevy components' `From<Untyped>` impls
# are the canonical insertion-time boundary; each one is the analogue
# of jeod_dynamics's `from_untyped_unchecked`.
# `src/lib.rs` is exempt — `spawn_bevy` performs insertion-time lifts
# from `VehicleConfig` (still untyped in jeod_sim) to typed components.
# Annotations document each per-step bypass in `src/systems.rs`. The
# annotation may be on the same line as the bypass *or* on the
# immediately-preceding line (multi-line generic patterns like
# `TypedX::<...>::from_untyped_unchecked(\n  arg\n)` can't fit a
# trailing comment on the matched line).
#
# Implementation: awk reads systems.rs, tracks whether a preceding
# comment block (contiguous `//`-prefixed lines, allowing blank lines
# between them) contained `// allowed:`. Reports any matching line
# that has neither inline nor preceding `// allowed:` annotation.
bypass_matches=$(awk '
    # Track contiguous comment / blank context. Any `// allowed:` in
    # the trailing comment block applies to the next non-comment line.
    /\/\/ allowed:/ { prev_allowed = 1; next }
    /^[[:space:]]*\/\// { next }   # comment line, dont reset
    /^[[:space:]]*$/ { next }      # blank line, dont reset
    /from_untyped_unchecked|from_dmat3_unchecked|from_raw_si/ {
        if (prev_allowed) { prev_allowed = 0; next }
        if ($0 ~ /\/\/ allowed:/) { prev_allowed = 0; next }
        printf "src/systems.rs:%d: %s\n", NR, $0
        prev_allowed = 0
        next
    }
    { prev_allowed = 0 }
' src/systems.rs)

failed=0

if [ -n "$marker_matches" ]; then
    echo "FAIL: escape-hatch markers detected" >&2
    echo "$marker_matches" >&2
    failed=1
fi

if [ -n "$bypass_matches" ]; then
    echo "FAIL: typed-quantity bypass constructors in the Bevy adapter (src/)" >&2
    echo "  (from_untyped_unchecked / from_dmat3_unchecked / from_raw_si)" >&2
    echo "  These constructors mint a typed value from raw storage without" >&2
    echo "  checking the caller's frame phantom. The Bevy adapter must use" >&2
    echo "  the typed APIs (Position<Inertial>, etc.) directly via the" >&2
    echo "  components, not lift raw storage per step." >&2
    echo "  See issue #172 (audit finding H1) for context." >&2
    echo "  If you have a documented boundary case, annotate the line with" >&2
    echo "  '// allowed: <reason>'." >&2
    echo "$bypass_matches" >&2
    failed=1
fi

if [ "$failed" -ne 0 ]; then
    exit 1
fi

echo "OK: no escape-hatch markers or unsanctioned typed-quantity bypasses in src/"
