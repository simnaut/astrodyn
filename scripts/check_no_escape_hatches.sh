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
# 2. **Typed-quantity bypass constructors** — raw constructors that mint
#    a typed value from primitive storage without checking the caller's
#    phantom tags / conventions / normalization invariants:
#
#      - `from_untyped_unchecked` — typed sibling boundary
#      - `from_dmat3_unchecked`   — `InertiaTensor` (skips symmetry check)
#      - `from_raw_si`            — `Qty3` (raw `DVec3` → typed)
#      - `from_seconds`           — `SecondsSince` (raw `f64` → tagged time)
#      - `from_array`             — `Quat`/`JeodQuat` (raw `[f64;4]` →
#                                   tagged quaternion, skips normalization
#                                   and convention checks)
#      - `from_matrix(`           — `FrameTransform` (raw `DMat3` → typed
#                                   transform, skips orthonormality check
#                                   in release builds; the validating
#                                   sibling is `from_matrix_validated`)
#
#    The Phase-8 typed-quantity facade promises that frame, time-scale,
#    quaternion-convention, and dimensional mismatches are compile
#    errors. These constructors mint a typed value from raw storage
#    without any check that the caller's phantom matches reality, so
#    per-step uses in the Bevy adapter are exactly the regression class
#    audit finding H1 documented.
#
#    They are legitimately part of the typed-sibling boundary inside
#    `crates/astrodyn_*/src/` — every typed sibling (`TranslationalStateTyped`,
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

# ── Category 2: typed-quantity bypass constructors (banned across src/**) ──
# `crates/**` is fully exempt — the typed siblings and their internal
# `_unchecked` bridges all live there by construction.
# `src/components/**` is exempt — Bevy components' `From<Untyped>` impls
# are the canonical insertion-time boundary; each one is the analogue
# of astrodyn_dynamics's `from_untyped_unchecked`. (Pre-split history: this
# was a single `src/components.rs` file; the same boundary semantics
# apply to every per-stage submodule under the new `src/components/`
# directory.)
# `src/lib.rs` is exempt — `spawn_bevy` performs insertion-time lifts
# from `VehicleConfig` (still untyped in astrodyn) to typed components.
# All other files under `src/**` are policed.
#
# Annotations document each per-step bypass. The annotation may be:
# - inline on the same line as the bypass (`from_raw_si(...) // allowed: …`); or
# - on a pure-comment line *immediately preceding* the bypass (with blank
#   or other comment lines allowed between them) — needed for multi-line
#   generic patterns like `TypedX::<...>::from_untyped_unchecked(\n …\n)`
#   that can't fit a trailing comment on the matched line.
#
# The awk script distinguishes these: only `// allowed:` on a
# pure-comment line propagates to the next bypass match. An inline
# `// allowed:` on a code line annotates **only that line** — it does
# not exempt subsequent unrelated lines.
src_files_to_scan=$(find src/ -name "*.rs" -type f \
    -not -path 'src/components.rs' \
    -not -path 'src/components/*' \
    -not -path 'src/lib.rs' \
    | sort)

bypass_matches=$(echo "$src_files_to_scan" | xargs awk '
    FNR == 1 { prev_allowed = 0 }
    # Pure-comment line with `// allowed:` propagates to the next
    # non-comment, non-blank line. (Code lines are handled by the
    # bypass-rule below; an inline `// allowed:` self-annotates and
    # never propagates.)
    /^[[:space:]]*\/\/.*allowed:/ { prev_allowed = 1; next }
    # Pure-comment line without `// allowed:`: keep prev_allowed as-is
    # so a `// allowed:` further up still applies through the comment
    # block.
    /^[[:space:]]*\/\// { next }
    # Blank: keep prev_allowed as-is.
    /^[[:space:]]*$/ { next }
    /from_untyped_unchecked|from_dmat3_unchecked|from_raw_si|from_seconds|(JeodQuat|Quat)::from_array|FrameTransform::from_matrix\(/ {
        if (prev_allowed) { prev_allowed = 0; next }
        if ($0 ~ /\/\/ allowed:/) { prev_allowed = 0; next }
        printf "%s:%d: %s\n", FILENAME, FNR, $0
        prev_allowed = 0
        next
    }
    # Any other code line resets the propagating-allowed state.
    { prev_allowed = 0 }
')

failed=0

if [ -n "$marker_matches" ]; then
    echo "FAIL: escape-hatch markers detected" >&2
    echo "$marker_matches" >&2
    failed=1
fi

if [ -n "$bypass_matches" ]; then
    echo "FAIL: typed-quantity bypass constructors in the Bevy adapter" >&2
    echo "  (scanned: src/**, except the canonical boundary modules" >&2
    echo "   src/components.rs and src/lib.rs)" >&2
    echo "  Banned: from_untyped_unchecked / from_dmat3_unchecked / from_raw_si /" >&2
    echo "          from_seconds / (JeodQuat|Quat)::from_array / FrameTransform::from_matrix(" >&2
    echo "  These constructors mint a typed value from raw storage without" >&2
    echo "  checking the caller's frame / time-scale / quaternion-convention /" >&2
    echo "  normalization phantoms. The Bevy adapter must use the typed APIs" >&2
    echo "  (Position<RootInertial>, SecondsSince<TAI>, NormalizedQuat<...>," >&2
    echo "  FrameTransform::from_matrix_validated, etc.) directly via the" >&2
    echo "  components, not lift raw storage per step." >&2
    echo "  See issue #172 (audit finding H1) for context." >&2
    echo "  If you have a documented boundary case, annotate the line with" >&2
    echo "  '// allowed: <reason>' (inline) or place '// allowed: <reason>'" >&2
    echo "  on a pure-comment line immediately preceding the bypass." >&2
    echo "$bypass_matches" >&2
    failed=1
fi

if [ "$failed" -ne 0 ]; then
    exit 1
fi

echo "OK: no escape-hatch markers or unsanctioned typed-quantity bypasses in src/"
