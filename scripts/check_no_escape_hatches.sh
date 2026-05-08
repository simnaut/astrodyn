#!/usr/bin/env bash
# CI guard: no escape-hatch APIs may leak into the gateway or the Bevy adapter.
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
#    per-step uses in the Bevy adapter or the gateway are exactly the
#    regression class audit finding H1 documented.
#
#    They are legitimately part of the typed-sibling boundary inside
#    `crates/astrodyn_*/src/` — every typed sibling (`TranslationalStateTyped`,
#    `MassPropertiesTyped`, `GravityAccelerationTyped`, …) implements a
#    `from_untyped_unchecked` / `from_raw_si` bridge by definition. So
#    `crates/**` (excluding `crates/astrodyn_bevy/`) is allowed for
#    category (2).
#
#    What's banned for category (2):
#
#    - The **Bevy adapter** (`crates/astrodyn_bevy/src/**`). The audit
#      (issue #172, finding H1) identified per-step `from_raw_si` lifts
#      in the adapter's systems as the load-bearing failure mode of the
#      typed-quantity facade — every system was extracting raw `DVec3`
#      from a component, hand-tagging it `Inertial`, and dropping back
#      to raw on exit, so the phantom never crossed the ECS boundary.
#      The fix is to make Bevy components wrap the typed siblings
#      directly; this guard prevents regression by refusing the bypass
#      APIs in the adapter.
#    - The **Bevy adapter integration tests**
#      (`crates/astrodyn_bevy/tests/**`). These are the closest thing
#      to user-facing worked examples for mission code — a mission
#      author looking at how to wire up a `VehicleBuilder` lands here
#      and copies the pattern. The bypass must not leak into that
#      surface (issue #388 review).
#    - The **gateway** (`src/**`, the workspace-root `astrodyn` crate).
#      Per CLAUDE.md the gateway is the single API surface for the
#      production path; bypass constructors there violate the spirit of
#      the typed-quantity facade introduced in #101 / #172 (issue #388).
#      Code under `#[cfg(test)] mod tests { ... }` blocks is skipped —
#      unit-test fixtures are a sanctioned boundary. Integration tests
#      under `crates/astrodyn_bevy/tests/**` are NOT skipped because
#      they are the worked-example surface described above.
#
#    `// allowed: <reason>` annotations exempt individual lines when
#    there is no choice (typically a documented orchestration boundary,
#    e.g. `integrate_body_typed`'s lift over the gateway-owned
#    `IntegratorType` wrapper). Use sparingly and document each
#    exemption in the PR.
set -euo pipefail

# ── Category 1: marker-based (banned across crates/ + src/) ──
marker_matches=$(grep -rEn '#\[doc\(hidden\)\]|tag_as_inertial!' crates/ src/ \
  | grep -v '// allowed:' || true)

# ── Category 2: typed-quantity bypass constructors ──
# Scope: gateway (`src/**`) and Bevy adapter (`crates/astrodyn_bevy/src/**`).
# Exempt under `crates/astrodyn_bevy/src/`:
#   - `components.rs` and `components/**`: `From<Untyped>` impls are the
#     canonical insertion-time boundary.
#   - `lib.rs`: `spawn_bevy` performs insertion-time lifts from
#     `VehicleConfig` (still untyped in astrodyn) to typed components.
# Inside the gateway (`src/**`), code under `#[cfg(test)] mod tests { ... }`
# is skipped — test fixtures are a sanctioned boundary.
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
src_files_to_scan=$( {
    find crates/astrodyn_bevy/src/ -name "*.rs" -type f \
        -not -path 'crates/astrodyn_bevy/src/components.rs' \
        -not -path 'crates/astrodyn_bevy/src/components/*' \
        -not -path 'crates/astrodyn_bevy/src/lib.rs'
    find crates/astrodyn_bevy/tests/ -name "*.rs" -type f
    # Runner construction-boundary modules are the canonical
    # composition/insertion sites — analog of the Bevy adapter's
    # `components.rs`/`lib.rs` boundary. They mint typed state from
    # untyped `VehicleConfig` / `RefFrameState` / mass-tree composition
    # inputs at config-time; per-step bypasses elsewhere in the runner
    # are still policed.
    find crates/astrodyn_runner/src/ -name "*.rs" -type f \
        -not -path 'crates/astrodyn_runner/src/simulation/types.rs' \
        -not -path 'crates/astrodyn_runner/src/simulation/bodies.rs' \
        -not -path 'crates/astrodyn_runner/src/simulation/frame_attach.rs' \
        -not -path 'crates/astrodyn_runner/src/simulation/mass_tree.rs'
    find crates/astrodyn_runner/tests/ -name "*.rs" -type f
    find src/ -name "*.rs" -type f
} | sort)

bypass_matches=$(echo "$src_files_to_scan" | xargs awk '
    FNR == 1 { prev_allowed = 0; in_test = 0; depth = 0; saw_cfg_test = 0 }
    # Track #[cfg(test)] followed by `mod ... {` so all lines inside a
    # gateway test module are skipped. The cfg attribute and `mod ... {`
    # may sit on adjacent lines; tolerate intervening blank/comment lines.
    /^[[:space:]]*#\[cfg\(test\)\]/ { saw_cfg_test = 1; next }
    saw_cfg_test && /^[[:space:]]*mod [A-Za-z_][A-Za-z_0-9]* \{/ {
        in_test = 1; depth = 1; saw_cfg_test = 0; next
    }
    in_test {
        # Brace depth tracking — every `{` increments, every `}` decrements.
        # Tolerates string and char literals containing braces because
        # gsub on the literal char counts them; in practice astrodyn rust
        # source has no such braces inside string literals, so this works.
        opens = gsub(/\{/, "{")
        closes = gsub(/\}/, "}")
        depth += opens - closes
        if (depth <= 0) { in_test = 0; depth = 0 }
        next
    }
    # Pure-comment line with `// allowed:` propagates to the next
    # non-comment, non-blank line.
    /^[[:space:]]*\/\/.*allowed:/ { prev_allowed = 1; next }
    # Pure-comment line without `// allowed:`: keep prev_allowed as-is.
    /^[[:space:]]*\/\// { next }
    # Blank: keep prev_allowed as-is.
    /^[[:space:]]*$/ { next }
    /from_untyped_unchecked|from_dmat3_unchecked|from_raw_si|SecondsSince[^[:space:]]*::from_seconds|(JeodQuat|Quat)::from_array|FrameTransform::from_matrix\(/ {
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
    echo "FAIL: typed-quantity bypass constructors in the gateway, Bevy adapter, or Bevy integration tests" >&2
    echo "  (scanned: src/**, crates/astrodyn_bevy/src/**, and" >&2
    echo "   crates/astrodyn_bevy/tests/**; canonical boundary modules" >&2
    echo "   crates/astrodyn_bevy/src/components.rs and" >&2
    echo "   crates/astrodyn_bevy/src/lib.rs are exempt; #[cfg(test)]" >&2
    echo "   modules in src/** are skipped.)" >&2
    echo "  Banned: from_untyped_unchecked / from_dmat3_unchecked / from_raw_si /" >&2
    echo "          from_seconds / (JeodQuat|Quat)::from_array / FrameTransform::from_matrix(" >&2
    echo "  These constructors mint a typed value from raw storage without" >&2
    echo "  checking the caller's frame / time-scale / quaternion-convention /" >&2
    echo "  normalization phantoms. The gateway and Bevy adapter must use the" >&2
    echo "  typed APIs (Position<RootInertial>, SecondsSince<TAI>," >&2
    echo "  NormalizedQuat<...>, FrameTransform::from_matrix_validated, etc.)" >&2
    echo "  directly, not lift raw storage per step." >&2
    echo "  See issues #172 (audit finding H1) and #388 for context." >&2
    echo "  If you have a documented boundary case, annotate the line with" >&2
    echo "  '// allowed: <reason>' (inline) or place '// allowed: <reason>'" >&2
    echo "  on a pure-comment line immediately preceding the bypass." >&2
    echo "$bypass_matches" >&2
    failed=1
fi

if [ "$failed" -ne 0 ]; then
    exit 1
fi

echo "OK: no escape-hatch markers or unsanctioned typed-quantity bypasses in the gateway or Bevy adapter"
