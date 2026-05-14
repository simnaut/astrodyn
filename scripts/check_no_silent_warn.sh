#!/usr/bin/env bash
# CI guard: every `log::warn!` / `bevy::log::warn!` in the physics crates
# and the gateway must carry either:
#   - a `// JEOD_INV: XX.YY` tag (the JEOD invariant whose policy this
#     warn implements — e.g., `IG.38` for the JEOD-faithful corrector
#     non-convergence warn, `TM.41` for the leap-second clamp warn,
#     `GV.08` for the gradient-degree auto-correction warn), or
#   - a `// FAIL_LOUD_EXEMPT: <reason>` annotation (the warn is the
#     operational report path of a known is_warning()-class
#     ValidationError, not a JEOD invariant).
#
# Rationale (#485 T1): warn-then-continue is the dominant escape hatch
# from the fail-loudly rule (CLAUDE.md "Fail Loudly"). #485 C1, C2, H2
# closed the worst-offender drift; this lint prevents new warn sites
# from being introduced without an explicit policy-documented reason.
#
# Scope: `crates/astrodyn_*/src/**` and `src/**`. Test code under
# `tests/`, examples under `examples/`, and `#[cfg(test)]` modules are
# not scanned — warn output is acceptable there (and uses `log::warn!`
# infrequently anyway).
#
# Tag placement: the tag must appear on the same line as the warn macro
# (rare — the macro is usually too long), on the immediately preceding
# code/comment line, or on a pure-comment line within 3 lines above the
# warn (to handle the multi-line `if ... { warn!(...); }` shape where
# the tag sits above the `if`).
set -euo pipefail

src_files_to_scan=$( {
    find crates -name "*.rs" -type f \
        -path 'crates/astrodyn_*/src/*'
    find src -name "*.rs" -type f
} | sort)

# Awk script:
# - Tracks the last 5 lines of context.
# - On a `log::warn!` / `bevy::log::warn!` / `warn!(` match in scope,
#   scans the previous 5 lines for `JEOD_INV: XX.YY` or `FAIL_LOUD_EXEMPT:`.
# - Skips inside `#[cfg(test)] mod ... { ... }` blocks (same shape as
#   check_no_escape_hatches.sh).
silent_warns=$(echo "$src_files_to_scan" | xargs awk '
    FNR == 1 {
        for (i = 1; i <= 10; i++) history[i] = ""
        in_test = 0; depth = 0; saw_cfg_test = 0
    }
    # cfg(test) mod tracking — same shape as check_no_escape_hatches.sh.
    /^[[:space:]]*#\[cfg\(test\)\]/ { saw_cfg_test = 1; next }
    saw_cfg_test && /^[[:space:]]*mod [A-Za-z_][A-Za-z_0-9]* \{/ {
        in_test = 1; depth = 1; saw_cfg_test = 0; next
    }
    in_test {
        opens = gsub(/\{/, "{")
        closes = gsub(/\}/, "}")
        depth += opens - closes
        if (depth <= 0) { in_test = 0; depth = 0 }
        next
    }
    # Maintain a 10-line history buffer for tag lookback. The longer
    # window handles JEOD_INV tags placed above a multi-line `if {
    # warn!(...); }` block (the tag sits at the comment heading the
    # whole policy block, possibly 5+ lines above the warn macro).
    {
        for (i = 10; i > 1; i--) history[i] = history[i - 1]
        history[1] = $0
    }
    # Doc comments (`///` or `//!`) that mention the warn macro
    # textually are documentation, not call sites. Skip them.
    /^[[:space:]]*\/\/[\/!]/ { next }
    # Match warn-macro invocations. `warn!(` covers `use log::warn` /
    # `use bevy::log::warn` imports; `log::warn!` and `bevy::log::warn!`
    # cover fully-qualified call sites.
    /log::warn!|bevy::log::warn!|^[[:space:]]*warn!\(/ {
        # Look in the current line and the 9-line preceding history
        # for an acceptable tag.
        ok = 0
        for (i = 1; i <= 10; i++) {
            if (history[i] ~ /JEOD_INV:[[:space:]]*[A-Z]+\.[0-9]+/ ||
                history[i] ~ /FAIL_LOUD_EXEMPT:/) {
                ok = 1
                break
            }
        }
        if (!ok) {
            printf "%s:%d: %s\n", FILENAME, FNR, $0
        }
    }
')

if [ -n "$silent_warns" ]; then
    echo "FAIL: log::warn! / bevy::log::warn! sites without a JEOD_INV or FAIL_LOUD_EXEMPT tag:" >&2
    echo "$silent_warns" >&2
    echo "" >&2
    echo "  Every warn site in crates/astrodyn_*/src/ or src/ must carry one of:" >&2
    echo "    // JEOD_INV: XX.YY — the JEOD invariant this warn implements (see" >&2
    echo "       docs/JEOD_invariants.md). Use this when the warn is the" >&2
    echo "       JEOD-faithful side of an opt-in fail-loudly divergence (IG.38," >&2
    echo "       TM.41) or a JEOD-faithful auto-correction (GV.07–GV.11)." >&2
    echo "    // FAIL_LOUD_EXEMPT: <reason> — the warn is the operational" >&2
    echo "       report path of a known is_warning()-class ValidationError" >&2
    echo "       (or similar bounded-scope diagnostic) and has no JEOD" >&2
    echo "       invariant counterpart." >&2
    echo "" >&2
    echo "  See #485 T1 for the rationale. The lint window is 5 lines above the" >&2
    echo "  warn macro; place the tag on the closest comment line." >&2
    exit 1
fi

echo "OK: every warn! in physics + gateway carries a JEOD_INV or FAIL_LOUD_EXEMPT tag"
