#!/usr/bin/env bash
# CI guard: every workspace consumer of the `astrodyn` pipeline
# depends on `astrodyn` and only `astrodyn` for physics.
#
# Specifically, neither `astrodyn_runner` nor `astrodyn_bevy` may
# declare a direct dependency on any `astrodyn_*` *physics* crate
# (`astrodyn_dynamics`, `astrodyn_gravity`, `astrodyn_time`,
# `astrodyn_frames`, `astrodyn_interactions`, `astrodyn_math`,
# `astrodyn_quantities`, `astrodyn_atmosphere`, `astrodyn_ephemeris`,
# `astrodyn_planet`).
#
# Allowed:
#   - `astrodyn` (the gateway).
#   - `astrodyn_runner` as a `[dev-dependencies]` entry on
#     `astrodyn_bevy` (parity-style tests, per CLAUDE.md).
#   - `astrodyn_verif_jeod` (verification crate; explicit exception).
#
# Verification crates (`astrodyn_verif_*`) are not scanned: by design
# they reach physics crates directly to construct test fixtures.
set -euo pipefail

physics_crate_re='astrodyn_(dynamics|gravity|time|frames|interactions|math|quantities|atmosphere|ephemeris|planet)'

failed=0

for crate_toml in crates/astrodyn_runner/Cargo.toml crates/astrodyn_bevy/Cargo.toml; do
    bad=$(grep -E "^${physics_crate_re}[[:space:]]*=" "$crate_toml" || true)
    if [ -n "$bad" ]; then
        echo "FAIL: $crate_toml declares a direct physics-crate dependency:" >&2
        echo "$bad" >&2
        echo "  Per CLAUDE.md, every consumer of the astrodyn pipeline" >&2
        echo "  goes through 'astrodyn' (+ bevy for the Bevy adapter; +" >&2
        echo "  astrodyn_runner as a dev-dep on astrodyn_bevy for parity)." >&2
        echo "  If 'astrodyn' is missing a symbol you need, widen its" >&2
        echo "  re-export surface in src/lib.rs rather than reaching around it." >&2
        failed=1
    fi
done

if [ "$failed" -ne 0 ]; then
    exit 1
fi

echo "OK: astrodyn_runner and astrodyn_bevy declare no direct astrodyn_* physics-crate deps"
