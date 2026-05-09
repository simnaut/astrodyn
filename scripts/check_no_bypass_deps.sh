#!/usr/bin/env bash
# CI guard: every workspace consumer of the `astrodyn` pipeline
# depends on `astrodyn` and only `astrodyn` for physics.
#
# Specifically, none of `astrodyn_runner`, `astrodyn_bevy`,
# `astrodyn_verif_jeod`, or `astrodyn_verif_parity` may declare a
# direct dependency on any `astrodyn_*` *physics* crate
# (`astrodyn_dynamics`, `astrodyn_gravity`, `astrodyn_time`,
# `astrodyn_frames`, `astrodyn_interactions`, `astrodyn_math`,
# `astrodyn_quantities`, `astrodyn_atmosphere`, `astrodyn_ephemeris`,
# `astrodyn_planet`).
#
# Allowed (non-physics workspace deps):
#   - `astrodyn` (the gateway).
#   - `astrodyn_runner` (arena harness — a gateway consumer itself).
#   - `astrodyn_bevy` (ECS adapter — a gateway consumer itself).
#   - `astrodyn_verif_jeod` (cross-validation infrastructure that
#     itself only depends on the gateway plus the runner).
#
# Owner-crate unit / Tier 2 / Tier 3 tests (e.g.
# `astrodyn_time/tests/tier3_*.rs`) live inside their owning crate's
# own `tests/` directory and reach the crate under test through normal
# in-crate test access — they are not scanned here.
set -euo pipefail

physics_crate_re='astrodyn_(dynamics|gravity|time|frames|interactions|math|quantities|atmosphere|ephemeris|planet)'

failed=0

for crate_toml in \
    crates/astrodyn_runner/Cargo.toml \
    crates/astrodyn_bevy/Cargo.toml \
    crates/astrodyn_verif_jeod/Cargo.toml \
    crates/astrodyn_verif_parity/Cargo.toml; do
    bad=$(grep -E "^${physics_crate_re}[[:space:]]*=" "$crate_toml" || true)
    if [ -n "$bad" ]; then
        echo "FAIL: $crate_toml declares a direct physics-crate dependency:" >&2
        echo "$bad" >&2
        echo "  Per CLAUDE.md, every consumer of the astrodyn pipeline" >&2
        echo "  goes through 'astrodyn' (+ bevy for the Bevy adapter; +" >&2
        echo "  astrodyn_runner / astrodyn_bevy / astrodyn_verif_jeod as" >&2
        echo "  workspace consumers themselves). If 'astrodyn' is missing" >&2
        echo "  a symbol you need, widen its re-export surface in" >&2
        echo "  src/lib.rs rather than reaching around it." >&2
        failed=1
    fi
done

if [ "$failed" -ne 0 ]; then
    exit 1
fi

echo "OK: gateway-only invariant holds for runner, bevy, and verif crates"
