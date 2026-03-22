#!/bin/bash
# Generate reference trajectory data from JEOD verification sims.
# Runs inside the Docker container with Trick and JEOD built.
# Outputs CSV files to /output/ for bevy_jeod Tier 3 cross-validation.
set -uo pipefail
# Note: -e is intentionally omitted so that individual sim failures don't
# kill the entire script. Each run_sim invocation handles its own errors.

OUTPUT_DIR="${1:-/output}"
mkdir -p "$OUTPUT_DIR"

export TRICK_HOME=/trick
export JEOD_HOME=/jeod
export PATH="${TRICK_HOME}/bin:${PATH}"
export MAKEFLAGS="-j$(nproc)"

echo "=== JEOD Reference Data Generator ==="
echo "Trick: $(trick-version 2>/dev/null || echo 'installed')"
echo "JEOD:  ${JEOD_HOME}"
echo "Output: ${OUTPUT_DIR}"
echo ""

# ── Helper: build and run a JEOD verification sim ──
run_sim() {
    local sim_dir="$1"
    local run_dir="$2"
    local label="$3"

    echo "--- Building ${label} ---"
    cd "${JEOD_HOME}/${sim_dir}" || return 1

    # Build the sim (trick-CP compiles S_define)
    if ! ls S_main*.exe >/dev/null 2>&1; then
        if ! trick-CP 2>&1 | tail -5; then
            echo "ERROR: trick-CP failed for ${label}"
            return 1
        fi
    fi

    echo "--- Running ${label} ---"

    # Run from the SIM root directory (JEOD input.py paths are relative to SIM root)
    local exe
    exe=$(ls S_main*.exe 2>/dev/null | head -1)
    if [ -z "$exe" ]; then
        echo "ERROR: No S_main executable found for ${label}"
        return 1
    fi

    if ! "./${exe}" "${run_dir}/input.py" 2>&1 | tail -3; then
        echo "ERROR: Sim execution failed for ${label}"
        return 1
    fi

    # Copy ASCII CSV output, canonicalizing key filenames so downstream
    # tests can find them at predictable paths.
    echo "--- Collecting output for ${label} ---"
    while IFS= read -r -d '' csv_file; do
        local base
        base=$(basename "$csv_file" .csv)
        # Canonicalize: strip "log_" prefix and "_ASCII" suffix for cleaner names.
        # e.g. "log_state_ASCII" -> "state", "log_Earth_RNP_ascii" -> "Earth_RNP"
        local canonical
        canonical=$(echo "$base" | sed -e 's/^log_//' -e 's/_[Aa][Ss][Cc][Ii][Ii]$//')
        local dest="${OUTPUT_DIR}/${label}_${canonical}.csv"
        cp "$csv_file" "$dest"
        echo "  -> ${dest}"
    done < <(find "${run_dir}" -name "*.csv" ! -name "_init_log.csv" -print0 2>/dev/null)

    # Convert any .trk files to CSV using Trick's data product tools
    while IFS= read -r -d '' trk_file; do
        local base
        base=$(basename "$trk_file" .trk)
        local dest="${OUTPUT_DIR}/${label}_${base}.csv"
        if command -v trick-trk2csv &>/dev/null; then
            trick-trk2csv "$trk_file" > "$dest"
            echo "  -> ${dest}"
        elif command -v trk2csv &>/dev/null; then
            trk2csv "$trk_file" > "$dest"
            echo "  -> ${dest}"
        else
            cp "$trk_file" "${OUTPUT_DIR}/${label}_${base}.trk"
            echo "  -> ${OUTPUT_DIR}/${label}_${base}.trk (binary, no trk2csv available)"
        fi
    done < <(find "${run_dir}" -name "*.trk" -print0 2>/dev/null)
    echo ""
}

# ════════════════════════════════════════════════════════════════════
# Sim 1: SIM_dyncomp RUN_2 — Spherical gravity, RK4, 8-hour ISS orbit
# Best for: Phase 1/2 translational dynamics validation
# ════════════════════════════════════════════════════════════════════
run_sim "verif/SIM_dyncomp" "SET_test/RUN_2" "dyncomp_run2" || exit 1

# ════════════════════════════════════════════════════════════════════
# Sim 2: SIM_orbinit RUN_0001 — Orbital initialization verification
# Best for: Phase 1 orbital elements validation
# ════════════════════════════════════════════════════════════════════
run_sim "models/dynamics/body_action/verif/SIM_orbinit" "SET_test/RUN_0001" "orbinit_0001" || true

# ════════════════════════════════════════════════════════════════════
# Sim 3: SIM_Euler RUN_inc — Euler angle derived state
# Best for: Phase 3 Euler angle validation
# ════════════════════════════════════════════════════════════════════
run_sim "models/dynamics/derived_state/verif/SIM_Euler" "SET_test/RUN_inc" "euler_inc" || true

# ════════════════════════════════════════════════════════════════════
# Sim 4: SIM_OrbElem — Orbital element computation
# Best for: Phase 1 orbital elements validation
# ════════════════════════════════════════════════════════════════════
run_sim "models/dynamics/derived_state/verif/SIM_OrbElem" "SET_test/RUN_circular" "orbelem_circular" || true

# ════════════════════════════════════════════════════════════════════
# Sim 5: Integration test — RK4 verification
# Best for: Phase 1 integrator accuracy
# ════════════════════════════════════════════════════════════════════
run_sim "models/utils/integration/verif/SIM_integ_test" "SET_test/RUN_rk4" "integ_rk4" || true

echo "=== Reference data generation complete ==="
echo "Files in ${OUTPUT_DIR}:"
ls -la "${OUTPUT_DIR}/"
