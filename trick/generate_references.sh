#!/bin/bash
# Generate reference trajectory data from JEOD verification sims.
# Runs inside the Docker container with Trick and JEOD built.
# Outputs CSV files to /output/ for bevy_jeod Tier 3 cross-validation.
set -euo pipefail

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
    cd "${JEOD_HOME}/${sim_dir}"

    # Build the sim (trick-CP compiles S_define)
    if [ ! -f S_main*.exe ]; then
        trick-CP 2>&1 | tail -5
    fi

    echo "--- Running ${label} ---"
    cd "${JEOD_HOME}/${sim_dir}/${run_dir}"

    # Run the simulation
    local exe=$(ls ../../S_main*.exe 2>/dev/null | head -1)
    if [ -z "$exe" ]; then
        echo "ERROR: No S_main executable found for ${label}"
        return 1
    fi

    "$exe" input.py 2>&1 | tail -3

    # Convert any .trk files to CSV using Trick's data product tools
    echo "--- Converting output for ${label} ---"
    for trk_file in $(find . -name "*.trk" 2>/dev/null); do
        local base=$(basename "$trk_file" .trk)
        local csv_file="${OUTPUT_DIR}/${label}_${base}.csv"
        # trick-trk2csv converts Trick binary data to CSV
        if command -v trick-trk2csv &>/dev/null; then
            trick-trk2csv "$trk_file" > "$csv_file"
        elif command -v trk2csv &>/dev/null; then
            trk2csv "$trk_file" > "$csv_file"
        else
            # Fallback: copy .trk as-is and note it needs conversion
            cp "$trk_file" "${OUTPUT_DIR}/${label}_${base}.trk"
            echo "  WARN: No trk2csv found, copied binary .trk file"
        fi
        echo "  -> ${csv_file}"
    done
    echo ""
}

# ════════════════════════════════════════════════════════════════════
# Sim 1: SIM_dyncomp RUN_2 — Spherical gravity, RK4, 8-hour ISS orbit
# Best for: Phase 1/2 translational dynamics validation
# ════════════════════════════════════════════════════════════════════
run_sim "verif/SIM_dyncomp" "SET_test/RUN_2" "dyncomp_run2"

# ════════════════════════════════════════════════════════════════════
# Sim 2: SIM_orbinit RUN_0001 — Orbital initialization verification
# Best for: Phase 1 orbital elements validation
# ════════════════════════════════════════════════════════════════════
run_sim "models/dynamics/body_action/verif/SIM_orbinit" "SET_test/RUN_0001" "orbinit_0001"

# ════════════════════════════════════════════════════════════════════
# Sim 3: SIM_Euler RUN_inc — Euler angle derived state
# Best for: Phase 3 Euler angle validation
# ════════════════════════════════════════════════════════════════════
run_sim "models/dynamics/derived_state/verif/SIM_Euler" "SET_test/RUN_inc" "euler_inc"

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
