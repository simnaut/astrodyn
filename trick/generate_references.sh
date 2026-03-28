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

# ── Helper: run a sim with an injected DRAscii logger for CSV output. ──
# Creates a temporary wrapper input that exec's the original and adds ASCII logging.
run_sim_with_ascii() {
    local sim_dir="$1"
    local run_dir="$2"
    local label="$3"
    local ascii_snippet="$4"  # Python code to create DRAscii logger

    echo "--- Building ${label} ---"
    cd "${JEOD_HOME}/${sim_dir}" || return 1

    if ! ls S_main*.exe >/dev/null 2>&1; then
        if ! trick-CP 2>&1 | tail -5; then
            echo "ERROR: trick-CP failed for ${label}"
            return 1
        fi
    fi

    echo "--- Running ${label} (with ASCII logging) ---"

    # Create wrapper that sources original input then adds ASCII logger
    local wrapper="${run_dir}/input_ascii_wrapper.py"
    cat > "$wrapper" << PYEOF
import sys, os
# Execute original input.py first
exec(compile(open("${run_dir}/input.py", "rb").read(), "${run_dir}/input.py", "exec"))
# Add ASCII data recorder
${ascii_snippet}
PYEOF

    local exe
    exe=$(ls S_main*.exe 2>/dev/null | head -1)
    if [ -z "$exe" ]; then
        echo "ERROR: No S_main executable found for ${label}"
        rm -f "$wrapper"
        return 1
    fi

    if ! "./${exe}" "${wrapper}" 2>&1 | tail -3; then
        echo "ERROR: Sim execution failed for ${label}"
        rm -f "$wrapper"
        return 1
    fi
    rm -f "$wrapper"

    # Collect CSV output (ASCII logger produces CSV directly, no .trk conversion needed)
    echo "--- Collecting output for ${label} ---"
    while IFS= read -r -d '' csv_file; do
        local base
        base=$(basename "$csv_file" .csv)
        local canonical
        canonical=$(echo "$base" | sed -e 's/^log_//' -e 's/_[Aa][Ss][Cc][Ii][Ii]$//')
        local dest="${OUTPUT_DIR}/${label}_${canonical}.csv"
        cp "$csv_file" "$dest"
        echo "  -> ${dest}"
    done < <(find "${run_dir}" -name "*.csv" ! -name "_init_log.csv" -print0 2>/dev/null)
    echo ""
}

# ════════════════════════════════════════════════════════════════════
# Sim 1: SIM_dyncomp RUN_2 — Spherical gravity, RK4, 8-hour ISS orbit
# Best for: Phase 1/2 translational dynamics validation
# ════════════════════════════════════════════════════════════════════
run_sim "verif/SIM_dyncomp" "SET_test/RUN_2" "dyncomp_run2" || exit 1

# Validate critical output
EXPECTED_CSV="${OUTPUT_DIR}/dyncomp_run2_state.csv"
if [ ! -f "$EXPECTED_CSV" ]; then
    echo "FATAL: Expected output file not found: $EXPECTED_CSV"
    exit 1
fi
LINE_COUNT=$(wc -l < "$EXPECTED_CSV")
if [ "$LINE_COUNT" -lt 100 ]; then
    echo "FATAL: $EXPECTED_CSV has only $LINE_COUNT lines (expected 400+)"
    exit 1
fi
echo "Validated: $EXPECTED_CSV ($LINE_COUNT lines)"

# ════════════════════════════════════════════════════════════════════
# Sim 2: SIM_dyncomp RUN_3A — 4x4 spherical harmonics gravity, 8-hour orbit
# Best for: Phase 2 spherical harmonics validation
# ════════════════════════════════════════════════════════════════════
run_sim "verif/SIM_dyncomp" "SET_test/RUN_3A" "dyncomp_run3a" || true

# ════════════════════════════════════════════════════════════════════
# Sim 3: SIM_dyncomp RUN_3B — 8x8 spherical harmonics gravity, 8-hour orbit
# Best for: Phase 2 spherical harmonics validation (higher fidelity)
# ════════════════════════════════════════════════════════════════════
run_sim "verif/SIM_dyncomp" "SET_test/RUN_3B" "dyncomp_run3b" || true

# ════════════════════════════════════════════════════════════════════
# Sim 4: SIM_dyncomp RUN_8B — Rotational dynamics, spherical mass
# Best for: Phase 3 6-DOF rotational dynamics validation
# Spherical mass body with LVLH orbital-rate init, spherical gravity,
# no torques, 8 hours. Cleanest rotational dynamics test case.
# ════════════════════════════════════════════════════════════════════
run_sim "verif/SIM_dyncomp" "SET_test/RUN_8B" "dyncomp_run8b" || true

# ════════════════════════════════════════════════════════════════════
# Sim 5: SIM_orbinit RUN_0001 — Orbital initialization verification
# Best for: Phase 1 orbital elements validation
# ════════════════════════════════════════════════════════════════════
run_sim "models/dynamics/body_action/verif/SIM_orbinit" "SET_test/RUN_0001" "orbinit_0001" || true

# ════════════════════════════════════════════════════════════════════
# Sim: SIM_OrbElem RUN_ecc — Orbital element computation
# Best for: Phase 3a orbital elements cross-validation
# ════════════════════════════════════════════════════════════════════
ORBELEM_SNIPPET='
dr = trick.sim_services.DRAscii("orbelem_ASCII")
dr.set_cycle(12)
dr.freq = trick.sim_services.DR_Always
for v in [
    "veh.orb_elem.elements.semi_major_axis",
    "veh.orb_elem.elements.semiparam",
    "veh.orb_elem.elements.e_mag",
    "veh.orb_elem.elements.inclination",
    "veh.orb_elem.elements.arg_periapsis",
    "veh.orb_elem.elements.long_asc_node",
    "veh.orb_elem.elements.r_mag",
    "veh.orb_elem.elements.vel_mag",
    "veh.orb_elem.elements.true_anom",
    "veh.orb_elem.elements.mean_anom",
    "veh.orb_elem.elements.mean_motion",
    "veh.orb_elem.elements.orbital_anom",
    "veh.orb_elem.elements.orb_energy",
    "veh.orb_elem.elements.orb_ang_momentum",
    "veh.dyn_body.composite_body.state.trans.position[0]",
    "veh.dyn_body.composite_body.state.trans.position[1]",
    "veh.dyn_body.composite_body.state.trans.position[2]",
    "veh.dyn_body.composite_body.state.trans.velocity[0]",
    "veh.dyn_body.composite_body.state.trans.velocity[1]",
    "veh.dyn_body.composite_body.state.trans.velocity[2]",
]:
    dr.add_variable(v)
trick.add_data_record_group(dr)
'
run_sim_with_ascii "models/dynamics/derived_state/verif/SIM_OrbElem" "SET_test/RUN_ecc" "orbelem_ecc" "$ORBELEM_SNIPPET" || true

# ════════════════════════════════════════════════════════════════════
# Sim: SIM_LVLH RUN_inc — LVLH frame computation
# Best for: Phase 3a LVLH frame cross-validation
# ════════════════════════════════════════════════════════════════════
LVLH_SNIPPET='
dr = trick.sim_services.DRAscii("lvlh_ASCII")
dr.set_cycle(12)
dr.freq = trick.sim_services.DR_Always
for prefix in ["vehA", "vehB"]:
    for i in range(3):
        for j in range(3):
            dr.add_variable(f"{prefix}.lvlh.lvlh_frame.state.rot.T_parent_this[{i}][{j}]")
    dr.add_variable(f"{prefix}.lvlh.lvlh_frame.state.rot.ang_vel_mag")
    for i in range(3):
        dr.add_variable(f"{prefix}.dyn_body.composite_body.state.trans.position[{i}]")
        dr.add_variable(f"{prefix}.dyn_body.composite_body.state.trans.velocity[{i}]")
trick.add_data_record_group(dr)
'
run_sim_with_ascii "models/dynamics/derived_state/verif/SIM_LVLH" "SET_test/RUN_inc" "lvlh_inc" "$LVLH_SNIPPET" || true

# ════════════════════════════════════════════════════════════════════
# Sim: SIM_NED RUN_ell_inc — NED / geodetic coordinate computation
# Best for: Phase 3a geodetic cross-validation
# ════════════════════════════════════════════════════════════════════
NED_SNIPPET='
dr = trick.sim_services.DRAscii("ned_ASCII")
dr.set_cycle(12)
dr.freq = trick.sim_services.DR_Always
for prefix in ["vehA"]:
    for i in range(3):
        dr.add_variable(f"{prefix}.ned.ned_state.cart_coords[{i}]")
    for coord in ["altitude", "latitude", "longitude"]:
        dr.add_variable(f"{prefix}.ned.ned_state.ellip_coords.{coord}")
        dr.add_variable(f"{prefix}.ned.ned_state.sphere_coords.{coord}")
    for i in range(3):
        dr.add_variable(f"{prefix}.dyn_body.structure.state.trans.position[{i}]")
        dr.add_variable(f"{prefix}.dyn_body.structure.state.trans.velocity[{i}]")
trick.add_data_record_group(dr)
'
run_sim_with_ascii "models/dynamics/derived_state/verif/SIM_NED" "SET_test/RUN_ell_inc" "ned_ell_inc" "$NED_SNIPPET" || true

# ════════════════════════════════════════════════════════════════════
# Sim: SIM_SolarBeta RUN_incl_51_6 — Solar beta angle
# Best for: Phase 3a solar beta cross-validation
# ════════════════════════════════════════════════════════════════════
SOLARBETA_SNIPPET='
dr = trick.sim_services.DRAscii("solarbeta_ASCII")
dr.set_cycle(5400)
dr.freq = trick.sim_services.DR_Always
dr.add_variable("veh.solar_beta.solar_beta")
for i in range(3):
    dr.add_variable(f"veh.dyn_body.structure.state.trans.position[{i}]")
    dr.add_variable(f"veh.dyn_body.structure.state.trans.velocity[{i}]")
trick.add_data_record_group(dr)
'
run_sim_with_ascii "models/dynamics/derived_state/verif/SIM_SolarBeta" "SET_test/RUN_incl_51_6" "solarbeta_incl_51_6" "$SOLARBETA_SNIPPET" || true

# ════════════════════════════════════════════════════════════════════
# Sim: SIM_Euler RUN_inc — Euler angle derived state
# Best for: Phase 3a Euler angle cross-validation
# ════════════════════════════════════════════════════════════════════
EULER_SNIPPET='
dr = trick.sim_services.DRAscii("euler_ASCII")
dr.set_cycle(12)
dr.freq = trick.sim_services.DR_Always
for seq in ["euler_rpy", "euler_pyr_lvlh", "euler_rpy_lvlh", "euler_ypr_lvlh", "euler_ryp_lvlh", "euler_yrp_lvlh"]:
    for form in ["ref_body_angles", "body_ref_angles"]:
        for i in range(3):
            dr.add_variable(f"veh.{seq}.{form}[{i}]")
for i in range(3):
    dr.add_variable(f"veh.dyn_body.structure.state.trans.position[{i}]")
    dr.add_variable(f"veh.dyn_body.structure.state.trans.velocity[{i}]")
for i in range(3):
    for j in range(3):
        dr.add_variable(f"veh.dyn_body.composite_body.state.rot.T_parent_this[{i}][{j}]")
for i in range(3):
    dr.add_variable(f"veh.dyn_body.composite_body.state.rot.Q_parent_this.vector[{i}]")
dr.add_variable("veh.dyn_body.composite_body.state.rot.Q_parent_this.scalar")
trick.add_data_record_group(dr)
'
run_sim_with_ascii "models/dynamics/derived_state/verif/SIM_Euler" "SET_test/RUN_inc" "euler_inc" "$EULER_SNIPPET" || true

# ════════════════════════════════════════════════════════════════════
# Sim 5: Integration test — RK4 verification
# Best for: Phase 1 integrator accuracy
# ════════════════════════════════════════════════════════════════════
run_sim "models/utils/integration/verif/SIM_integ_test" "SET_test/RUN_rk4" "integ_rk4" || true

# ════════════════════════════════════════════════════════════════════
# Phase 4 Sims: Interactions (gravity torque, drag, SRP)
# ════════════════════════════════════════════════════════════════════

# ── Sim: SIM_dyncomp RUN_9A — Gravity gradient torque (ISS inertia) ──
# Best for: Phase 4 gravity torque validation
# ISS-like body with gravity gradient torque enabled, 8-hour propagation.
run_sim "verif/SIM_dyncomp" "SET_test/RUN_9A" "dyncomp_run9a" || true

# ── Sim: SIM_dyncomp RUN_9B — Gravity gradient torque (asymmetric body) ──
# Best for: Phase 4 gravity torque validation (different inertia)
run_sim "verif/SIM_dyncomp" "SET_test/RUN_9B" "dyncomp_run9b" || true

# ── Sim: SIM_dyncomp RUN_5A — Drag enabled (if available) ──
# Best for: Phase 4 aerodynamic drag trajectory validation
run_sim "verif/SIM_dyncomp" "SET_test/RUN_5A" "dyncomp_run5a" || true

echo "=== Reference data generation complete ==="
echo "Files in ${OUTPUT_DIR}:"
ls -la "${OUTPUT_DIR}/"
