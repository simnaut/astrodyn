#!/bin/bash
# Generate reference trajectory data from JEOD verification sims.
# Runs inside the Docker container with Trick and JEOD built.
# Outputs CSV files to /output/ for bevy_jeod Tier 3 cross-validation.
#
# Parallelization strategy:
#   - SIM_dyncomp runs share one executable → run sequentially after one build
#   - Derived-state sims (SIM_OrbElem, SIM_LVLH, etc.) each have their own
#     S_define → build and run in parallel with each other
#   - SIM_integ_test is independent → runs in parallel
set -uo pipefail
# Note: -e is intentionally omitted so that individual sim failures don't
# kill the entire script. Each run_sim invocation handles its own errors.

OUTPUT_DIR="${1:-/output}"
mkdir -p "$OUTPUT_DIR"

# Skip generation for labels that already have CSV output in OUTPUT_DIR.
# Set FORCE=1 to regenerate everything: FORCE=1 ./generate_references.sh
FORCE="${FORCE:-0}"

has_output() {
    local label="$1"
    local required="${2:-}"  # optional: specific file that must exist
    # When FORCE=1, always report "no valid output" so data is regenerated.
    if [ "$FORCE" = "1" ]; then
        return 1
    fi
    if [ -n "$required" ]; then
        # Check for a specific required file (non-empty).
        # Use this for sims that produce multiple CSVs where only one
        # is critical (e.g. dyncomp _state.csv).
        [ -s "${OUTPUT_DIR}/${required}" ]
    else
        # Fallback: any non-empty CSV for this label.
        # Safe for sims that produce exactly one CSV (derived-state sims).
        find "$OUTPUT_DIR" -maxdepth 1 -type f -name "${label}_*.csv" ! -size 0c 2>/dev/null | grep -q .
    fi
}

export TRICK_HOME=/trick
export JEOD_HOME=/jeod
export PATH="${TRICK_HOME}/bin:${PATH}"
export MAKEFLAGS="-j$(nproc)"

echo "=== JEOD Reference Data Generator ==="
echo "Trick: $(trick-version 2>/dev/null || echo 'installed')"
echo "JEOD:  ${JEOD_HOME}"
echo "Output: ${OUTPUT_DIR}"
if [ "$FORCE" = "1" ]; then
    echo "Mode:   FORCE (regenerating all)"
else
    echo "Mode:   incremental (skipping existing outputs)"
fi
echo ""

# ── Helper: build and run a JEOD verification sim ──
run_sim() {
    local sim_dir="$1"
    local run_dir="$2"
    local label="$3"
    local required="${4:-}"  # optional: specific file to check (e.g. dyncomp_run2_state.csv)

    if has_output "$label" "$required"; then
        echo "--- Skipping ${label} (output exists in ${OUTPUT_DIR}) ---"
        return 0
    fi

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
    local required="${5:-}"   # optional: specific file to check (e.g. label_snippet.csv)

    if has_output "$label" "$required"; then
        echo "--- Skipping ${label} (output exists in ${OUTPUT_DIR}) ---"
        return 0
    fi

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
# GROUP 1: SIM_dyncomp — one build, many runs (sequential within group)
# All share the same S_define and working directory.
# ════════════════════════════════════════════════════════════════════
run_dyncomp_group() {
    # Single source of truth: RUN_DIR:label:primary_file mapping.
    # Used for both the pre-scan (skip if all present) and execution.
    # Primary file is the CSV that Tier 3 tests actually load — checking
    # only this file prevents skipping when a partial run left behind
    # supplementary files (e.g. _Earth_RNP.csv) but not the critical one.
    local -a DYNCOMP_RUNS=(
        "SET_test/RUN_2:dyncomp_run2:dyncomp_run2_state.csv"
        "SET_test/RUN_3A:dyncomp_run3a:dyncomp_run3a_state.csv"
        "SET_test/RUN_3B:dyncomp_run3b:dyncomp_run3b_state.csv"
        "SET_test/RUN_8B:dyncomp_run8b:dyncomp_run8b_state.csv"
        "SET_test/RUN_9A:dyncomp_run9a:dyncomp_run9a_state.csv"
        "SET_test/RUN_9B:dyncomp_run9b:dyncomp_run9b_state.csv"
        "SET_test/RUN_5A:dyncomp_run5a:dyncomp_run5a_state.csv"
        "SET_test/RUN_6B:dyncomp_run6b:dyncomp_run6b_state.csv"
        "SET_test/RUN_10A:dyncomp_run10a:dyncomp_run10a_state.csv"
        "SET_test/RUN_10B:dyncomp_run10b:dyncomp_run10b_state.csv"
        # Phase 4a additions (zero build cost — same SIM_dyncomp executable)
        "SET_test/RUN_5B:dyncomp_run5b:dyncomp_run5b_state.csv"
        "SET_test/RUN_5C:dyncomp_run5c:dyncomp_run5c_state.csv"
        "SET_test/RUN_6A:dyncomp_run6a:dyncomp_run6a_state.csv"
        "SET_test/RUN_9C:dyncomp_run9c:dyncomp_run9c_state.csv"
        "SET_test/RUN_9D:dyncomp_run9d:dyncomp_run9d_state.csv"
        "SET_test/RUN_10C:dyncomp_run10c:dyncomp_run10c_state.csv"
        "SET_test/RUN_10D:dyncomp_run10d:dyncomp_run10d_state.csv"
        # Phase 4b-A additions (combined-force; consumed by Phase 5 tests)
        "SET_test/RUN_4:dyncomp_run4:dyncomp_run4_state.csv"
        # Phase 5c: polar motion validation (identical to RUN_2 but enable_polar=True)
        "SET_test/RUN_2P:dyncomp_run2p:dyncomp_run2p_state.csv"
        "SET_test/RUN_7A:dyncomp_run7a:dyncomp_run7a_state.csv"
        "SET_test/RUN_7B:dyncomp_run7b:dyncomp_run7b_state.csv"
        "SET_test/RUN_7C:dyncomp_run7c:dyncomp_run7c_state.csv"
        "SET_test/RUN_7D:dyncomp_run7d:dyncomp_run7d_state.csv"
    )

    # Skip entire group (including build) if all primary outputs exist
    local needs_build=0
    for entry in "${DYNCOMP_RUNS[@]}"; do
        IFS=: read -r _run_dir label primary <<< "$entry"
        if ! has_output "$label" "$primary"; then
            needs_build=1
            break
        fi
    done
    if [ "$needs_build" = "0" ]; then
        echo "=== Skipping SIM_dyncomp group (all outputs exist) ==="
        return 0
    fi

    # Build once
    echo "=== Building SIM_dyncomp ==="
    cd "${JEOD_HOME}/verif/SIM_dyncomp" || return 1
    if ! ls S_main*.exe >/dev/null 2>&1; then
        trick-CP 2>&1 | tail -5 || return 1
    fi

    # Run all dyncomp sims sequentially (share working directory).
    # Continue past individual failures so later sims still run,
    # but track failures so the caller can detect partial generation.
    local fail=0
    for entry in "${DYNCOMP_RUNS[@]}"; do
        IFS=: read -r run_dir label primary <<< "$entry"
        run_sim "verif/SIM_dyncomp" "$run_dir" "$label" "$primary" || fail=1
    done

    # Validate critical first output
    local expected="${OUTPUT_DIR}/dyncomp_run2_state.csv"
    if [ ! -f "$expected" ]; then
        echo "FATAL: Expected output file not found: $expected"
        return 1
    fi
    local lines
    lines=$(wc -l < "$expected")
    if [ "$lines" -lt 100 ]; then
        echo "FATAL: $expected has only $lines lines (expected 400+)"
        return 1
    fi
    echo "Validated: $expected ($lines lines)"
    return $fail
}

# ════════════════════════════════════════════════════════════════════
# GROUP 2: Derived-state sims — each has its own S_define.
# These can run in parallel with each other and with GROUP 1.
# ════════════════════════════════════════════════════════════════════

# --- Snippets for ASCII logging (defined upfront) ---

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
        dr.add_variable(f"{prefix}.dyn_body.composite_body.state.trans.position[{i}]")
        dr.add_variable(f"{prefix}.dyn_body.composite_body.state.trans.velocity[{i}]")
trick.add_data_record_group(dr)
'

SOLARBETA_SNIPPET='
dr = trick.sim_services.DRAscii("solarbeta_ASCII")
dr.set_cycle(5400)
dr.freq = trick.sim_services.DR_Always
dr.add_variable("veh.solar_beta.solar_beta")
for i in range(3):
    dr.add_variable(f"veh.dyn_body.composite_body.state.trans.position[{i}]")
    dr.add_variable(f"veh.dyn_body.composite_body.state.trans.velocity[{i}]")
trick.add_data_record_group(dr)
'

EULER_SNIPPET='
dr = trick.sim_services.DRAscii("euler_ASCII")
dr.set_cycle(12)
dr.freq = trick.sim_services.DR_Always
for seq in ["euler_rpy", "euler_pyr_lvlh", "euler_rpy_lvlh", "euler_ypr_lvlh", "euler_ryp_lvlh", "euler_yrp_lvlh"]:
    for form in ["ref_body_angles", "body_ref_angles"]:
        for i in range(3):
            dr.add_variable(f"veh.{seq}.{form}[{i}]")
for i in range(3):
    dr.add_variable(f"veh.dyn_body.composite_body.state.trans.position[{i}]")
    dr.add_variable(f"veh.dyn_body.composite_body.state.trans.velocity[{i}]")
for i in range(3):
    for j in range(3):
        dr.add_variable(f"veh.dyn_body.composite_body.state.rot.T_parent_this[{i}][{j}]")
for i in range(3):
    dr.add_variable(f"veh.dyn_body.composite_body.state.rot.Q_parent_this.vector[{i}]")
dr.add_variable("veh.dyn_body.composite_body.state.rot.Q_parent_this.scalar")
trick.add_data_record_group(dr)
'

# ── ASCII logging snippet for SIM_3_ORBIT (radiation pressure) ──
# SIM_3_ORBIT uses DRBinary by default. We inject a DRAscii logger to get CSV
# output with position, velocity, gravity accel, SRP force, and flux.
SRP_ORBIT_SNIPPET='
dr = trick.sim_services.DRAscii("srp_orbit_ASCII")
dr.set_cycle(1000.0)
dr.freq = trick.sim_services.DR_Always
for ii in range(3):
    dr.add_variable("vehicle.dyn_body.structure.state.trans.position[" + str(ii) + "]")
for ii in range(3):
    dr.add_variable("vehicle.dyn_body.structure.state.trans.velocity[" + str(ii) + "]")
for ii in range(3):
    dr.add_variable("vehicle.dyn_body.grav_interaction.grav_accel[" + str(ii) + "]")
for ii in range(3):
    dr.add_variable("radiation.rad_pressure.force[" + str(ii) + "]")
for ii in range(3):
    dr.add_variable("radiation.rad_pressure.torque[" + str(ii) + "]")
dr.add_variable("radiation.rad_pressure.source.flux_mag")
trick.add_data_record_group(dr)
'

# ── ASCII logging snippet for SIM_torque_compare_simple (gravity torque) ──
# Logs gravity torque, state, and angular velocity at 1-second resolution.
TORQUE_SIMPLE_SNIPPET='
dr = trick.sim_services.DRAscii("torque_simple_ASCII")
dr.set_cycle(1.0)
dr.freq = trick.sim_services.DR_Always
for ii in range(3):
    dr.add_variable("sv_dyn.body.composite_body.state.trans.position[" + str(ii) + "]")
for ii in range(3):
    dr.add_variable("sv_dyn.body.composite_body.state.trans.velocity[" + str(ii) + "]")
for ii in range(3):
    dr.add_variable("sv_dyn.body.composite_body.state.rot.ang_vel_this[" + str(ii) + "]")
for ii in range(3):
    for jj in range(3):
        dr.add_variable("sv_dyn.body.composite_body.state.rot.T_parent_this[" + str(ii) + "][" + str(jj) + "]")
for ii in range(4):
    if ii < 3:
        dr.add_variable("sv_dyn.body.composite_body.state.rot.Q_parent_this.vector[" + str(ii) + "]")
    else:
        dr.add_variable("sv_dyn.body.composite_body.state.rot.Q_parent_this.scalar")
for ii in range(3):
    dr.add_variable("sv_dyn.grav_torque.torque[" + str(ii) + "]")
trick.add_data_record_group(dr)
'

# ── ASCII logging snippet for SIM_2_SHADOW_CALC (eclipse geometry) ──
# Logs vehicle position, flux magnitude, and radiation force/torque.
SHADOW_CALC_SNIPPET='
dr = trick.sim_services.DRAscii("shadow_calc_ASCII")
dr.set_cycle(1.0)
dr.freq = trick.sim_services.DR_Always
for ii in range(3):
    dr.add_variable("vehicle.dyn_body.structure.state.trans.position[" + str(ii) + "]")
dr.add_variable("radiation.rad_pressure.source.flux_mag")
for ii in range(3):
    dr.add_variable("radiation.rad_pressure.force[" + str(ii) + "]")
for ii in range(3):
    dr.add_variable("radiation.rad_pressure.torque[" + str(ii) + "]")
trick.add_data_record_group(dr)
'

# ── ASCII logging snippet for SIM_orbinit (body initialization) ──
# Logs position and velocity of the initialized body.
ORBINIT_SNIPPET='
dr = trick.sim_services.DRAscii("orbinit_ASCII")
dr.set_cycle(1.0)
dr.freq = trick.sim_services.DR_Always
for ii in range(3):
    dr.add_variable("target.dyn_body.composite_body.state.trans.position[" + str(ii) + "]")
for ii in range(3):
    dr.add_variable("target.dyn_body.composite_body.state.trans.velocity[" + str(ii) + "]")
trick.add_data_record_group(dr)
'

# ── ASCII logging snippet for SIM_VER_DRAG (aerodynamic drag verification) ──
# Logs aggregate aero force/torque, inertial velocity, and acceleration magnitude.
DRAG_SNIPPET='
dr = trick.sim_services.DRAscii("drag_ASCII")
dr.set_cycle(1.0)
dr.freq = trick.sim_services.DR_Always
for ii in range(3):
    dr.add_variable("aero_test.aero_drag.aero_force[" + str(ii) + "]")
for ii in range(3):
    dr.add_variable("aero_test.aero_drag.aero_torque[" + str(ii) + "]")
for ii in range(3):
    dr.add_variable("aero_test.inertial_vel[" + str(ii) + "]")
dr.add_variable("aero_test.logging.accel_mag")
trick.add_data_record_group(dr)
'

# ── ASCII logging snippet for SIM_1_BASIC (basic SRP verification) ──
# Logs radiation force/torque, flux magnitude, and surface temperature.
SRP_BASIC_SNIPPET='
dr = trick.sim_services.DRAscii("srp_basic_ASCII")
dr.set_cycle(1.0)
dr.freq = trick.sim_services.DR_Always
for ii in range(3):
    dr.add_variable("radiation.rad_pressure.force[" + str(ii) + "]")
for ii in range(3):
    dr.add_variable("radiation.rad_pressure.torque[" + str(ii) + "]")
dr.add_variable("radiation.rad_pressure.source.flux_mag")
dr.add_variable("radiation_simple.rad_surface.temperature")
trick.add_data_record_group(dr)
'

# ── ASCII logging snippet for SIM_2A_SHADOW_CALC (advanced shadow geometry) ──
# SIM_2A uses "radiation_simple" object (not "radiation" like SIM_2_SHADOW_CALC).
SHADOW_2A_SNIPPET='
dr = trick.sim_services.DRAscii("shadow_calc_ASCII")
dr.set_cycle(1.0)
dr.freq = trick.sim_services.DR_Always
for ii in range(3):
    dr.add_variable("vehicle.dyn_body.structure.state.trans.position[" + str(ii) + "]")
dr.add_variable("radiation_simple.rad_pressure.source.flux_mag")
for ii in range(3):
    dr.add_variable("radiation_simple.rad_pressure.force[" + str(ii) + "]")
for ii in range(3):
    dr.add_variable("radiation_simple.rad_pressure.torque[" + str(ii) + "]")
trick.add_data_record_group(dr)
'

# ════════════════════════════════════════════════════════════════════
# LAUNCH ALL GROUPS IN PARALLEL
# ════════════════════════════════════════════════════════════════════
echo "=== Launching sim groups in parallel ==="

# Group 1: SIM_dyncomp (sequential internally)
run_dyncomp_group &
PID_DYNCOMP=$!

# Group 2: SIM_orbinit (multiple initialization methods, sequential within group)
run_orbinit_group() {
    local sim_dir="models/dynamics/body_action/verif/SIM_orbinit"
    local -a RUNS=(
        "SET_test/RUN_0001:orbinit_0001:orbinit_0001_orbinit.csv"
        # Phase 4b-B additions (4 initialization methods)
        "SET_test/RUN_0101:orbinit_0101:orbinit_0101_orbinit.csv"
        "SET_test/RUN_0201:orbinit_0201:orbinit_0201_orbinit.csv"
        "SET_test/RUN_0301:orbinit_0301:orbinit_0301_orbinit.csv"
        "SET_test/RUN_0401:orbinit_0401:orbinit_0401_orbinit.csv"
    )
    local needs_build=0
    for entry in "${RUNS[@]}"; do
        IFS=: read -r _run_dir label required <<< "$entry"
        if ! has_output "$label" "$required"; then
            needs_build=1
            break
        fi
    done
    if [ "$needs_build" = "0" ]; then
        echo "=== Skipping SIM_orbinit group (all outputs exist) ==="
        return 0
    fi
    local fail=0
    for entry in "${RUNS[@]}"; do
        IFS=: read -r run_dir label required <<< "$entry"
        run_sim_with_ascii "$sim_dir" "$run_dir" "$label" "$ORBINIT_SNIPPET" "$required" || fail=1
    done
    return $fail
}
run_orbinit_group &
PID_ORBINIT=$!

# Group 3: SIM_OrbElem
run_sim_with_ascii "models/dynamics/derived_state/verif/SIM_OrbElem" "SET_test/RUN_ecc" "orbelem_ecc" "$ORBELEM_SNIPPET" &
PID_ORBELEM=$!

# Group 4: SIM_LVLH (multiple orbit types, sequential within group)
run_lvlh_group() {
    local sim_dir="models/dynamics/derived_state/verif/SIM_LVLH"
    local -a RUNS=(
        "SET_test/RUN_inc:lvlh_inc:lvlh_inc_lvlh.csv"
        # Phase 4b-B additions
        "SET_test/RUN_ecc:lvlh_ecc:lvlh_ecc_lvlh.csv"
        "SET_test/RUN_equ:lvlh_equ:lvlh_equ_lvlh.csv"
    )
    local needs_build=0
    for entry in "${RUNS[@]}"; do
        IFS=: read -r _run_dir label required <<< "$entry"
        if ! has_output "$label" "$required"; then
            needs_build=1
            break
        fi
    done
    if [ "$needs_build" = "0" ]; then
        echo "=== Skipping SIM_LVLH group (all outputs exist) ==="
        return 0
    fi
    local fail=0
    for entry in "${RUNS[@]}"; do
        IFS=: read -r run_dir label required <<< "$entry"
        run_sim_with_ascii "$sim_dir" "$run_dir" "$label" "$LVLH_SNIPPET" "$required" || fail=1
    done
    return $fail
}
run_lvlh_group &
PID_LVLH=$!

# Group 5: SIM_NED (multiple orbit types + Earth models, sequential within group)
run_ned_group() {
    local sim_dir="models/dynamics/derived_state/verif/SIM_NED"
    local -a RUNS=(
        "SET_test/RUN_ell_inc:ned_ell_inc:ned_ell_inc_ned.csv"
        # Phase 4b-B additions
        "SET_test/RUN_ell_polar:ned_ell_polar:ned_ell_polar_ned.csv"
        "SET_test/RUN_sph_inc:ned_sph_inc:ned_sph_inc_ned.csv"
        "SET_test/RUN_sph_polar:ned_sph_polar:ned_sph_polar_ned.csv"
    )
    local needs_build=0
    for entry in "${RUNS[@]}"; do
        IFS=: read -r _run_dir label required <<< "$entry"
        if ! has_output "$label" "$required"; then
            needs_build=1
            break
        fi
    done
    if [ "$needs_build" = "0" ]; then
        echo "=== Skipping SIM_NED group (all outputs exist) ==="
        return 0
    fi
    local fail=0
    for entry in "${RUNS[@]}"; do
        IFS=: read -r run_dir label required <<< "$entry"
        run_sim_with_ascii "$sim_dir" "$run_dir" "$label" "$NED_SNIPPET" "$required" || fail=1
    done
    return $fail
}
run_ned_group &
PID_NED=$!

# Group 6: SIM_SolarBeta (multiple inclinations, sequential within group)
run_solarbeta_group() {
    local sim_dir="models/dynamics/derived_state/verif/SIM_SolarBeta"
    local -a RUNS=(
        "SET_test/RUN_incl_51_6:solarbeta_incl_51_6:solarbeta_incl_51_6_solarbeta.csv"
        # Phase 4b-B additions
        "SET_test/RUN_incl_0:solarbeta_incl_0:solarbeta_incl_0_solarbeta.csv"
        "SET_test/RUN_incl_23_4:solarbeta_incl_23_4:solarbeta_incl_23_4_solarbeta.csv"
        "SET_test/RUN_comp_ISS:solarbeta_comp_iss:solarbeta_comp_iss_solarbeta.csv"
    )
    local needs_build=0
    for entry in "${RUNS[@]}"; do
        IFS=: read -r _run_dir label required <<< "$entry"
        if ! has_output "$label" "$required"; then
            needs_build=1
            break
        fi
    done
    if [ "$needs_build" = "0" ]; then
        echo "=== Skipping SIM_SolarBeta group (all outputs exist) ==="
        return 0
    fi
    local fail=0
    for entry in "${RUNS[@]}"; do
        IFS=: read -r run_dir label required <<< "$entry"
        run_sim_with_ascii "$sim_dir" "$run_dir" "$label" "$SOLARBETA_SNIPPET" "$required" || fail=1
    done
    return $fail
}
run_solarbeta_group &
PID_SOLARBETA=$!

# Group 7: SIM_Euler (multiple orbit types, sequential within group)
run_euler_group() {
    local sim_dir="models/dynamics/derived_state/verif/SIM_Euler"
    local -a RUNS=(
        "SET_test/RUN_inc:euler_inc:euler_inc_euler.csv"
        # Phase 4b-B additions
        "SET_test/RUN_ecc:euler_ecc:euler_ecc_euler.csv"
        "SET_test/RUN_equ:euler_equ:euler_equ_euler.csv"
    )
    local needs_build=0
    for entry in "${RUNS[@]}"; do
        IFS=: read -r _run_dir label required <<< "$entry"
        if ! has_output "$label" "$required"; then
            needs_build=1
            break
        fi
    done
    if [ "$needs_build" = "0" ]; then
        echo "=== Skipping SIM_Euler group (all outputs exist) ==="
        return 0
    fi
    local fail=0
    for entry in "${RUNS[@]}"; do
        IFS=: read -r run_dir label required <<< "$entry"
        run_sim_with_ascii "$sim_dir" "$run_dir" "$label" "$EULER_SNIPPET" "$required" || fail=1
    done
    return $fail
}
run_euler_group &
PID_EULER=$!

# Group 8: SIM_integ_test
run_sim "models/utils/integration/verif/SIM_integ_test" "SET_test/RUN_rk4" "integ_rk4" &
PID_INTEG=$!

# Group 9: SIM_3_ORBIT (radiation pressure SRP verification)
run_sim_with_ascii "models/interactions/radiation_pressure/verif/SIM_3_ORBIT" "SET_test/RUN_radiation" "srp_orbit_radiation" "$SRP_ORBIT_SNIPPET" &
PID_SRP_ORBIT=$!

# Group 10: SIM_torque_compare_simple (high-resolution gravity torque, 6 runs)
run_torque_compare_simple_group() {
    local sim_dir="models/interactions/gravity_torque/verif/SIM_torque_compare_simple"
    local -a RUNS=(
        "SET_test/RUN_01:torque_simple_run01:torque_simple_run01_torque_simple.csv"
        "SET_test/RUN_02:torque_simple_run02:torque_simple_run02_torque_simple.csv"
        "SET_test/RUN_03:torque_simple_run03:torque_simple_run03_torque_simple.csv"
        "SET_test/RUN_04:torque_simple_run04:torque_simple_run04_torque_simple.csv"
        "SET_test/RUN_05:torque_simple_run05:torque_simple_run05_torque_simple.csv"
        "SET_test/RUN_06:torque_simple_run06:torque_simple_run06_torque_simple.csv"
    )
    local needs_build=0
    for entry in "${RUNS[@]}"; do
        IFS=: read -r _run_dir label required <<< "$entry"
        if ! has_output "$label" "$required"; then
            needs_build=1
            break
        fi
    done
    if [ "$needs_build" = "0" ]; then
        echo "=== Skipping SIM_torque_compare_simple group (all outputs exist) ==="
        return 0
    fi
    local fail=0
    for entry in "${RUNS[@]}"; do
        IFS=: read -r run_dir label required <<< "$entry"
        run_sim_with_ascii "$sim_dir" "$run_dir" "$label" "$TORQUE_SIMPLE_SNIPPET" "$required" || fail=1
    done
    return $fail
}
run_torque_compare_simple_group &
PID_TORQUE_SIMPLE=$!

# Group 11: SIM_2_SHADOW_CALC (eclipse geometry, 2 runs)
run_shadow_calc_group() {
    local sim_dir="models/interactions/radiation_pressure/verif/SIM_2_SHADOW_CALC"
    local -a RUNS=(
        "SET_test/RUN_annular_eclipse:shadow_annular_eclipse:shadow_annular_eclipse_shadow_calc.csv"
        "SET_test/RUN_transverse_shadow:shadow_transverse_shadow:shadow_transverse_shadow_shadow_calc.csv"
    )
    local needs_build=0
    for entry in "${RUNS[@]}"; do
        IFS=: read -r _run_dir label required <<< "$entry"
        if ! has_output "$label" "$required"; then
            needs_build=1
            break
        fi
    done
    if [ "$needs_build" = "0" ]; then
        echo "=== Skipping SIM_2_SHADOW_CALC group (all outputs exist) ==="
        return 0
    fi
    local fail=0
    for entry in "${RUNS[@]}"; do
        IFS=: read -r run_dir label required <<< "$entry"
        run_sim_with_ascii "$sim_dir" "$run_dir" "$label" "$SHADOW_CALC_SNIPPET" "$required" || fail=1
    done
    return $fail
}
run_shadow_calc_group &
PID_SHADOW_CALC=$!

# Group 12: SIM_VER_DRAG (aerodynamic drag verification, 3 Cd modes)
# Phase 4b-C — requires its own trick-CP build
run_drag_group() {
    local sim_dir="models/interactions/aerodynamics/verif/SIM_VER_DRAG"
    local -a RUNS=(
        "SET_test/RUN_aero_drag_const:drag_const:drag_const_drag.csv"
        "SET_test/RUN_aero_drag_CD:drag_cd:drag_cd_drag.csv"
        "SET_test/RUN_aero_drag_BC:drag_bc:drag_bc_drag.csv"
    )
    local needs_build=0
    for entry in "${RUNS[@]}"; do
        IFS=: read -r _run_dir label required <<< "$entry"
        if ! has_output "$label" "$required"; then
            needs_build=1
            break
        fi
    done
    if [ "$needs_build" = "0" ]; then
        echo "=== Skipping SIM_VER_DRAG group (all outputs exist) ==="
        return 0
    fi
    local fail=0
    for entry in "${RUNS[@]}"; do
        IFS=: read -r run_dir label required <<< "$entry"
        run_sim_with_ascii "$sim_dir" "$run_dir" "$label" "$DRAG_SNIPPET" "$required" || fail=1
    done
    return $fail
}
run_drag_group &
PID_DRAG=$!

# Group 13: SIM_1_BASIC (basic SRP verification, 2 runs)
# Phase 4b-C — requires its own trick-CP build
run_srp_basic_group() {
    local sim_dir="models/interactions/radiation_pressure/verif/SIM_1_BASIC"
    local -a RUNS=(
        "SET_test/RUN_basic:srp_basic:srp_basic_srp_basic.csv"
        "SET_test/RUN_basic_cr:srp_basic_cr:srp_basic_cr_srp_basic.csv"
    )
    local needs_build=0
    for entry in "${RUNS[@]}"; do
        IFS=: read -r _run_dir label required <<< "$entry"
        if ! has_output "$label" "$required"; then
            needs_build=1
            break
        fi
    done
    if [ "$needs_build" = "0" ]; then
        echo "=== Skipping SIM_1_BASIC group (all outputs exist) ==="
        return 0
    fi
    local fail=0
    for entry in "${RUNS[@]}"; do
        IFS=: read -r run_dir label required <<< "$entry"
        run_sim_with_ascii "$sim_dir" "$run_dir" "$label" "$SRP_BASIC_SNIPPET" "$required" || fail=1
    done
    return $fail
}
run_srp_basic_group &
PID_SRP_BASIC=$!

# Group 14: SIM_2A_SHADOW_CALC (advanced shadow with thermal effects)
# Phase 4b-C — different S_define from SIM_2_SHADOW_CALC, needs own build
run_shadow_2a_group() {
    local sim_dir="models/interactions/radiation_pressure/verif/SIM_2A_SHADOW_CALC"
    local -a RUNS=(
        "SET_test/RUN_annular_eclipse:shadow_2a_annular:shadow_2a_annular_shadow_calc.csv"
        "SET_test/RUN_shadow_cooling:shadow_2a_cooling:shadow_2a_cooling_shadow_calc.csv"
    )
    local needs_build=0
    for entry in "${RUNS[@]}"; do
        IFS=: read -r _run_dir label required <<< "$entry"
        if ! has_output "$label" "$required"; then
            needs_build=1
            break
        fi
    done
    if [ "$needs_build" = "0" ]; then
        echo "=== Skipping SIM_2A_SHADOW_CALC group (all outputs exist) ==="
        return 0
    fi
    local fail=0
    for entry in "${RUNS[@]}"; do
        IFS=: read -r run_dir label required <<< "$entry"
        run_sim_with_ascii "$sim_dir" "$run_dir" "$label" "$SHADOW_2A_SNIPPET" "$required" || fail=1
    done
    return $fail
}
run_shadow_2a_group &
PID_SHADOW_2A=$!

# Group 15: SIM_3_ORBIT_1st_ORDER (first-order SRP model)
# Phase 4b-C — different S_define from SIM_3_ORBIT, needs own build
run_sim_with_ascii "models/interactions/radiation_pressure/verif/SIM_3_ORBIT_1st_ORDER" \
    "SET_test/RUN_radiation" "srp_1st_order_radiation" "$SRP_ORBIT_SNIPPET" &
PID_SRP_1ST_ORDER=$!

# Group 16: SIM_tide_verif (solid body tides, Phase 5e)
TIDE_SNIPPET='
dr = trick.sim_services.DRAscii("tide_ASCII")
dr.set_cycle(60)
dr.freq = trick.sim_services.DR_Always
for v in [
    "sv_dyn.dyn_body.composite_body.state.trans.position[0]",
    "sv_dyn.dyn_body.composite_body.state.trans.position[1]",
    "sv_dyn.dyn_body.composite_body.state.trans.position[2]",
    "sv_dyn.dyn_body.composite_body.state.trans.velocity[0]",
    "sv_dyn.dyn_body.composite_body.state.trans.velocity[1]",
    "sv_dyn.dyn_body.composite_body.state.trans.velocity[2]",
    "earth.sb_tide.dC20",
]:
    dr.add_variable(v)
trick.add_data_record_group(dr, trick.DR_Buffer)
'

run_tide_group() {
    local sim_path="models/environment/gravity/verif/SIM_tide_verif"
    local runs=(
        "SET_test/RUN_01:tide_run01:tide_run01_tide_ASCII.csv"
        "SET_test/RUN_02:tide_run02:tide_run02_tide_ASCII.csv"
    )
    local needs_build=0
    for entry in "${runs[@]}"; do
        IFS=: read -r _run_dir label primary <<< "$entry"
        if ! has_output "$label" "$primary"; then
            needs_build=1; break
        fi
    done
    if [ "$needs_build" = "0" ]; then
        echo "=== Skipping SIM_tide_verif group (all outputs exist) ==="; return 0
    fi
    local fail=0
    for entry in "${runs[@]}"; do
        IFS=: read -r run_dir label primary <<< "$entry"
        run_sim_with_ascii "$sim_path" "$run_dir" "$label" "$TIDE_SNIPPET" || fail=1
    done
    return $fail
}
run_tide_group &
PID_TIDE=$!

# Group 17: SIM_integ_test (Gauss-Jackson reference, Phase 5f)
INTEG_SNIPPET='
dr = trick.sim_services.DRAscii("integ_ASCII")
dr.set_cycle(60)
dr.freq = trick.sim_services.DR_Always
for v in [
    "test.orbit.prop_integ_state.position[0]",
    "test.orbit.prop_integ_state.position[1]",
    "test.orbit.prop_integ_state.position[2]",
    "test.orbit.prop_integ_state.velocity[0]",
    "test.orbit.prop_integ_state.velocity[1]",
    "test.orbit.prop_integ_state.velocity[2]",
    "test.orbit.true_canon_state.position[0]",
    "test.orbit.true_canon_state.position[1]",
    "test.orbit.true_canon_state.position[2]",
    "test.orbit.true_canon_state.velocity[0]",
    "test.orbit.true_canon_state.velocity[1]",
    "test.orbit.true_canon_state.velocity[2]",
]:
    dr.add_variable(v)
trick.add_data_record_group(dr, trick.DR_Buffer)
'

run_sim_with_ascii "models/utils/integration/verif/SIM_integ_test" \
    "SET_test/RUN_gauss_jackson" "integ_gj" "$INTEG_SNIPPET" &
PID_INTEG_GJ=$!

# ════════════════════════════════════════════════════════════════════
# WAIT FOR ALL GROUPS
# ════════════════════════════════════════════════════════════════════
echo "=== Waiting for all sim groups to complete ==="
FAIL=0

wait $PID_DYNCOMP       || { echo "WARN: SIM_dyncomp group had failures"; FAIL=1; }
wait $PID_ORBINIT       || { echo "WARN: SIM_orbinit group had failures"; FAIL=1; }
wait $PID_ORBELEM       || { echo "WARN: SIM_OrbElem failed"; FAIL=1; }
wait $PID_LVLH          || { echo "WARN: SIM_LVLH group had failures"; FAIL=1; }
wait $PID_NED           || { echo "WARN: SIM_NED group had failures"; FAIL=1; }
wait $PID_SOLARBETA     || { echo "WARN: SIM_SolarBeta group had failures"; FAIL=1; }
wait $PID_EULER         || { echo "WARN: SIM_Euler group had failures"; FAIL=1; }
wait $PID_INTEG         || { echo "WARN: SIM_integ_test failed"; FAIL=1; }
wait $PID_SRP_ORBIT     || { echo "WARN: SIM_3_ORBIT SRP failed"; FAIL=1; }
wait $PID_TORQUE_SIMPLE || { echo "WARN: SIM_torque_compare_simple failed"; FAIL=1; }
wait $PID_SHADOW_CALC   || { echo "WARN: SIM_2_SHADOW_CALC failed"; FAIL=1; }
# Phase 4b-C additions
wait $PID_DRAG          || { echo "WARN: SIM_VER_DRAG group had failures"; FAIL=1; }
wait $PID_SRP_BASIC     || { echo "WARN: SIM_1_BASIC group had failures"; FAIL=1; }
wait $PID_SHADOW_2A     || { echo "WARN: SIM_2A_SHADOW_CALC group had failures"; FAIL=1; }
wait $PID_SRP_1ST_ORDER || { echo "WARN: SIM_3_ORBIT_1st_ORDER failed"; FAIL=1; }
# Phase 5e-5f additions
wait $PID_TIDE          || { echo "WARN: SIM_tide_verif group had failures"; FAIL=1; }
wait $PID_INTEG_GJ      || { echo "WARN: SIM_integ_test GJ failed"; FAIL=1; }

echo ""
echo "=== Reference data generation complete ==="
echo "Files in ${OUTPUT_DIR}:"
ls -la "${OUTPUT_DIR}/"

exit $FAIL
