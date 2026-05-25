#!/bin/bash
# Generate reference trajectory data from JEOD verification sims.
# Runs inside the Docker container with Trick and JEOD built.
# Outputs CSV files to /output/ for astrodyn_bevy Tier 3 cross-validation.
#
# Parallelization strategy:
#   - SIM_dyncomp runs share one executable → run sequentially after one build
#   - Derived-state sims (SIM_OrbElem, SIM_LVLH, etc.) each have their own
#     S_define → build and run in parallel with each other
#   - SIM_integ_test is independent → runs in parallel
set -uo pipefail
# Note: -e is intentionally omitted so that individual sim failures don't
# kill the entire script. Each run_sim invocation handles its own errors.

# Require Bash >= 4.3 for `wait -n` (used by throttled_bg).
if [[ "${BASH_VERSINFO[0]}" -lt 4 || ( "${BASH_VERSINFO[0]}" -eq 4 && "${BASH_VERSINFO[1]}" -lt 3 ) ]]; then
    echo "ERROR: Bash >= 4.3 required for 'wait -n'. Found: ${BASH_VERSION}" >&2
    echo "On macOS, install a newer bash: brew install bash" >&2
    exit 1
fi

OUTPUT_DIR="${1:-/output}"
mkdir -p "$OUTPUT_DIR"

# Skip generation for labels that already have CSV output in OUTPUT_DIR.
# Set FORCE=1 to regenerate everything: FORCE=1 ./generate_references.sh
FORCE="${FORCE:-0}"

# ── Parallelism throttle ──
# trick-CP builds are memory-hungry (~1-2 GB each). Limit concurrent builds
# to avoid OOM on machines with limited RAM. Set MAX_PARALLEL=1 for serial.
MAX_PARALLEL="${MAX_PARALLEL:-4}"
RUNNING_PIDS=()

# Launch a background job, blocking if MAX_PARALLEL slots are full.
throttled_bg() {
    # Reap finished PIDs
    local alive=()
    for pid in "${RUNNING_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            alive+=("$pid")
        fi
    done
    RUNNING_PIDS=("${alive[@]}")

    # Wait for a slot if at capacity
    while [ "${#RUNNING_PIDS[@]}" -ge "$MAX_PARALLEL" ]; do
        # Wait for any one child to finish
        wait -n 2>/dev/null || true
        alive=()
        for pid in "${RUNNING_PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                alive+=("$pid")
            fi
        done
        RUNNING_PIDS=("${alive[@]}")
    done

    # Launch the command in background and track its PID.
    # Set LAST_BG_PID instead of echoing — command substitution would
    # capture sim stdout into the PID variable and risk deadlock.
    "$@" &
    LAST_BG_PID=$!
    RUNNING_PIDS+=("$LAST_BG_PID")
}

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

# ── Helper: build sim, run with injected DRAscii wrapper, collect CSVs. ──
# No skip-check — caller is responsible for deciding whether to invoke this.
# Used by both run_sim_with_ascii (which adds the standard skip-check) and
# run_apollo_group (which has a multi-output skip-check and additional .out
# collection that wraps this core).
_run_sim_with_ascii_impl() {
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
}

# ── Helper: run a sim with an injected DRAscii logger for CSV output. ──
# Creates a temporary wrapper input that exec's the original and adds ASCII
# logging. Skips if the requested output already exists in $OUTPUT_DIR.
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

    _run_sim_with_ascii_impl "$sim_dir" "$run_dir" "$label" "$ascii_snippet" || return 1
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
        "SET_test/RUN_6C:dyncomp_run6c:dyncomp_run6c_state.csv"
        "SET_test/RUN_6D:dyncomp_run6d:dyncomp_run6d_state.csv"
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
        # Multi-attach lifecycle: 8x8 SH + sun/moon + drag + grav-grad torque
        # + chained attach/detach pairs + mid-trajectory set_state rewinds.
        # 12000 s window (extended past common_input.py's 28800 by the run's
        # own trick.stop(stop_time) wired through chkpt_restart_times.py).
        "SET_test/RUN_attach_to_ref_frame:dyncomp_run_attach_to_ref_frame:dyncomp_run_attach_to_ref_frame_state.csv"
    )

    # Skip entire group (including build) if all primary outputs exist.
    # This includes both base DYNCOMP_RUNS and ASCII-injected runs so that
    # adding a new ASCII run triggers a rebuild when its output is missing.
    local needs_build=0
    for entry in "${DYNCOMP_RUNS[@]}"; do
        IFS=: read -r _run_dir label primary <<< "$entry"
        if ! has_output "$label" "$primary"; then
            needs_build=1
            break
        fi
    done
    # Also check ASCII-injected run outputs
    if [ "$needs_build" = "0" ]; then
        for required in \
            "dyncomp_run5a_atmos_atmos_traj.csv" \
            "dyncomp_run6b_aero_aero_traj.csv" \
            "dyncomp_run6b_rot_aero_traj.csv" \
        ; do
            if ! [ -s "${OUTPUT_DIR}/${required}" ]; then
                needs_build=1
                break
            fi
        done
    fi
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

    # ── Additional ASCII-injected runs for trajectory + interaction data ──
    # These reuse the SIM_dyncomp executable built above. Each adds an ASCII
    # data recorder alongside the existing JEOD run configuration, capturing
    # interaction-specific columns (aero forces, density) that the default
    # log_suite only writes to binary .trk files.

    # RUN_5A with atmosphere trajectory (pos + vel + density + temperature)
    run_sim_with_ascii "verif/SIM_dyncomp" "SET_test/RUN_5A" \
        "dyncomp_run5a_atmos" "$DYNCOMP_ATMOS_SNIPPET" \
        "dyncomp_run5a_atmos_atmos_traj.csv" || fail=1

    # RUN_6B with drag trajectory (pos + vel + aero_force + aero_torque + density)
    run_sim_with_ascii "verif/SIM_dyncomp" "SET_test/RUN_6B" \
        "dyncomp_run6b_aero" "$DYNCOMP_AERO_SNIPPET" \
        "dyncomp_run6b_aero_aero_traj.csv" || fail=1

    # RUN_6B with rotated structural frame (15 deg about [1,1,1]) — issue #14
    run_sim_with_ascii "verif/SIM_dyncomp" "SET_test/RUN_6B" \
        "dyncomp_run6b_rot" "$DYNCOMP_AERO_ROT_SNIPPET" \
        "dyncomp_run6b_rot_aero_traj.csv" || fail=1

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

# ── ASCII logging snippet for SIM_csr_compare (gravity-accel octant sweep) ──
# Non-integrating vehicle teleported through octant positions at t=1..5;
# logs gravity potential + acceleration + position at 1 Hz (matches the
# DRBinary "gravity_compare" group the input.py defines). The cross-check
# (#207) evaluates our GGM05C 70x70 accel at these positions vs grav_accel.
CSR_COMPARE_SNIPPET='
dr = trick.sim_services.DRAscii("csr_compare_ASCII")
dr.set_cycle(1.0)
dr.freq = trick.sim_services.DR_Always
dr.add_variable("vehicle.dyn_body.grav_interaction.grav_pot")
for ii in range(3):
    dr.add_variable("vehicle.dyn_body.grav_interaction.grav_accel[" + str(ii) + "]")
for ii in range(3):
    dr.add_variable("vehicle.dyn_body.composite_body.state.trans.position[" + str(ii) + "]")
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

# ── ASCII logging snippet for SIM_dyncomp atmosphere (RUN_5A/5B/5C) ──
# Logs trajectory + atmosphere state. Used for tier3_sim_met.
DYNCOMP_ATMOS_SNIPPET='
dr = trick.sim_services.DRAscii("atmos_traj_ASCII")
dr.set_cycle(60.0)
dr.freq = trick.sim_services.DR_Always
for ii in range(3):
    dr.add_variable("vehicle.dyn_body.composite_body.state.trans.position[" + str(ii) + "]")
for ii in range(3):
    dr.add_variable("vehicle.dyn_body.composite_body.state.trans.velocity[" + str(ii) + "]")
dr.add_variable("vehicle.atmos_state.density")
dr.add_variable("vehicle.atmos_state.temperature")
trick.add_data_record_group(dr)
'

# ── ASCII logging snippet for SIM_dyncomp aerodynamic drag (RUN_6B) ──
# Logs trajectory + aero force/torque + density. Used for tier3_sim_drag_verif.
DYNCOMP_AERO_SNIPPET='
dr = trick.sim_services.DRAscii("aero_traj_ASCII")
dr.set_cycle(60.0)
dr.freq = trick.sim_services.DR_Always
for ii in range(3):
    dr.add_variable("vehicle.dyn_body.composite_body.state.trans.position[" + str(ii) + "]")
for ii in range(3):
    dr.add_variable("vehicle.dyn_body.composite_body.state.trans.velocity[" + str(ii) + "]")
for ii in range(3):
    dr.add_variable("vehicle.aero_drag.aero_force[" + str(ii) + "]")
for ii in range(3):
    dr.add_variable("vehicle.aero_drag.aero_torque[" + str(ii) + "]")
dr.add_variable("vehicle.atmos_state.density")
trick.add_data_record_group(dr)
'

# ── ASCII logging snippet for SIM_dyncomp drag with rotated structural frame ──
# Same logging as DYNCOMP_AERO_SNIPPET but overrides eigen_angle to 15 deg about
# [1,1,1] (normalized) before sim start. Used for tier3_sim_drag_rot_verif (issue #14).
DYNCOMP_AERO_ROT_SNIPPET='
import math
# Override structural-to-body orientation: 15 deg about [1,1,1] normalized
vehicle.mass_init.properties.pt_orientation.data_source = trick.Orientation.InputEigenRotation
vehicle.mass_init.properties.pt_orientation.eigen_angle = trick.attach_units("degree", 15.0)
inv_sqrt3 = 1.0 / math.sqrt(3.0)
vehicle.mass_init.properties.pt_orientation.eigen_axis = [inv_sqrt3, inv_sqrt3, inv_sqrt3]

dr = trick.sim_services.DRAscii("aero_traj_ASCII")
dr.set_cycle(60.0)
dr.freq = trick.sim_services.DR_Always
for ii in range(3):
    dr.add_variable("vehicle.dyn_body.composite_body.state.trans.position[" + str(ii) + "]")
for ii in range(3):
    dr.add_variable("vehicle.dyn_body.composite_body.state.trans.velocity[" + str(ii) + "]")
for ii in range(3):
    dr.add_variable("vehicle.aero_drag.aero_force[" + str(ii) + "]")
for ii in range(3):
    dr.add_variable("vehicle.aero_drag.aero_torque[" + str(ii) + "]")
dr.add_variable("vehicle.atmos_state.density")
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
echo "=== Launching sim groups (max ${MAX_PARALLEL} parallel) ==="

# Group 1: SIM_dyncomp (sequential internally)
throttled_bg run_dyncomp_group
PID_DYNCOMP=$LAST_BG_PID

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
        # set02 mean-anomaly parameterization (ISS + STS, inertial)
        "SET_test/RUN_0002:orbinit_0002:orbinit_0002_orbinit.csv"
        "SET_test/RUN_0102:orbinit_0102:orbinit_0102_orbinit.csv"
        # set03 semi-latus-rectum + true-anomaly parameterization (ISS + STS, inertial)
        "SET_test/RUN_0003:orbinit_0003:orbinit_0003_orbinit.csv"
        "SET_test/RUN_0103:orbinit_0103:orbinit_0103_orbinit.csv"
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
throttled_bg run_orbinit_group
PID_ORBINIT=$LAST_BG_PID

# Group 3: SIM_OrbElem
throttled_bg run_sim_with_ascii "models/dynamics/derived_state/verif/SIM_OrbElem" "SET_test/RUN_ecc" "orbelem_ecc" "$ORBELEM_SNIPPET"
PID_ORBELEM=$LAST_BG_PID

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
throttled_bg run_lvlh_group
PID_LVLH=$LAST_BG_PID

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
throttled_bg run_ned_group
PID_NED=$LAST_BG_PID

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
throttled_bg run_solarbeta_group
PID_SOLARBETA=$LAST_BG_PID

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
throttled_bg run_euler_group
PID_EULER=$LAST_BG_PID

# Group 8: SIM_integ_test — one build, many runs (sequential within group)
# Runs the integrator verification sim with multiple integrator selections.
# All runs reuse the same trick-CP executable; only the input.py changes.
#
# The orbit test (case 4 of 5) integrates a Kepler orbit with sma=6811.137 km,
# e=0, omega=1.1231543952404041e-3 rad/s. JEOD logs `true_canon_state` (true
# Kepler solution) and `prop_integ_state` (our integrator's output).
# Stop time is 80000s, log_cycle = 200s → 401 points.
#
# Data is logged via DRAscii injected by INTEG_SNIPPET below.
INTEG_SNIPPET='
dr = trick.sim_services.DRAscii("integ_ASCII")
dr.set_cycle(200)
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
    "test.orbit.rel_position_err_mag",
    "test.orbit.rel_velocity_err_mag",
    "test.orbit.rel_energy_error",
]:
    dr.add_variable(v)
trick.add_data_record_group(dr, trick.DR_Buffer)
'

run_integ_test_group() {
    local sim_dir="models/utils/integration/verif/SIM_integ_test"
    # RUN_rk4 is also listed but uses the plain (non-ASCII) run — the existing
    # integ_rk4 .trk logging remains in place. ABM4 and LSODE get ASCII CSVs
    # with the orbit test trajectory so Tier 3 tests can compare against them.
    local -a RUNS=(
        "SET_test/RUN_abm4:integ_abm4:integ_abm4_integ.csv"
        "SET_test/RUN_lsode:integ_lsode:integ_lsode_integ.csv"
    )
    local needs_build=0
    if ! has_output "integ_rk4" ""; then
        needs_build=1
    fi
    for entry in "${RUNS[@]}"; do
        IFS=: read -r _run_dir label required <<< "$entry"
        if ! has_output "$label" "$required"; then
            needs_build=1
            break
        fi
    done
    if [ "$needs_build" = "0" ]; then
        echo "=== Skipping SIM_integ_test group (all outputs exist) ==="
        return 0
    fi
    local fail=0
    # RK4 keeps the original .trk-only logging.
    run_sim "$sim_dir" "SET_test/RUN_rk4" "integ_rk4" || fail=1
    # ABM4 and LSODE use ASCII logging for the orbit test.
    for entry in "${RUNS[@]}"; do
        IFS=: read -r run_dir label required <<< "$entry"
        run_sim_with_ascii "$sim_dir" "$run_dir" "$label" "$INTEG_SNIPPET" "$required" || fail=1
    done
    return $fail
}
throttled_bg run_integ_test_group
PID_INTEG=$LAST_BG_PID

# Group 9: SIM_3_ORBIT (radiation pressure SRP verification)
throttled_bg run_sim_with_ascii "models/interactions/radiation_pressure/verif/SIM_3_ORBIT" "SET_test/RUN_radiation" "srp_orbit_radiation" "$SRP_ORBIT_SNIPPET"
PID_SRP_ORBIT=$LAST_BG_PID

# Group 9b: SIM_csr_compare (GGM05C 70x70 gravity-accel octant sweep, #207)
throttled_bg run_sim_with_ascii "models/environment/gravity/verif/SIM_csr_compare" "SET_test/RUN_01" "csr_compare_run01" "$CSR_COMPARE_SNIPPET"
PID_CSR_COMPARE=$LAST_BG_PID

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
throttled_bg run_torque_compare_simple_group
PID_TORQUE_SIMPLE=$LAST_BG_PID

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
throttled_bg run_shadow_calc_group
PID_SHADOW_CALC=$LAST_BG_PID

# Group 12: SIM_VER_DRAG (aerodynamic drag verification)
# Phase 4b-C — requires its own trick-CP build
# Covers ballistic (const/CD/BC) and flat-plate (specular/diffuse/mixed/calc_coef/torque/orbiter) runs.
run_drag_group() {
    local sim_dir="models/interactions/aerodynamics/verif/SIM_VER_DRAG"
    local -a RUNS=(
        # Ballistic (DefaultAero) runs
        "SET_test/RUN_aero_drag_const:drag_const:drag_const_drag.csv"
        "SET_test/RUN_aero_drag_CD:drag_cd:drag_cd_drag.csv"
        "SET_test/RUN_aero_drag_BC:drag_bc:drag_bc_drag.csv"
        # Flat-plate runs — exercise each coefficient method in FlatPlateAeroFacet,
        # plus a centered-vs-offset pair to exercise the torque path.
        "SET_test/RUN_one_plate_accel_spec_max_coef:drag_one_plate_spec:drag_one_plate_spec_drag.csv"
        "SET_test/RUN_one_plate_accel_diff_max_coef:drag_one_plate_diff:drag_one_plate_diff_drag.csv"
        "SET_test/RUN_one_plate_accel_mixed_eps05_max_coef:drag_one_plate_mixed:drag_one_plate_mixed_drag.csv"
        "SET_test/RUN_one_plate_accel_calc_coef_eps00:drag_one_plate_calc_eps00:drag_one_plate_calc_eps00_drag.csv"
        "SET_test/RUN_one_plate_accel_calc_coef_eps05:drag_one_plate_calc_eps05:drag_one_plate_calc_eps05_drag.csv"
        "SET_test/RUN_one_plate_accel_calc_coef_eps1:drag_one_plate_calc_eps1:drag_one_plate_calc_eps1_drag.csv"
        "SET_test/RUN_one_plate_torque:drag_one_plate_torque:drag_one_plate_torque_drag.csv"
        "SET_test/RUN_orbiter:drag_orbiter:drag_orbiter_drag.csv"
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
throttled_bg run_drag_group
PID_DRAG=$LAST_BG_PID

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
throttled_bg run_srp_basic_group
PID_SRP_BASIC=$LAST_BG_PID

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
throttled_bg run_shadow_2a_group
PID_SHADOW_2A=$LAST_BG_PID

# Group 15: SIM_3_ORBIT_1st_ORDER (first-order SRP model)
# Phase 4b-C — different S_define from SIM_3_ORBIT, needs own build
throttled_bg run_sim_with_ascii "models/interactions/radiation_pressure/verif/SIM_3_ORBIT_1st_ORDER" \
    "SET_test/RUN_radiation" "srp_1st_order_radiation" "$SRP_ORBIT_SNIPPET"
PID_SRP_1ST_ORDER=$LAST_BG_PID

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
        "SET_test/RUN_01:tide_run01:tide_run01_tide.csv"
        "SET_test/RUN_02:tide_run02:tide_run02_tide.csv"
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
throttled_bg run_tide_group
PID_TIDE=$LAST_BG_PID

# Group 17: SIM_GJ_test (Gauss-Jackson reference, Phase 5f)
# Uses SIM_GJ_test instead of SIM_integ_test (which fails to compile in
# Docker due to header incompatibility — see issue #33).
# Scenario: circular orbit, mu=5.76e14, r0=9e6m, GJ order 8, dt=1s, 300000s.
GJ_SNIPPET='
dr = trick.sim_services.DRAscii("gj_ASCII")
dr.set_cycle(300)
dr.freq = trick.sim_services.DR_Always
for v in [
    "vehicle.dyn_body.composite_body.state.trans.position[0]",
    "vehicle.dyn_body.composite_body.state.trans.position[1]",
    "vehicle.dyn_body.composite_body.state.trans.position[2]",
    "vehicle.dyn_body.composite_body.state.trans.velocity[0]",
    "vehicle.dyn_body.composite_body.state.trans.velocity[1]",
    "vehicle.dyn_body.composite_body.state.trans.velocity[2]",
]:
    dr.add_variable(v)
trick.add_data_record_group(dr, trick.DR_Buffer)
'

run_gj_group() {
    local sim_path="models/utils/integration/verif/SIM_GJ_test"
    local fail=0
    # Baseline: order 8, dt=1s
    run_sim_with_ascii "$sim_path" \
        "SET_test/RUN_GJ_step1_order8_noeval_nobs" "integ_gj" "$GJ_SNIPPET" || fail=1
    # Order 4, dt=1s
    run_sim_with_ascii "$sim_path" \
        "SET_test/RUN_GJ_step1_order4_noeval_nobs" "integ_gj_order4" "$GJ_SNIPPET" || fail=1
    # Order 12, dt=1s
    run_sim_with_ascii "$sim_path" \
        "SET_test/RUN_GJ_step1_order12_noeval_nobs" "integ_gj_order12" "$GJ_SNIPPET" || fail=1
    # Order 8, dt=10s (scale_factor=10 → cycle=30 sim-seconds = 300 dynamic-seconds)
    local GJ_SNIPPET_DT10='
dr = trick.sim_services.DRAscii("gj_ASCII")
dr.set_cycle(30)
dr.freq = trick.sim_services.DR_Always
for v in [
    "vehicle.dyn_body.composite_body.state.trans.position[0]",
    "vehicle.dyn_body.composite_body.state.trans.position[1]",
    "vehicle.dyn_body.composite_body.state.trans.position[2]",
    "vehicle.dyn_body.composite_body.state.trans.velocity[0]",
    "vehicle.dyn_body.composite_body.state.trans.velocity[1]",
    "vehicle.dyn_body.composite_body.state.trans.velocity[2]",
]:
    dr.add_variable(v)
trick.add_data_record_group(dr, trick.DR_Buffer)
'
    run_sim_with_ascii "$sim_path" \
        "SET_test/RUN_GJ_step10_order8_noeval_nobs" "integ_gj_dt10" "$GJ_SNIPPET_DT10" || fail=1
    return $fail
}
throttled_bg run_gj_group
PID_INTEG_GJ=$LAST_BG_PID

# ════════════════════════════════════════════════════════════════════
# Phase 6 additions: comprehensive JEOD parity validation
# ════════════════════════════════════════════════════════════════════

# ── Snippet: SIM_orb_elem (different object name from SIM_OrbElem) ──
# SIM_orb_elem is a static verification sim — runs for 1 second, computes
# orbital elements from input position/velocity. Object: orb_elem_test.
ORBELEM_VERIF_SNIPPET='
dr = trick.sim_services.DRAscii("orbelem_ASCII")
dr.set_cycle(1)
dr.freq = trick.sim_services.DR_Always
for v in [
    "orb_elem_test.orb_elem.semi_major_axis",
    "orb_elem_test.orb_elem.semiparam",
    "orb_elem_test.orb_elem.e_mag",
    "orb_elem_test.orb_elem.inclination",
    "orb_elem_test.orb_elem.arg_periapsis",
    "orb_elem_test.orb_elem.long_asc_node",
    "orb_elem_test.orb_elem.r_mag",
    "orb_elem_test.orb_elem.vel_mag",
    "orb_elem_test.orb_elem.true_anom",
    "orb_elem_test.orb_elem.mean_anom",
    "orb_elem_test.orb_elem.mean_motion",
    "orb_elem_test.orb_elem.orbital_anom",
    "orb_elem_test.orb_elem.orb_energy",
    "orb_elem_test.orb_elem.orb_ang_momentum",
    "orb_elem_test.orb_elem_ver.position[0]",
    "orb_elem_test.orb_elem_ver.position[1]",
    "orb_elem_test.orb_elem_ver.position[2]",
    "orb_elem_test.orb_elem_ver.velocity[0]",
    "orb_elem_test.orb_elem_ver.velocity[1]",
    "orb_elem_test.orb_elem_ver.velocity[2]",
]:
    dr.add_variable(v)
trick.add_data_record_group(dr)
'

# Group 18: SIM_orb_elem (7 representative orbit families)
run_orbelem_verif_group() {
    local sim_dir="models/utils/orbital_elements/verif/SIM_orb_elem"
    local -a RUNS=(
        "SET_test/RUN_T01_OE_VER:orbelem_verif_t01:orbelem_verif_t01_orbelem.csv"
        "SET_test/RUN_T10_OE_VER:orbelem_verif_t10:orbelem_verif_t10_orbelem.csv"
        "SET_test/RUN_T20_OE_VER:orbelem_verif_t20:orbelem_verif_t20_orbelem.csv"
        "SET_test/RUN_T30_OE_VER:orbelem_verif_t30:orbelem_verif_t30_orbelem.csv"
        "SET_test/RUN_T40_OE_VER:orbelem_verif_t40:orbelem_verif_t40_orbelem.csv"
        "SET_test/RUN_T50_OE_VER:orbelem_verif_t50:orbelem_verif_t50_orbelem.csv"
        "SET_test/RUN_T55_OE_VER:orbelem_verif_t55:orbelem_verif_t55_orbelem.csv"
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
        echo "=== Skipping SIM_orb_elem verif group (all outputs exist) ==="
        return 0
    fi
    local fail=0
    for entry in "${RUNS[@]}"; do
        IFS=: read -r run_dir label required <<< "$entry"
        run_sim_with_ascii "$sim_dir" "$run_dir" "$label" "$ORBELEM_VERIF_SNIPPET" "$required" || fail=1
    done
    return $fail
}
throttled_bg run_orbelem_verif_group
PID_ORBELEM_VERIF=$LAST_BG_PID

# ── Snippet: SIM_Planetary (orbital elements + position/velocity) ──
PLANETARY_SNIPPET='
dr = trick.sim_services.DRAscii("planetary_ASCII")
dr.set_cycle(12)
dr.freq = trick.sim_services.DR_Always
for v in [
    "veh.orb_elem.elements.semi_major_axis",
    "veh.orb_elem.elements.semiparam",
    "veh.orb_elem.elements.e_mag",
    "veh.orb_elem.elements.inclination",
    "veh.orb_elem.elements.arg_periapsis",
    "veh.orb_elem.elements.long_asc_node",
    "veh.orb_elem.elements.true_anom",
    "veh.orb_elem.elements.mean_anom",
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

# Group 19: SIM_Planetary (5 orbit regimes)
run_planetary_group() {
    local sim_dir="models/dynamics/derived_state/verif/SIM_Planetary"
    local -a RUNS=(
        "SET_test/RUN_LEO_inc:planetary_leo_inc:planetary_leo_inc_planetary.csv"
        "SET_test/RUN_LEO_polar:planetary_leo_polar:planetary_leo_polar_planetary.csv"
        "SET_test/RUN_LEO_ecc:planetary_leo_ecc:planetary_leo_ecc_planetary.csv"
        "SET_test/RUN_LEO_equ:planetary_leo_equ:planetary_leo_equ_planetary.csv"
        "SET_test/RUN_GEO:planetary_geo:planetary_geo_planetary.csv"
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
        echo "=== Skipping SIM_Planetary group (all outputs exist) ==="
        return 0
    fi
    local fail=0
    for entry in "${RUNS[@]}"; do
        IFS=: read -r run_dir label required <<< "$entry"
        run_sim_with_ascii "$sim_dir" "$run_dir" "$label" "$PLANETARY_SNIPPET" "$required" || fail=1
    done
    return $fail
}
throttled_bg run_planetary_group
PID_PLANETARY=$LAST_BG_PID

# ── Snippet: SIM_MET (atmosphere density/temperature at altitude) ──
MET_VERIF_SNIPPET='
dr = trick.sim_services.DRAscii("met_ASCII")
dr.set_cycle(1)
dr.freq = trick.sim_services.DR_Always
for v in [
    "vehicle.atmos_state.density",
    "vehicle.atmos_state.temperature",
    "vehicle.pos.ellip_coords.altitude",
    "vehicle.pos.ellip_coords.latitude",
    "vehicle.pos.ellip_coords.longitude",
]:
    dr.add_variable(v)
trick.add_data_record_group(dr)
'

# Group 20: SIM_MET (7 atmosphere validation runs)
#
# Every RUN in SIM_MET exercises the SAME jeod::METAtmosphere model — the
# S_define instantiates only a METAtmosphere (there is no GRAM or Jacchia
# model anywhere in JEOD; `models/environment/atmosphere/` contains only
# MET + base_atmos). The `_GRAM_MET` / `_JAC_COMP` RUN names refer to the
# altitude/latitude sample SCHEDULE used by external GRAM/Jacchia comparison
# studies, not to a different model under test. All RUNs therefore log the
# MET density/temperature via `vehicle.atmos_state` and are portable against
# our MetAtmosphere. RUN_data_compare is omitted: it re-runs RUN_T01_MET_VER's
# inputs purely to confirm JEOD's several MET implementations agree, adding no
# new sample points.
run_met_verif_group() {
    local sim_dir="models/environment/atmosphere/MET/verif/SIM_MET"
    local -a RUNS=(
        "SET_test/RUN_T01_MET_VER:met_t01:met_t01_met.csv"
        "SET_test/RUN_T02_MET_VER:met_t02:met_t02_met.csv"
        "SET_test/RUN_T03_GRAM_MET:met_t03_gram:met_t03_gram_met.csv"
        "SET_test/RUN_T01_GRAM_MET:met_t01_gram:met_t01_gram_met.csv"
        "SET_test/RUN_T02_GRAM_MET:met_t02_gram:met_t02_gram_met.csv"
        "SET_test/RUN_T01_JAC_COMP:met_t01_jac:met_t01_jac_met.csv"
        "SET_test/RUN_T02_JAC_COMP:met_t02_jac:met_t02_jac_met.csv"
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
        echo "=== Skipping SIM_MET verif group (all outputs exist) ==="
        return 0
    fi
    local fail=0
    for entry in "${RUNS[@]}"; do
        IFS=: read -r run_dir label required <<< "$entry"
        run_sim_with_ascii "$sim_dir" "$run_dir" "$label" "$MET_VERIF_SNIPPET" "$required" || fail=1
    done
    return $fail
}
throttled_bg run_met_verif_group
PID_MET_VERIF=$LAST_BG_PID

# ── Snippet: SIM_5_all_inclusive (all time scales) ──
TIMESCALE_SNIPPET='
dr = trick.sim_services.DRAscii("timescale_ASCII")
dr.set_cycle(60)
dr.freq = trick.sim_services.DR_Always
for v in [
    "jeod_time.tai.trunc_julian_time",
    "jeod_time.tai.seconds",
    "jeod_time.utc.trunc_julian_time",
    "jeod_time.ut1.trunc_julian_time",
    "jeod_time.tt.trunc_julian_time",
    "jeod_time.tdb.trunc_julian_time",
    "jeod_time.gmst.seconds",
    "jeod_time.gps.trunc_julian_time",
]:
    dr.add_variable(v)
trick.add_data_record_group(dr)
'

# Group 21: SIM_5_all_inclusive (2 time scale runs)
run_timescale_group() {
    local sim_dir="models/environment/time/verif/SIM_5_all_inclusive"
    local -a RUNS=(
        "SET_test/RUN_UTC_initialized:timescale_utc:timescale_utc_timescale.csv"
        "SET_test/RUN_UTC_initialized_tdb:timescale_tdb:timescale_tdb_timescale.csv"
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
        echo "=== Skipping SIM_5_all_inclusive group (all outputs exist) ==="
        return 0
    fi
    local fail=0
    for entry in "${RUNS[@]}"; do
        IFS=: read -r run_dir label required <<< "$entry"
        run_sim_with_ascii "$sim_dir" "$run_dir" "$label" "$TIMESCALE_SNIPPET" "$required" || fail=1
    done
    return $fail
}
throttled_bg run_timescale_group
PID_TIMESCALE=$LAST_BG_PID

# ── Snippet: SIM_7_time_reversal (state + time for reversal tests) ──
# Reuses the dyncomp state snippet since the sim has the same DynBody structure.
TIME_REVERSAL_SNIPPET='
dr = trick.sim_services.DRAscii("reversal_ASCII")
dr.set_cycle(60)
dr.freq = trick.sim_services.DR_Always
for v in [
    "sv_dyn.body.composite_body.state.trans.position[0]",
    "sv_dyn.body.composite_body.state.trans.velocity[0]",
    "sv_dyn.body.composite_body.state.trans.position[1]",
    "sv_dyn.body.composite_body.state.trans.velocity[1]",
    "sv_dyn.body.composite_body.state.trans.position[2]",
    "sv_dyn.body.composite_body.state.trans.velocity[2]",
    "jeod_time.tai.seconds",
    "jeod_time.tai.trunc_julian_time",
]:
    dr.add_variable(v)
trick.add_data_record_group(dr)
'

# Group 22: SIM_7_time_reversal (all 8 reversal runs)
run_time_reversal_group() {
    local sim_dir="models/environment/time/verif/SIM_7_time_reversal"
    local -a RUNS=(
        "SET_test/RUN_1:reversal_run1:reversal_run1_reversal.csv"
        "SET_test/RUN_3A:reversal_run3a:reversal_run3a_reversal.csv"
        "SET_test/RUN_3B:reversal_run3b:reversal_run3b_reversal.csv"
        "SET_test/RUN_4:reversal_run4:reversal_run4_reversal.csv"
        "SET_test/RUN_6A:reversal_run6a:reversal_run6a_reversal.csv"
        "SET_test/RUN_8B:reversal_run8b:reversal_run8b_reversal.csv"
        "SET_test/RUN_9D:reversal_run9d:reversal_run9d_reversal.csv"
        "SET_test/RUN_10A:reversal_run10a:reversal_run10a_reversal.csv"
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
        echo "=== Skipping SIM_7_time_reversal group (all outputs exist) ==="
        return 0
    fi
    local fail=0
    for entry in "${RUNS[@]}"; do
        IFS=: read -r run_dir label required <<< "$entry"
        run_sim_with_ascii "$sim_dir" "$run_dir" "$label" "$TIME_REVERSAL_SNIPPET" "$required" || fail=1
    done
    return $fail
}
throttled_bg run_time_reversal_group
PID_TIME_REVERSAL=$LAST_BG_PID

# ── Snippet: SIM_RNP_J2000_prop (RNP transform matrices, ASCII) ──
# Logs the precession / nutation / GAST-rotation component matrices, the
# composite inertial→pfix T_parent_this, the NP product, GAST angle, the
# equation of equinoxes, and the transformed test vector. All are pure
# functions of time (integrator-independent), so these validate our RNP
# model directly. Object names match the SIM_RNP_J2000_prop S_define
# (`earth.rnp.*`, `earth.logging.*`, `earth.planet.pfix.*`).
RNP_VERIF_SNIPPET='
dr = trick.sim_services.DRAscii("rnp_ASCII")
dr.set_cycle(1.0)
dr.freq = trick.sim_services.DR_Always
dr.add_variable("earth.rnp.RJ2000.theta_gast")
dr.add_variable("earth.rnp.NJ2000.equa_of_equi")
for ii in range(0,3):
    dr.add_variable("earth.output_vector[" + str(ii) + "]")
for grp in ["earth.logging.nut_trans", "earth.logging.prec_trans",
            "earth.logging.rot_trans", "earth.planet.pfix.state.rot.T_parent_this",
            "earth.rnp.NP_matrix"]:
    for ii in range(0,3):
        for jj in range(0,3):
            dr.add_variable(grp + "[" + str(ii) + "][" + str(jj) + "]")
trick.add_data_record_group(dr)
'

# Group 23: SIM_RNP_J2000_prop — RNP transform validation. Only the two RUNs
# with explicit leap-second / UT1 overrides are regenerated here; their time
# setup is exact and deterministic. The default-EOP RUNs (prop, prop_off,
# Polar_off) need JEOD's EOP/UT1 table sourced to validate without feeding
# JEOD output (computational independence) — tracked as the remainder of #99.
run_rnp_verif_group() {
    local sim_dir="models/environment/RNP/RNPJ2000/verif/SIM_RNP_J2000_prop"
    local -a RUNS=(
        "SET_test/RUN_J2000_RNP_Transform:rnp_transform:rnp_transform_rnp.csv"
        "SET_test/RUN_J2000_RNP_init:rnp_init:rnp_init_rnp.csv"
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
        echo "=== Skipping SIM_RNP_J2000_prop group (all outputs exist) ==="
        return 0
    fi
    local fail=0
    for entry in "${RUNS[@]}"; do
        IFS=: read -r run_dir label required <<< "$entry"
        run_sim_with_ascii "$sim_dir" "$run_dir" "$label" "$RNP_VERIF_SNIPPET" "$required" || fail=1
    done
    return $fail
}
throttled_bg run_rnp_verif_group
PID_RNP_VERIF=$LAST_BG_PID

# ── Snippet: SIM_Relative (relative state between two vehicles) ──
# Override frame names to use composite_body (matches our logged states) instead
# of the default RefPoint frames configured in input_common.py.
RELATIVE_SNIPPET='
rel_state.vehA_wrt_vehB_in_B.subject_frame_name = "vehicleA.composite_body"
rel_state.vehA_wrt_vehB_in_B.target_frame_name  = "vehicleB.composite_body"

dr = trick.sim_services.DRAscii("relative_ASCII")
dr.set_cycle(1)
dr.freq = trick.sim_services.DR_Always
for prefix in ["vehA", "vehB"]:
    for i in range(3):
        dr.add_variable(f"{prefix}.dyn_body.composite_body.state.trans.position[{i}]")
        dr.add_variable(f"{prefix}.dyn_body.composite_body.state.trans.velocity[{i}]")
    for i in range(4):
        dr.add_variable(f"{prefix}.dyn_body.composite_body.state.rot.Q_parent_this.scalar")
        dr.add_variable(f"{prefix}.dyn_body.composite_body.state.rot.Q_parent_this.vector[0]")
        dr.add_variable(f"{prefix}.dyn_body.composite_body.state.rot.Q_parent_this.vector[1]")
        dr.add_variable(f"{prefix}.dyn_body.composite_body.state.rot.Q_parent_this.vector[2]")
    for i in range(3):
        dr.add_variable(f"{prefix}.dyn_body.composite_body.state.rot.ang_vel_this[{i}]")
for v in [
    "rel_state.vehA_wrt_vehB_in_B.rel_state.trans.position[0]",
    "rel_state.vehA_wrt_vehB_in_B.rel_state.trans.position[1]",
    "rel_state.vehA_wrt_vehB_in_B.rel_state.trans.position[2]",
    "rel_state.vehA_wrt_vehB_in_B.rel_state.trans.velocity[0]",
    "rel_state.vehA_wrt_vehB_in_B.rel_state.trans.velocity[1]",
    "rel_state.vehA_wrt_vehB_in_B.rel_state.trans.velocity[2]",
]:
    dr.add_variable(v)
trick.add_data_record_group(dr)
'

# Group 23: SIM_Relative (7 rotation+translation combinations)
run_relative_group() {
    local sim_dir="models/dynamics/derived_state/verif/SIM_Relative"
    local -a RUNS=(
        "SET_test/RUN_AB_rot_AB_trans:relative_ab_rot_ab_trans:relative_ab_rot_ab_trans_relative.csv"
        "SET_test/RUN_AB_rot_no_trans:relative_ab_rot_no_trans:relative_ab_rot_no_trans_relative.csv"
        "SET_test/RUN_A_rot_no_trans:relative_a_rot_no_trans:relative_a_rot_no_trans_relative.csv"
        "SET_test/RUN_B_rot_no_trans:relative_b_rot_no_trans:relative_b_rot_no_trans_relative.csv"
        "SET_test/RUN_no_rot_AB_trans:relative_no_rot_ab_trans:relative_no_rot_ab_trans_relative.csv"
        "SET_test/RUN_no_rot_A_trans:relative_no_rot_a_trans:relative_no_rot_a_trans_relative.csv"
        "SET_test/RUN_no_rot_B_trans:relative_no_rot_b_trans:relative_no_rot_b_trans_relative.csv"
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
        echo "=== Skipping SIM_Relative group (all outputs exist) ==="
        return 0
    fi
    local fail=0
    for entry in "${RUNS[@]}"; do
        IFS=: read -r run_dir label required <<< "$entry"
        run_sim_with_ascii "$sim_dir" "$run_dir" "$label" "$RELATIVE_SNIPPET" "$required" || fail=1
    done
    return $fail
}
throttled_bg run_relative_group
PID_RELATIVE=$LAST_BG_PID

# ── Snippet: SIM_LvlhRelative (LVLH-relative state) ──
LVLH_RELATIVE_SNIPPET='
dr = trick.sim_services.DRAscii("lvlhrel_ASCII")
dr.set_cycle(1)
dr.freq = trick.sim_services.DR_Always
for prefix in ["vehA", "vehB"]:
    for i in range(3):
        dr.add_variable(f"{prefix}.dyn_body.composite_body.state.trans.position[{i}]")
        dr.add_variable(f"{prefix}.dyn_body.composite_body.state.trans.velocity[{i}]")
for v in [
    "rel_state.vehB_in_vehA_rectilvlh.rel_state.trans.position[0]",
    "rel_state.vehB_in_vehA_rectilvlh.rel_state.trans.position[1]",
    "rel_state.vehB_in_vehA_rectilvlh.rel_state.trans.position[2]",
    "rel_state.vehB_in_vehA_rectilvlh.rel_state.trans.velocity[0]",
    "rel_state.vehB_in_vehA_rectilvlh.rel_state.trans.velocity[1]",
    "rel_state.vehB_in_vehA_rectilvlh.rel_state.trans.velocity[2]",
]:
    dr.add_variable(v)
trick.add_data_record_group(dr)
'

# Group 24: SIM_LvlhRelative (2 LVLH proximity runs)
run_lvlh_relative_group() {
    local sim_dir="models/dynamics/derived_state/verif/SIM_LvlhRelative"
    local -a RUNS=(
        "SET_test/RUN_test0:lvlhrel_test0:lvlhrel_test0_lvlhrel.csv"
        "SET_test/RUN_test1:lvlhrel_test1:lvlhrel_test1_lvlhrel.csv"
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
        echo "=== Skipping SIM_LvlhRelative group (all outputs exist) ==="
        return 0
    fi
    local fail=0
    for entry in "${RUNS[@]}"; do
        IFS=: read -r run_dir label required <<< "$entry"
        run_sim_with_ascii "$sim_dir" "$run_dir" "$label" "$LVLH_RELATIVE_SNIPPET" "$required" || fail=1
    done
    return $fail
}
throttled_bg run_lvlh_relative_group
PID_LVLH_RELATIVE=$LAST_BG_PID

# ── Snippet: SIM_LIGHT_CIR (earth lighting circle intersection) ──
LIGHTING_SNIPPET='
dr = trick.sim_services.DRAscii("lighting_ASCII")
dr.set_cycle(1)
dr.freq = trick.sim_services.DR_Always
for v in [
    "light.r_bottom",
    "light.r_top",
    "light.d_centers",
    "light.area",
    "light.lighting.sun_earth.obs_angle",
    "light.lighting.sun_earth.phase",
    "light.lighting.sun_earth.occlusion",
    "light.lighting.sun_earth.visible",
    "light.lighting.sun_earth.lighting",
    "light.lighting.moon_earth.obs_angle",
    "light.lighting.moon_earth.occlusion",
    "light.lighting.moon_earth.visible",
    "light.lighting.moon_earth.lighting",
    "light.lighting.earth_albedo.lighting",
]:
    dr.add_variable(v)
trick.add_data_record_group(dr)
'

# Group 25: SIM_LIGHT_CIR (10 lighting geometry scenarios)
run_lighting_group() {
    local sim_dir="models/environment/earth_lighting/verif/SIM_LIGHT_CIR"
    local -a RUNS=(
        "SET_test/RUN_T01_LIGHT_VER:lighting_t01:lighting_t01_lighting.csv"
        "SET_test/RUN_T02_LIGHT_VER:lighting_t02:lighting_t02_lighting.csv"
        "SET_test/RUN_T03_LIGHT_VER:lighting_t03:lighting_t03_lighting.csv"
        "SET_test/RUN_T04_LIGHT_VER:lighting_t04:lighting_t04_lighting.csv"
        "SET_test/RUN_T05_LIGHT_VER:lighting_t05:lighting_t05_lighting.csv"
        "SET_test/RUN_T06_LIGHT_VER:lighting_t06:lighting_t06_lighting.csv"
        "SET_test/RUN_T07_LIGHT_VER:lighting_t07:lighting_t07_lighting.csv"
        "SET_test/RUN_T08_LIGHT_VER:lighting_t08:lighting_t08_lighting.csv"
        "SET_test/RUN_T09_LIGHT_VER:lighting_t09:lighting_t09_lighting.csv"
        "SET_test/RUN_T10_LIGHT_VER:lighting_t10:lighting_t10_lighting.csv"
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
        echo "=== Skipping SIM_LIGHT_CIR group (all outputs exist) ==="
        return 0
    fi
    local fail=0
    for entry in "${RUNS[@]}"; do
        IFS=: read -r run_dir label required <<< "$entry"
        run_sim_with_ascii "$sim_dir" "$run_dir" "$label" "$LIGHTING_SNIPPET" "$required" || fail=1
    done
    return $fail
}
throttled_bg run_lighting_group
PID_LIGHTING=$LAST_BG_PID

# ── Snippet: SIM_Earth_Moon (vehicle state in Moon-centered orbit) ──
EARTH_MOON_SNIPPET='
# Override common_input.py stop time (3600s) to 7 days for full validation
trick.sim_services.exec_set_terminate_time(604800)
dr = trick.sim_services.DRAscii("earth_moon_ASCII")
dr.set_cycle(60)
dr.freq = trick.sim_services.DR_Always
for i in range(3):
    dr.add_variable(f"vehicle.dyn_body.composite_body.state.trans.position[{i}]")
    dr.add_variable(f"vehicle.dyn_body.composite_body.state.trans.velocity[{i}]")
trick.add_data_record_group(dr)
'

# ── Snippet: SIM_Earth_Moon RUN_rosetta (Earth swing-by, 15000s arc) ──
# RUN_rosetta's input.py sets a 15000s terminate time and is Earth-centric
# (integ frame Earth.inertial), so the same composite_body state log is
# Earth-centered. We do NOT override the terminate time (unlike the clem
# snippet's 7-day override) — the input.py 15000s stands.
ROSETTA_SNIPPET='
dr = trick.sim_services.DRAscii("earth_moon_ASCII")
dr.set_cycle(60)
dr.freq = trick.sim_services.DR_Always
for i in range(3):
    dr.add_variable(f"vehicle.dyn_body.composite_body.state.trans.position[{i}]")
    dr.add_variable(f"vehicle.dyn_body.composite_body.state.trans.velocity[{i}]")
trick.add_data_record_group(dr)
'

# Group 26: SIM_Earth_Moon (Clementine lunar orbit + Rosetta swing-by).
# Both reuse the one SIM executable; run_sim_with_ascii skips per-run when
# the output already exists and builds the exe on the first miss.
run_earth_moon_group() {
    local sim_dir="verif/Integrated_Validation/SIM_Earth_Moon"
    local fail=0
    run_sim_with_ascii "$sim_dir" "SET_test/RUN_clem" "earth_moon_clem" \
        "$EARTH_MOON_SNIPPET" "earth_moon_clem_earth_moon.csv" || fail=1
    run_sim_with_ascii "$sim_dir" "SET_test/RUN_rosetta" "earth_moon_rosetta" \
        "$ROSETTA_SNIPPET" "earth_moon_rosetta_earth_moon.csv" || fail=1
    return $fail
}
throttled_bg run_earth_moon_group
PID_EARTH_MOON=$LAST_BG_PID

# ── Snippet: SIM_Mars (Dawn spacecraft at Mars) ──
MARS_SNIPPET='
dr = trick.sim_services.DRAscii("mars_ASCII")
dr.set_cycle(60)
dr.freq = trick.sim_services.DR_Always
for i in range(3):
    dr.add_variable(f"dawn.dyn_body.composite_body.state.trans.position[{i}]")
    dr.add_variable(f"dawn.dyn_body.composite_body.state.trans.velocity[{i}]")
trick.add_data_record_group(dr)
'

# Group 27: SIM_Mars (Dawn orbit)
run_mars_group() {
    local sim_dir="verif/Integrated_Validation/SIM_Mars"
    local -a RUNS=(
        "SET_test/RUN_dawn:mars_dawn:mars_dawn_mars.csv"
        "SET_test/RUN_phobos:mars_phobos:mars_phobos_mars.csv"
        "SET_test/RUN_orb_init_phobos:mars_orb_init_phobos:mars_orb_init_phobos_mars.csv"
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
        echo "=== Skipping SIM_Mars group (all outputs exist) ==="
        return 0
    fi
    local fail=0
    for entry in "${RUNS[@]}"; do
        IFS=: read -r run_dir label required <<< "$entry"
        run_sim_with_ascii "$sim_dir" "$run_dir" "$label" "$MARS_SNIPPET" "$required" || fail=1
    done
    return $fail
}
throttled_bg run_mars_group
PID_MARS=$LAST_BG_PID

# ── Snippet: SIM_mercury (Mercury propagation) ──
MERCURY_SNIPPET='
dr = trick.sim_services.DRAscii("mercury_ASCII")
dr.set_cycle(3600)
dr.freq = trick.sim_services.DR_Always
for i in range(3):
    dr.add_variable(f"mercury.prop_planet.body.composite_body.state.trans.position[{i}]")
    dr.add_variable(f"mercury.prop_planet.body.composite_body.state.trans.velocity[{i}]")
trick.add_data_record_group(dr)
'

# Group 28: SIM_mercury (Newtonian + relativistic)
run_mercury_group() {
    local sim_dir="models/environment/gravity/verif/SIM_mercury"
    local -a RUNS=(
        "SET_test/RUN_newtonian:mercury_newtonian:mercury_newtonian_mercury.csv"
        "SET_test/RUN_relativistic_sun:mercury_relativistic:mercury_relativistic_mercury.csv"
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
        echo "=== Skipping SIM_mercury group (all outputs exist) ==="
        return 0
    fi
    local fail=0
    for entry in "${RUNS[@]}"; do
        IFS=: read -r run_dir label required <<< "$entry"
        run_sim_with_ascii "$sim_dir" "$run_dir" "$label" "$MERCURY_SNIPPET" "$required" || fail=1
    done
    return $fail
}
throttled_bg run_mercury_group
PID_MERCURY=$LAST_BG_PID

# Group 29: SIM_Apollo — mass-tree attach/detach over a 12-second LEO trajectory.
# Produces both .out files (mass tree printouts via print_tree) and a
# trajectory CSV (DRAscii of the active vehicle cm_dyn). The trajectory is
# non-trivial because cm_dyn's composite mass and CoM jump 11 times as
# stages attach and detach.
APOLLO_SNIPPET='
# JEOD input.py bug fix: set_vehicle_grav_controls is only called for
# les_dyn, but after launch_stack assembly cm_dyn is the integration
# root. cm_dyns grav_interaction.controls list is therefore empty and
# the integrated stack experiences essentially no gravity — the
# trajectory recorded without this fix is unphysical (mostly ballistic).
# Replicate the LES setup sequence on cm_dyn (set_vehicle_grav_controls
# wires earth/moon/sun controls; set_vehicle_sv_at_earth re-applies the
# 8x8 degree/order JEOD evidently intends).
set_vehicle_grav_controls(cm_dyn)
set_vehicle_sv_at_earth(cm_dyn, earth)

# Log core_body rather than composite_body. JEODs detach handler resets
# the integrated-state source to core_body and propagates derived states
# from it; composite_body therefore has discrete inertial jumps at every
# attach/detach event, while core_body is preserved across them. core_body
# is the natural comparison point for an integrated trajectory.
dr = trick.DRAscii("trajectory")
dr.thisown = 0
dr.set_cycle(0.1)
dr.freq = trick.sim_services.DR_Always
for i in range(3):
    dr.add_variable(f"cm_dyn.dyn_body.core_body.state.trans.position[{i}]")
    dr.add_variable(f"cm_dyn.dyn_body.core_body.state.trans.velocity[{i}]")
dr.add_variable("cm_dyn.dyn_body.core_body.state.rot.Q_parent_this.scalar")
for i in range(3):
    dr.add_variable(f"cm_dyn.dyn_body.core_body.state.rot.Q_parent_this.vector[{i}]")
for i in range(3):
    dr.add_variable(f"cm_dyn.dyn_body.core_body.state.rot.ang_vel_this[{i}]")
trick.add_data_record_group(dr)

# Ground-truth recorder for the t=6 attach algorithm investigation
# (see #248). Captures, at 1 ms cadence, every input that flows into
# JEODs DynBody::attach_child momentum-conservation algorithm for cm_dyn
# (parent/integrated body), lm_dyn (child subtree), and s3_dyn
# (intermediate detached subtree root during t=4 to t=5 — needed to
# disambiguate chain-walk-from-S3-to-LM errors from S3 propagation
# errors). The Rust port astrodyn_dynamics::attach::combine_states_at_attach
# is replayed against the cm/lm values; the s3 columns drive the
# tier3_sim_apollo_lm_state_vs_truth diagnostic.
#
# Column layout (cols are 0-indexed; col 0 = time):
#   1..35   cm_dyn  (35 cols: 6 trans + 4 quat + 3 ang_vel + 1 mass + 3 cm + 9 inertia + 9 T)
#   36..70  lm_dyn
#   71..105 s3_dyn  (added with #248 follow-up)
dr2 = trick.DRAscii("attach_truth")
dr2.thisown = 0
dr2.set_cycle(0.001)
dr2.freq = trick.sim_services.DR_Always
for veh in ("cm_dyn", "lm_dyn", "s3_dyn"):
    for i in range(3):
        dr2.add_variable(f"{veh}.dyn_body.composite_body.state.trans.position[{i}]")
        dr2.add_variable(f"{veh}.dyn_body.composite_body.state.trans.velocity[{i}]")
    dr2.add_variable(f"{veh}.dyn_body.composite_body.state.rot.Q_parent_this.scalar")
    for i in range(3):
        dr2.add_variable(f"{veh}.dyn_body.composite_body.state.rot.Q_parent_this.vector[{i}]")
    for i in range(3):
        dr2.add_variable(f"{veh}.dyn_body.composite_body.state.rot.ang_vel_this[{i}]")
    dr2.add_variable(f"{veh}.dyn_body.mass.composite_properties.mass")
    for i in range(3):
        dr2.add_variable(f"{veh}.dyn_body.mass.composite_properties.position[{i}]")
    for i in range(3):
        for j in range(3):
            dr2.add_variable(f"{veh}.dyn_body.mass.composite_properties.inertia[{i}][{j}]")
    for i in range(3):
        for j in range(3):
            dr2.add_variable(f"{veh}.dyn_body.mass.composite_properties.T_parent_this[{i}][{j}]")
trick.add_data_record_group(dr2)
'

run_apollo_group() {
    local sim_dir="sims/SIM_Apollo"
    local label="apollo"
    local run_dir="SET_test/RUN_test"
    local trajectory_csv="${label}_trajectory.csv"

    # All required outputs: trajectory CSV + every .out file the input.py
    # writes. If any is missing we re-run the sim once to regenerate the
    # full set (build is shared, so the cost is one trick-CP at most).
    local out_files=(
        "Initialization.out"
        "Full_Stack.out"
        "1st_Stage_Sep.out"
        "2nd_Stage_Sep.out"
        "LES_Jettison.out"
        "3rd_Stage_Sep.out"
        "LEM_Sep.out"
        "Trans_Lunar.out"
        "Lunar_Orbit.out"
        "LM_Descent.out"
        "Lunar_Rendezvous.out"
        "LM_Ascent.out"
        "Apollo.out"
        "Entry.out"
        "Final.out"
        "Return.out"
    )

    local need_run=0
    if ! has_output "$label" "$trajectory_csv"; then
        need_run=1
    fi
    if [ "$need_run" = "0" ]; then
        for out in "${out_files[@]}"; do
            if [ ! -s "${OUTPUT_DIR}/${label}_${out}" ]; then
                need_run=1
                break
            fi
        done
    fi
    if [ "$need_run" = "0" ] && [ "$FORCE" != "1" ]; then
        echo "=== Skipping SIM_Apollo (all outputs exist) ==="
        return 0
    fi

    # Build, run with ASCII wrapper, and collect the trajectory CSV. Reuses
    # the same wrapper-generation + CSV-canonicalization path as every other
    # ASCII-logged sim.
    _run_sim_with_ascii_impl "$sim_dir" "$run_dir" "$label" "$APOLLO_SNIPPET" || return 1

    # Collect .out files (mass tree printouts) from the RUN directory —
    # SIM_Apollo-specific extra on top of the shared CSV path.
    echo "--- Collecting SIM_Apollo mass tree output ---"
    while IFS= read -r -d '' out_file; do
        local base
        base=$(basename "$out_file")
        local dest="${OUTPUT_DIR}/${label}_${base}"
        cp "$out_file" "$dest"
        echo "  -> ${dest}"
    done < <(find "${run_dir}" -name "*.out" -print0 2>/dev/null)

    # Validate every expected output is present and non-empty before
    # declaring success. Trick's DRAscii silently drops unregistered
    # variables and the sim can return 0 even if the data recorder failed
    # to register, so a passing exit status is not enough — assert each
    # expected file directly.
    local missing=()
    if [ ! -s "${OUTPUT_DIR}/${trajectory_csv}" ]; then
        missing+=("${trajectory_csv}")
    fi
    for out in "${out_files[@]}"; do
        if [ ! -s "${OUTPUT_DIR}/${label}_${out}" ]; then
            missing+=("${label}_${out}")
        fi
    done
    if [ "${#missing[@]}" -gt 0 ]; then
        echo "ERROR: SIM_Apollo run completed but expected outputs are missing or empty:"
        for f in "${missing[@]}"; do
            echo "  - ${OUTPUT_DIR}/${f}"
        done
        return 1
    fi
    echo ""
}
throttled_bg run_apollo_group
PID_APOLLO=$LAST_BG_PID

# Group 30: SIM_verif_attach_mass (mass-tree attach/detach via MassBody)
# Initialization-only sim; each run prints `mass.out` (print_tree output).
# We cover a variety of scenarios: simple attach, chained attach, detach
# at runtime, reattach, and attach_aligned via named mass points.
run_attach_mass_group() {
    local sim_dir="models/dynamics/body_action/verif/SIM_verif_attach_mass"
    local -a RUNS=(
        # Simple MassBody.attach via explicit offset+rotation (Body inertia spec).
        "SET_test/RUN_01:attach_mass_01"
        "SET_test/RUN_02:attach_mass_02"
        "SET_test/RUN_03:attach_mass_03"
        "SET_test/RUN_04:attach_mass_04"
        # Single body / offset attach with Struct / Spec / SpecCG inertia specs.
        "SET_test/RUN_05:attach_mass_05"
        "SET_test/RUN_06:attach_mass_06"
        "SET_test/RUN_07:attach_mass_07"
        # Runtime detach (trick.add_read at t=1s, stop at t=2s).
        "SET_test/RUN_10:attach_mass_10"
        # Runtime reattach (trick.add_read at t=1s, stop at t=2s).
        "SET_test/RUN_11:attach_mass_11"
        # Named-point attach (pt_attach → BodyAttachAligned).
        "SET_test/RUN_101:attach_mass_101"
        "SET_test/RUN_102:attach_mass_102"
        # Named-point chains, Spec/SpecCG via named points.
        "SET_test/RUN_103:attach_mass_103"
        "SET_test/RUN_104:attach_mass_104"
        "SET_test/RUN_106:attach_mass_106"
        "SET_test/RUN_107:attach_mass_107"
        # Named-point attach + runtime detach / reattach.
        "SET_test/RUN_110:attach_mass_110"
        "SET_test/RUN_111:attach_mass_111"
    )

    # Skip entire group if all .out files are already present.
    local needs_build=0
    for entry in "${RUNS[@]}"; do
        IFS=: read -r _run_dir label <<< "$entry"
        if ! has_output "$label" "${label}_mass.out"; then
            needs_build=1
            break
        fi
    done
    if [ "$needs_build" = "0" ]; then
        echo "=== Skipping SIM_verif_attach_mass group (all outputs exist) ==="
        return 0
    fi

    # Build once — all runs share the same S_define.
    echo "=== Building SIM_verif_attach_mass ==="
    cd "${JEOD_HOME}/${sim_dir}" || return 1
    if ! ls S_main*.exe >/dev/null 2>&1; then
        if ! trick-CP 2>&1 | tail -5; then
            echo "ERROR: trick-CP failed for SIM_verif_attach_mass"
            return 1
        fi
    fi

    local exe
    exe=$(ls S_main*.exe 2>/dev/null | head -1)
    if [ -z "$exe" ]; then
        echo "ERROR: No S_main executable found for SIM_verif_attach_mass"
        return 1
    fi

    local fail=0
    for entry in "${RUNS[@]}"; do
        IFS=: read -r run_dir label <<< "$entry"
        local dest="${OUTPUT_DIR}/${label}_mass.out"
        if [ -s "$dest" ] && [ "$FORCE" != "1" ]; then
            echo "--- Skipping ${label} (exists) ---"
            continue
        fi
        local src="${run_dir}/mass.out"
        # Clear any stale mass.out from a prior run so a silent sim failure
        # (or a sim that doesn't regenerate the file) can't cause us to copy
        # outdated data to the reference directory.
        rm -f "$src"
        echo "--- Running ${label} (${run_dir}) ---"
        if ! "./${exe}" "${run_dir}/input.py" 2>&1 | tail -3; then
            echo "ERROR: Sim execution failed for ${label}"
            fail=1
            continue
        fi
        if [ ! -s "$src" ]; then
            echo "ERROR: ${src} was not produced"
            fail=1
            continue
        fi
        cp "$src" "$dest"
        echo "  -> ${dest}"
    done
    return $fail
}
throttled_bg run_attach_mass_group
PID_ATTACH_MASS=$LAST_BG_PID

# Group 31: SIM_verif_attach_detach (dyn-body attach/detach with propagation)
# Inject ASCII logging for per-vehicle composite mass + state. The test
# validates composite_mass evolution over time as attach/detach actions fire.
ATTACH_DETACH_SNIPPET='
dr = trick.sim_services.DRAscii("attach_detach_ASCII")
dr.set_cycle(0.5)
dr.freq = trick.sim_services.DR_Always
for prefix in ["veh1", "veh2", "veh3"]:
    dr.add_variable(f"{prefix}.dyn_body.mass.composite_properties.mass")
trick.add_data_record_group(dr)
'

run_attach_detach_group() {
    local sim_dir="models/dynamics/dyn_body/verif/SIM_verif_attach_detach"
    local -a RUNS=(
        "SET_test/RUN_simple_attach_detach:attach_detach_simple:attach_detach_simple_attach_detach.csv"
        "SET_test/RUN_complex_attach_detach:attach_detach_complex:attach_detach_complex_attach_detach.csv"
        "SET_test/RUN_compute_child_derivative:attach_detach_child_deriv:attach_detach_child_deriv_attach_detach.csv"
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
        echo "=== Skipping SIM_verif_attach_detach group (all outputs exist) ==="
        return 0
    fi
    local fail=0
    for entry in "${RUNS[@]}"; do
        IFS=: read -r run_dir label required <<< "$entry"
        run_sim_with_ascii "$sim_dir" "$run_dir" "$label" "$ATTACH_DETACH_SNIPPET" "$required" || fail=1
    done
    return $fail
}
throttled_bg run_attach_detach_group
PID_ATTACH_DETACH=$LAST_BG_PID

# Group 31b: SIM_verif_attach_detach kinematic-propagation logging
# Same SIM_verif_attach_detach binary as group 31, but a richer ASCII
# snippet that logs composite-body state (translational + rotational)
# for veh1 + veh2 + veh3, used by the runner-side
# tier3_sim_kinematic_propagation test (issue #294). Run as a separate
# DR group so the existing composite-mass CSV under
# attach_detach_*_attach_detach.csv stays bit-identical to its older
# regen output.
KINEMATIC_PROP_SNIPPET='
dr = trick.sim_services.DRAscii("kinematic_propagation_state")
dr.thisown = 0
dr.set_cycle(0.5)
dr.freq = trick.sim_services.DR_Always
for prefix in ["veh1", "veh2", "veh3"]:
    for i in range(3):
        dr.add_variable(f"{prefix}.dyn_body.composite_body.state.trans.position[{i}]")
    for i in range(3):
        dr.add_variable(f"{prefix}.dyn_body.composite_body.state.trans.velocity[{i}]")
    dr.add_variable(f"{prefix}.dyn_body.composite_body.state.rot.Q_parent_this.scalar")
    for i in range(3):
        dr.add_variable(f"{prefix}.dyn_body.composite_body.state.rot.Q_parent_this.vector[{i}]")
    for i in range(3):
        dr.add_variable(f"{prefix}.dyn_body.composite_body.state.rot.ang_vel_this[{i}]")
trick.add_data_record_group(dr)
'

run_kinematic_propagation_group() {
    local sim_dir="models/dynamics/dyn_body/verif/SIM_verif_attach_detach"
    local -a RUNS=(
        "SET_test/RUN_simple_attach_detach:kinematic_propagation_simple:kinematic_propagation_simple_kinematic_propagation_state.csv"
        "SET_test/RUN_complex_attach_detach:chained_attach_complex:chained_attach_complex_kinematic_propagation_state.csv"
        "SET_test/RUN_compute_child_derivative:chained_attach_child_deriv:chained_attach_child_deriv_kinematic_propagation_state.csv"
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
        echo "=== Skipping SIM_verif_attach_detach kinematic-propagation group (all outputs exist) ==="
        return 0
    fi
    local fail=0
    for entry in "${RUNS[@]}"; do
        IFS=: read -r run_dir label required <<< "$entry"
        run_sim_with_ascii "$sim_dir" "$run_dir" "$label" "$KINEMATIC_PROP_SNIPPET" "$required" || fail=1
    done
    return $fail
}
throttled_bg run_kinematic_propagation_group
PID_KINEMATIC_PROP=$LAST_BG_PID

# Group 32: SIM_verif_frame_switch (Apollo 8 frame switching)
# Inject ASCII logging for 6-DOF state (translational + rotational).
APOLLO8_SNIPPET='
dr = trick.DRAscii("sixdof_state")
dr.thisown = 0
dr.set_cycle(0.5)
dr.freq = trick.sim_services.DR_Always
for i in range(3):
    dr.add_variable(f"veh.dyn_body.composite_body.state.trans.position[{i}]")
    dr.add_variable(f"veh.dyn_body.composite_body.state.trans.velocity[{i}]")
dr.add_variable("veh.dyn_body.composite_body.state.rot.Q_parent_this.scalar")
for i in range(3):
    dr.add_variable(f"veh.dyn_body.composite_body.state.rot.Q_parent_this.vector[{i}]")
for i in range(3):
    dr.add_variable(f"veh.dyn_body.composite_body.state.rot.ang_vel_this[{i}]")
trick.add_data_record_group(dr)
'

run_frame_switch_group() {
    local sim_dir="models/dynamics/body_action/verif/SIM_verif_frame_switch"
    local -a RUNS=(
        "SET_test/RUN_Apollo_08_ECI_integ:apollo8_eci:apollo8_eci_sixdof_state.csv"
        "SET_test/RUN_Apollo_08_frame_switch:apollo8_frame_switch:apollo8_frame_switch_sixdof_state.csv"
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
        echo "=== Skipping SIM_verif_frame_switch group (all outputs exist) ==="
        return 0
    fi

    local fail=0
    for entry in "${RUNS[@]}"; do
        IFS=: read -r run_dir label required <<< "$entry"
        run_sim_with_ascii "$sim_dir" "$run_dir" "$label" "$APOLLO8_SNIPPET" "$required" || fail=1
    done
    return $fail
}
throttled_bg run_frame_switch_group
PID_FRAME_SWITCH=$LAST_BG_PID

# Group 33: SIM_ref_attach — JEOD's reference-frame attachment verification
# (issues #198 / #206). The two RUNs each attach the target vehicle to a
# parent ref frame at t=50s and run to t=100s; the body's state stops
# integrating and is derived each tick from the parent frame.
# RUN_ref_attach_matrix uses BodyAttachMatrix (offset + rotation), parent =
# Earth.pfix.
# RUN_ref_attach_pt2pt uses BodyAttachAligned (mass-point to mass-point
# alignment), parent = Earth.inertial.
# We log composite-body state so the runner-side Tier 3 test
# (`tier3_sim_ref_attach.rs`) can cross-validate trajectory + post-attach
# rigid attachment.
REF_ATTACH_SNIPPET='
dr = trick.DRAscii("ref_attach_state")
dr.thisown = 0
dr.set_cycle(0.5)
dr.freq = trick.sim_services.DR_Always
for i in range(3):
    dr.add_variable(f"target.dyn_body.composite_body.state.trans.position[{i}]")
for i in range(3):
    dr.add_variable(f"target.dyn_body.composite_body.state.trans.velocity[{i}]")
dr.add_variable("target.dyn_body.composite_body.state.rot.Q_parent_this.scalar")
for i in range(3):
    dr.add_variable(f"target.dyn_body.composite_body.state.rot.Q_parent_this.vector[{i}]")
for i in range(3):
    dr.add_variable(f"target.dyn_body.composite_body.state.rot.ang_vel_this[{i}]")
trick.add_data_record_group(dr)
'

run_ref_attach_group() {
    local sim_dir="models/dynamics/body_action/verif/SIM_ref_attach"
    local -a RUNS=(
        "SET_test/RUN_ref_attach_matrix:ref_attach_matrix:ref_attach_matrix_ref_attach_state.csv"
        "SET_test/RUN_ref_attach_pt2pt:ref_attach_pt2pt:ref_attach_pt2pt_ref_attach_state.csv"
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
        echo "=== Skipping SIM_ref_attach group (all outputs exist) ==="
        return 0
    fi

    local fail=0
    for entry in "${RUNS[@]}"; do
        IFS=: read -r run_dir label required <<< "$entry"
        run_sim_with_ascii "$sim_dir" "$run_dir" "$label" "$REF_ATTACH_SNIPPET" "$required" || fail=1
    done
    return $fail
}
throttled_bg run_ref_attach_group
PID_REF_ATTACH=$LAST_BG_PID

# Group 34: SIM_contact — free-space contact dynamics (5 scenarios)
# ASCII snippet logs: time (implicit), veh{1,2} position/velocity,
# contact force/torque on each vehicle, composite masses.
# Matches JEOD Log_data/log_contact_data.py variables.
CONTACT_SNIPPET='
dr = trick.DRAscii("contact_state")
dr.thisown = 0
dr.set_cycle(0.05)
dr.freq = trick.sim_services.DR_Always
for i in range(3):
    dr.add_variable(f"veh1_dyn.body.composite_body.state.trans.position[{i}]")
for i in range(3):
    dr.add_variable(f"veh1_dyn.body.composite_body.state.trans.velocity[{i}]")
for i in range(3):
    dr.add_variable(f"veh1_dyn.contact_surface.contact_force[{i}]")
for i in range(3):
    dr.add_variable(f"veh1_dyn.contact_surface.contact_torque[{i}]")
for i in range(3):
    dr.add_variable(f"veh2_dyn.body.composite_body.state.trans.position[{i}]")
for i in range(3):
    dr.add_variable(f"veh2_dyn.body.composite_body.state.trans.velocity[{i}]")
for i in range(3):
    dr.add_variable(f"veh2_dyn.contact_surface.contact_force[{i}]")
for i in range(3):
    dr.add_variable(f"veh2_dyn.contact_surface.contact_torque[{i}]")
dr.add_variable("veh1_dyn.body.mass.composite_properties.mass")
dr.add_variable("veh2_dyn.body.mass.composite_properties.mass")
trick.add_data_record_group(dr)
'

run_contact_group() {
    local sim_dir="models/interactions/contact/verif/SIM_contact"
    local -a RUNS=(
        "SET_test/RUN_point:contact_point:contact_point_contact_state.csv"
        "SET_test/RUN_line:contact_line:contact_line_contact_state.csv"
        "SET_test/RUN_line_point:contact_line_point:contact_line_point_contact_state.csv"
        "SET_test/RUN_line_side_to_side:contact_line_side:contact_line_side_contact_state.csv"
        "SET_test/RUN_point_off_center:contact_point_off_center:contact_point_off_center_contact_state.csv"
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
        echo "=== Skipping SIM_contact group (all outputs exist) ==="
        return 0
    fi

    local fail=0
    for entry in "${RUNS[@]}"; do
        IFS=: read -r run_dir label required <<< "$entry"
        run_sim_with_ascii "$sim_dir" "$run_dir" "$label" "$CONTACT_SNIPPET" "$required" || fail=1
        # JEOD declares `contact_torque` as trick_units(N/m) in
        # contact_surface.hh — a typo for the correct N*m. Patch the CSV
        # header so our reference data has physically correct units.
        local out_file="$OUTPUT_DIR/$required"
        if [ -f "$out_file" ]; then
            # Portable in-place edit (avoid GNU-only `sed -i`): rewrite to
            # temp then replace atomically. Works under GNU and BSD sed.
            sed '1s/contact_torque\(\[[0-2]\]\) {N\/m}/contact_torque\1 {N*m}/g' \
                "$out_file" > "$out_file.tmp" && mv "$out_file.tmp" "$out_file"
        fi
    done
    return $fail
}
throttled_bg run_contact_group
PID_CONTACT=$LAST_BG_PID

# Group 35: SIM_ground_contact — Earth-frame ground contact (1 scenario)
# Shares the CONTACT_SNIPPET log variables; adds Earth central body.
run_ground_contact_group() {
    local sim_dir="models/interactions/contact/verif/SIM_ground_contact"
    local -a RUNS=(
        "SET_test/RUN_contact_ground:contact_ground:contact_ground_contact_state.csv"
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
        echo "=== Skipping SIM_ground_contact group (all outputs exist) ==="
        return 0
    fi

    local fail=0
    for entry in "${RUNS[@]}"; do
        IFS=: read -r run_dir label required <<< "$entry"
        run_sim_with_ascii "$sim_dir" "$run_dir" "$label" "$CONTACT_SNIPPET" "$required" || fail=1
        # See note in run_contact_group: JEOD's contact_surface.hh mislabels
        # contact_torque units as N/m; correct to N*m in the CSV header.
        local out_file="$OUTPUT_DIR/$required"
        if [ -f "$out_file" ]; then
            # Portable in-place edit (avoid GNU-only `sed -i`): rewrite to
            # temp then replace atomically. Works under GNU and BSD sed.
            sed '1s/contact_torque\(\[[0-2]\]\) {N\/m}/contact_torque\1 {N*m}/g' \
                "$out_file" > "$out_file.tmp" && mv "$out_file.tmp" "$out_file"
        fi
    done
    return $fail
}
throttled_bg run_ground_contact_group
PID_GROUND_CONTACT=$LAST_BG_PID

# ════════════════════════════════════════════════════════════════════
# JEOD time verification SIMs (1-6) for Tier 3 time cross-validation.
# Consumed by crates/astrodyn_runner/tests/tier3_sim_time_docker.rs.
#
# Object-path convention differs between sims:
#   SIM_1-4  use `jeod_time.time_manager` / `time_tai` / `time_utc` / ...
#   SIM_5-6  use `jeod_time.manager` / `tai` / `utc` / ...
# This mirrors the S_define declarations for each sim exactly — any
# mismatch causes Trick's DRAscii to silently omit the variable.
# ════════════════════════════════════════════════════════════════════

# ── Snippet: SIM_1_dyn_only RUN_dyn (DynamicTime only) ──
TIME_V1_SNIPPET='
dr = trick.sim_services.DRAscii("time_v1")
dr.set_cycle(1)
dr.freq = trick.sim_services.DR_Always
for v in [
    "jeod_time.time_manager.dyn_time.seconds",
]:
    dr.add_variable(v)
trick.add_data_record_group(dr)
'

run_time_v1_group() {
    local sim_dir="models/environment/time/verif/SIM_1_dyn_only"
    local -a RUNS=(
        "SET_test/RUN_dyn:time_v1_dyn_only:time_v1_dyn_only_time_v1.csv"
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
        echo "=== Skipping SIM_1_dyn_only group (all outputs exist) ==="
        return 0
    fi
    local fail=0
    for entry in "${RUNS[@]}"; do
        IFS=: read -r run_dir label required <<< "$entry"
        run_sim_with_ascii "$sim_dir" "$run_dir" "$label" "$TIME_V1_SNIPPET" "$required" || fail=1
    done
    return $fail
}
throttled_bg run_time_v1_group
PID_TIME_V1=$LAST_BG_PID

# ── Snippet: SIM_2_dyn_plus_STD RUN_initialize_by_value (TAI + Dyn) ──
TIME_V2_SNIPPET='
dr = trick.sim_services.DRAscii("time_v2")
dr.set_cycle(1)
dr.freq = trick.sim_services.DR_Always
for v in [
    "jeod_time.time_manager.dyn_time.seconds",
    "jeod_time.time_tai.trunc_julian_time",
    "jeod_time.time_tai.seconds",
]:
    dr.add_variable(v)
trick.add_data_record_group(dr)
'

run_time_v2_group() {
    local sim_dir="models/environment/time/verif/SIM_2_dyn_plus_STD"
    local -a RUNS=(
        "SET_test/RUN_initialize_by_value:time_v2_std:time_v2_std_time_v2.csv"
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
        echo "=== Skipping SIM_2_dyn_plus_STD group (all outputs exist) ==="
        return 0
    fi
    local fail=0
    for entry in "${RUNS[@]}"; do
        IFS=: read -r run_dir label required <<< "$entry"
        run_sim_with_ascii "$sim_dir" "$run_dir" "$label" "$TIME_V2_SNIPPET" "$required" || fail=1
    done
    return $fail
}
throttled_bg run_time_v2_group
PID_TIME_V2=$LAST_BG_PID

# ── Snippet: SIM_3_dyn_plus_UDE RUN_init_by_ude (UDE + Dyn) ──
TIME_V3_SNIPPET='
dr = trick.sim_services.DRAscii("time_v3")
dr.set_cycle(1)
dr.freq = trick.sim_services.DR_Always
for v in [
    "jeod_time.time_manager.dyn_time.seconds",
    "jeod_time.time_ude.seconds",
]:
    dr.add_variable(v)
trick.add_data_record_group(dr)
'

run_time_v3_group() {
    local sim_dir="models/environment/time/verif/SIM_3_dyn_plus_UDE"
    local -a RUNS=(
        "SET_test/RUN_init_by_ude:time_v3_ude:time_v3_ude_time_v3.csv"
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
        echo "=== Skipping SIM_3_dyn_plus_UDE group (all outputs exist) ==="
        return 0
    fi
    local fail=0
    for entry in "${RUNS[@]}"; do
        IFS=: read -r run_dir label required <<< "$entry"
        run_sim_with_ascii "$sim_dir" "$run_dir" "$label" "$TIME_V3_SNIPPET" "$required" || fail=1
    done
    return $fail
}
throttled_bg run_time_v3_group
PID_TIME_V3=$LAST_BG_PID

# ── Snippet: SIM_4_common_usage (TAI + UTC + UT1 across leap sec) ──
# Two RUNs, identical except for the UTC/UT1 convention:
#   RUN_JEOD2x            — true_utc / true_ut1 (default): UTC/UT1 track the
#                           leap-second table, so UTC TJT jumps at 1999-01-01.
#   RUN_JEOD1x_compatible — true_utc=False / true_ut1=False: the TAI−UTC and
#                           UT1−TAI offsets are frozen at the epoch value, so
#                           UTC/UT1 TJT do NOT jump across the boundary.
# Log every 60 s to keep the CSV small; run spans 86460 s, crossing the
# 1999-01-01 leap second boundary.
TIME_V4_SNIPPET='
dr = trick.sim_services.DRAscii("time_v4")
dr.set_cycle(60)
dr.freq = trick.sim_services.DR_Always
for v in [
    "jeod_time.time_manager.dyn_time.seconds",
    "jeod_time.time_tai.trunc_julian_time",
    "jeod_time.time_tai.seconds",
    "jeod_time.time_utc.trunc_julian_time",
    "jeod_time.time_utc.seconds",
    "jeod_time.time_ut1.trunc_julian_time",
    "jeod_time.time_ut1.seconds",
]:
    dr.add_variable(v)
trick.add_data_record_group(dr)
'

run_time_v4_group() {
    local sim_dir="models/environment/time/verif/SIM_4_common_usage"
    local -a RUNS=(
        "SET_test/RUN_JEOD2x:time_v4_common:time_v4_common_time_v4.csv"
        "SET_test/RUN_JEOD1x_compatible:time_v4_jeod1x:time_v4_jeod1x_time_v4.csv"
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
        echo "=== Skipping SIM_4_common_usage group (all outputs exist) ==="
        return 0
    fi
    local fail=0
    for entry in "${RUNS[@]}"; do
        IFS=: read -r run_dir label required <<< "$entry"
        run_sim_with_ascii "$sim_dir" "$run_dir" "$label" "$TIME_V4_SNIPPET" "$required" || fail=1
    done
    return $fail
}
throttled_bg run_time_v4_group
PID_TIME_V4=$LAST_BG_PID

# ── Snippet: SIM_5_all_inclusive RUN_UDE_initialized (all 10+ scales + MET) ──
# SIM_5 uses bare `manager`/`tai`/... (no `time_` prefix) per its S_define.
TIME_V5_SNIPPET='
dr = trick.sim_services.DRAscii("time_v5")
dr.set_cycle(1)
dr.freq = trick.sim_services.DR_Always
for v in [
    "jeod_time.manager.dyn_time.seconds",
    "jeod_time.tai.trunc_julian_time",
    "jeod_time.tai.seconds",
    "jeod_time.utc.trunc_julian_time",
    "jeod_time.utc.seconds",
    "jeod_time.ut1.trunc_julian_time",
    "jeod_time.ut1.seconds",
    "jeod_time.tt.trunc_julian_time",
    "jeod_time.tt.seconds",
    "jeod_time.tdb.trunc_julian_time",
    "jeod_time.tdb.seconds",
    "jeod_time.gmst.seconds",
    "jeod_time.gps.trunc_julian_time",
    "jeod_time.gps.seconds",
    "jeod_time.metveh1.seconds",
    "jeod_time.metveh2.seconds",
]:
    dr.add_variable(v)
trick.add_data_record_group(dr)
'

run_time_v5_group() {
    local sim_dir="models/environment/time/verif/SIM_5_all_inclusive"
    local -a RUNS=(
        "SET_test/RUN_UDE_initialized:time_v5_all:time_v5_all_time_v5.csv"
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
        echo "=== Skipping SIM_5_all_inclusive (UDE) group (all outputs exist) ==="
        return 0
    fi
    local fail=0
    for entry in "${RUNS[@]}"; do
        IFS=: read -r run_dir label required <<< "$entry"
        run_sim_with_ascii "$sim_dir" "$run_dir" "$label" "$TIME_V5_SNIPPET" "$required" || fail=1
    done
    return $fail
}
throttled_bg run_time_v5_group
PID_TIME_V5=$LAST_BG_PID

# ── Snippet: SIM_6_extension RUN_tai_initialized (TAI only; we skip the
# user-defined "new" scale registered by the sim). Bare object paths. ──
TIME_V6_SNIPPET='
dr = trick.sim_services.DRAscii("time_v6")
dr.set_cycle(1)
dr.freq = trick.sim_services.DR_Always
for v in [
    "jeod_time.manager.dyn_time.seconds",
    "jeod_time.tai.trunc_julian_time",
    "jeod_time.tai.seconds",
]:
    dr.add_variable(v)
trick.add_data_record_group(dr)
'

run_time_v6_group() {
    local sim_dir="models/environment/time/verif/SIM_6_extension"
    local -a RUNS=(
        "SET_test/RUN_tai_initialized:time_v6_ext:time_v6_ext_time_v6.csv"
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
        echo "=== Skipping SIM_6_extension group (all outputs exist) ==="
        return 0
    fi
    local fail=0
    for entry in "${RUNS[@]}"; do
        IFS=: read -r run_dir label required <<< "$entry"
        run_sim_with_ascii "$sim_dir" "$run_dir" "$label" "$TIME_V6_SNIPPET" "$required" || fail=1
    done
    return $fail
}
throttled_bg run_time_v6_group
PID_TIME_V6=$LAST_BG_PID

# ════════════════════════════════════════════════════════════════════
# WAIT FOR ALL GROUPS
# ════════════════════════════════════════════════════════════════════
echo "=== Waiting for all sim groups to complete ==="
FAIL=0

wait $PID_DYNCOMP       || { echo "WARN: SIM_dyncomp group had failures"; FAIL=1; }
wait $PID_CSR_COMPARE   || { echo "WARN: SIM_csr_compare group had failures"; FAIL=1; }
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
wait $PID_INTEG_GJ      || { echo "WARN: SIM_GJ_test failed"; FAIL=1; }
# Phase 6 additions
wait $PID_ORBELEM_VERIF  || { echo "WARN: SIM_orb_elem verif group had failures"; FAIL=1; }
wait $PID_PLANETARY      || { echo "WARN: SIM_Planetary group had failures"; FAIL=1; }
wait $PID_MET_VERIF      || { echo "WARN: SIM_MET verif group had failures"; FAIL=1; }
wait $PID_TIMESCALE      || { echo "WARN: SIM_5_all_inclusive group had failures"; FAIL=1; }
wait $PID_TIME_REVERSAL  || { echo "WARN: SIM_7_time_reversal group had failures"; FAIL=1; }
wait $PID_RNP_VERIF      || { echo "WARN: SIM_RNP_J2000_prop group had failures"; FAIL=1; }
wait $PID_RELATIVE       || { echo "WARN: SIM_Relative group had failures"; FAIL=1; }
wait $PID_LVLH_RELATIVE  || { echo "WARN: SIM_LvlhRelative group had failures"; FAIL=1; }
wait $PID_LIGHTING       || { echo "WARN: SIM_LIGHT_CIR group had failures"; FAIL=1; }
wait $PID_EARTH_MOON     || { echo "WARN: SIM_Earth_Moon group had failures"; FAIL=1; }
wait $PID_MARS           || { echo "WARN: SIM_Mars group had failures"; FAIL=1; }
wait $PID_MERCURY        || { echo "WARN: SIM_mercury group had failures"; FAIL=1; }
wait $PID_APOLLO         || { echo "WARN: SIM_Apollo group had failures"; FAIL=1; }
wait $PID_ATTACH_MASS    || { echo "WARN: SIM_verif_attach_mass group had failures"; FAIL=1; }
wait $PID_ATTACH_DETACH  || { echo "WARN: SIM_verif_attach_detach group had failures"; FAIL=1; }
wait $PID_KINEMATIC_PROP || { echo "WARN: SIM_verif_attach_detach kinematic-propagation group had failures"; FAIL=1; }
wait $PID_FRAME_SWITCH   || { echo "WARN: SIM_verif_frame_switch group had failures"; FAIL=1; }
wait $PID_REF_ATTACH     || { echo "WARN: SIM_ref_attach group had failures"; FAIL=1; }
wait $PID_CONTACT        || { echo "WARN: SIM_contact group had failures"; FAIL=1; }
wait $PID_GROUND_CONTACT || { echo "WARN: SIM_ground_contact group had failures"; FAIL=1; }
# WS-R4: JEOD time verification SIMs 1-6
wait $PID_TIME_V1        || { echo "WARN: SIM_1_dyn_only group had failures"; FAIL=1; }
wait $PID_TIME_V2        || { echo "WARN: SIM_2_dyn_plus_STD group had failures"; FAIL=1; }
wait $PID_TIME_V3        || { echo "WARN: SIM_3_dyn_plus_UDE group had failures"; FAIL=1; }
wait $PID_TIME_V4        || { echo "WARN: SIM_4_common_usage group had failures"; FAIL=1; }
wait $PID_TIME_V5        || { echo "WARN: SIM_5_all_inclusive UDE group had failures"; FAIL=1; }
wait $PID_TIME_V6        || { echo "WARN: SIM_6_extension group had failures"; FAIL=1; }

echo ""
echo "=== Reference data generation complete ==="
echo "Files in ${OUTPUT_DIR}:"
ls -la "${OUTPUT_DIR}/"

exit $FAIL
