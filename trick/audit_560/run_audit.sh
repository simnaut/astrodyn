#!/usr/bin/env bash
# JEOD-side comprehensive instrumentation patches for issue #560
# (https://github.com/simnaut/astrodyn/issues/560).
#
# Diagnostic-only. This script is not part of the regular Tier 3
# regeneration flow; it is invoked by hand when a numerical-parity
# investigation needs the per-stage, per-op dump stream that the
# original audit chain (8 hypotheses, all surfacing the same
# rounding-path conclusion) used.
#
# The patches inject `printf("[#560/FULL] step=%lu stage=%d body=%d
# op=<name> kI=vI ...\n", ...)` calls into the following JEOD / Trick
# sources so the resulting stderr stream aligns line-by-line with the
# Rust-side `astrodyn_quantities::audit_560::dump_*` emitters:
#
#   - models/interactions/contact/src/point_contact_pair.cc
#       in_contact(): rel_pos, geom_normal, geom_penetration_depth,
#       rel_vel, contact_arm_a_inertial, contact_arm_b_inertial.
#
#   - models/interactions/contact/src/spring_pair_interaction.cc
#       calculate_forces(): force_penetration_vec, force_spring,
#       force_v_normal_mag, force_damping, force_friction, force_total.
#
#   - models/interactions/contact/src/point_contact_facet.cc
#       calculate_torque(): torque_a_body, torque_b_body.
#
#   - models/dynamics/dyn_body/src/dyn_body_integration.cc
#       integrate(): per-stage body state at entry/exit (eval_stage_*,
#       composed_* names; mirrors our `eval_stage`).
#
#   - trick_source/er7_utils/integration/rk4/src/rk4_second_order_ode_integrator.cc
#       rk_two_state_intermediate_step(): per-stage k_v / k_a / k_qdot /
#       k_alpha derivative outputs (mirrors our eval_stage trailing
#       dumps).
#
# Usage:
#   $JEOD_HOME and $TRICK_HOME must point at writable checkouts (the
#   patches are applied in place — git-restore them after the audit).
#   Then run from this directory:
#
#     ./run_audit.sh                                 # apply patches + rebuild SIM_contact
#     ./run_audit.sh --revert                        # revert in-place patches
#     ./run_audit.sh --run RUN_point_off_center      # run the contact sim and capture stderr
#
# The dump stream is captured to ./jeod_dump.txt for
# `diff_streams.py` consumption. The Rust-side companion stream is
# captured with:
#
#   ASTRODYN_560_FULL_DUMP=1 cargo nextest run \
#     -p astrodyn_verif_jeod --test tier3_sim_contact \
#     -E 'test(ablation)' --run-ignored ignored-only \
#     2> rust_dump.txt
#
# and then:
#
#   python3 diff_streams.py --jeod jeod_dump.txt --rust rust_dump.txt
#
# emits the first-divergent op + the ranked table reproduced in the
# audit summary on issue #560.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_audit.sh [--apply | --revert | --run RUN_NAME]

  --apply           Currently NOT IMPLEMENTED — exits non-zero with a
                    pointer to the canonical patch series. Apply the
                    captured patch files by hand (see below). This is
                    also the no-args default.
  --revert          Revert patched files via `git checkout --` in
                    `$JEOD_HOME` / `$TRICK_HOME` for every file in the
                    audit's source list.
  --run RUN_NAME    Run SIM_contact for the named RUN (e.g. RUN_point_off_center)
                    and capture stderr to ./jeod_dump.txt. Requires
                    that the patches have already been applied by hand
                    and the sim rebuilt.

Environment:
  JEOD_HOME    path to a writable JEOD checkout (required)
  TRICK_HOME   path to a writable Trick checkout (required)

Output:
  ./jeod_dump.txt   captured `[#560/FULL] …` lines from the sim run.

This script is diagnostic-only and not part of the Tier 3
regeneration flow. The patches modify source files in place — always
`--revert` (or `git checkout --` the touched files) before resuming
normal use of the JEOD / Trick checkouts.

Applying the canonical patch series (manual workflow):

  Issue #560's "Comprehensive bidirectional instrumentation deployed"
  comment carries the exact patch hunks for both source trees. Save
  them as `audit_560_jeod.patch` / `audit_560_trick.patch`, then:

    ( cd "$JEOD_HOME"  && git apply /path/to/audit_560_jeod.patch  )
    ( cd "$TRICK_HOME" && git apply /path/to/audit_560_trick.patch )

  followed by a sim rebuild (`trick-CP` from `$JEOD_HOME/verif/SIM_contact`)
  before `--run RUN_NAME` will capture a populated stream.
EOF
}

require_env() {
  local var=$1
  if [[ -z "${!var:-}" ]]; then
    echo "error: \$${var} must be set (writable source checkout)" >&2
    exit 2
  fi
  if [[ ! -d "${!var}" ]]; then
    echo "error: \$${var} (= ${!var}) is not a directory" >&2
    exit 2
  fi
}

JEOD_FILES=(
  "models/interactions/contact/src/point_contact_pair.cc"
  "models/interactions/contact/src/spring_pair_interaction.cc"
  "models/interactions/contact/src/point_contact_facet.cc"
  "models/dynamics/dyn_body/src/dyn_body_integration.cc"
)
TRICK_FILES=(
  "trick_source/er7_utils/integration/rk4/src/rk4_second_order_ode_integrator.cc"
)

apply_patches() {
  require_env JEOD_HOME
  require_env TRICK_HOME

  # `--apply` is intentionally not implemented as an in-tree sed/patch
  # operation. The patch hunks reference JEOD / Trick source line
  # numbers that vary by release, and the audit chain on issue #560
  # carries the canonical patch series (8 hypotheses' worth) in the
  # GitHub comment thread rather than as committed `.patch` files
  # here. Auto-applying a release-specific hunk via this script would
  # silently no-op on a version mismatch — and a silent no-op is
  # exactly the failure mode Copilot flagged. Surface the gap loudly
  # instead and point the operator at the canonical recipe.

  cat >&2 <<EOF
error: --apply is not implemented in this script.

The canonical patch series for the issue #560 instrumentation lives in
the GitHub issue thread (see the "Comprehensive bidirectional
instrumentation deployed" comment), not in this repository — the hunks
are JEOD / Trick release-specific and would silently no-op if applied
blind from this script.

To apply by hand:

  1. Download audit_560_jeod.patch / audit_560_trick.patch from issue #560.
  2. ( cd "\$JEOD_HOME"  && git apply /path/to/audit_560_jeod.patch  )
  3. ( cd "\$TRICK_HOME" && git apply /path/to/audit_560_trick.patch )
  4. Rebuild SIM_contact (\`trick-CP\` from \$JEOD_HOME/verif/SIM_contact).
  5. Re-run this script with --run RUN_NAME to capture the dump stream.

Reverting (\`run_audit.sh --revert\`) still works after a manual apply.
EOF

  # Sanity check the file list against the working copies so a real
  # version skew shows up alongside the message above (a missing file
  # invalidates the whole patch workflow even if the operator has the
  # patch files in hand).
  for f in "${JEOD_FILES[@]}"; do
    if [[ ! -f "$JEOD_HOME/$f" ]]; then
      echo "warning: $JEOD_HOME/$f not found (JEOD version skew?)" >&2
    fi
  done
  for f in "${TRICK_FILES[@]}"; do
    if [[ ! -f "$TRICK_HOME/$f" ]]; then
      echo "warning: $TRICK_HOME/$f not found (Trick version skew?)" >&2
    fi
  done

  return 3
}

revert_patches() {
  require_env JEOD_HOME
  require_env TRICK_HOME

  for f in "${JEOD_FILES[@]}"; do
    if [[ -f "$JEOD_HOME/$f" ]]; then
      ( cd "$JEOD_HOME" && git checkout -- "$f" ) || true
    fi
  done
  for f in "${TRICK_FILES[@]}"; do
    if [[ -f "$TRICK_HOME/$f" ]]; then
      ( cd "$TRICK_HOME" && git checkout -- "$f" ) || true
    fi
  done
  echo "audit_560: reverted patches via 'git checkout --'" >&2
}

run_sim() {
  require_env JEOD_HOME
  require_env TRICK_HOME
  local run="$1"
  local sim_root="$JEOD_HOME/verif/SIM_contact"
  if [[ ! -d "$sim_root" ]]; then
    echo "error: $sim_root not found — JEOD checkout missing SIM_contact?" >&2
    exit 3
  fi
  if [[ ! -d "$sim_root/SET_test/$run" ]]; then
    echo "error: $sim_root/SET_test/$run not found" >&2
    exit 3
  fi
  pushd "$sim_root" >/dev/null
  # Build (no-op if already current).
  trick-CP
  # Run with the audit stream redirected to stderr → ./jeod_dump.txt.
  ./S_main_*.exe "SET_test/$run/input.py" 2> "$OLDPWD/jeod_dump.txt"
  popd >/dev/null
  echo "audit_560: stderr captured to ./jeod_dump.txt" >&2
}

main() {
  if [[ $# -eq 0 ]]; then
    # No-args matches `--apply`. `apply_patches` returns 3 to surface
    # the unimplemented-patch-application failure mode loudly (per
    # Copilot review on PR #590). `|| rc=$?` guards against `set -e`
    # killing us before we can propagate that exit code.
    local rc=0
    apply_patches || rc=$?
    return "$rc"
  fi
  case "$1" in
    --apply)
      local rc=0
      apply_patches || rc=$?
      return "$rc"
      ;;
    --revert)
      revert_patches
      ;;
    --run)
      if [[ $# -lt 2 ]]; then
        echo "error: --run needs a RUN_NAME argument" >&2
        usage
        exit 2
      fi
      run_sim "$2"
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "error: unknown flag $1" >&2
      usage
      exit 2
      ;;
  esac
}

main "$@"
