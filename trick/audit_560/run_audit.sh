#!/bin/bash
# #560 audit driver: patches JEOD contact source inside the container, builds
# SIM_contact, runs RUN_point_off_center, and emits per-call dumps in the
# common `[#560/FULL] step=N stage=K body=B op=<name> <fields>` format used
# by `trick/audit_560/diff_streams.py`.
#
# Alignment with the Rust-side dump (gated by ASTRODYN_560_FULL_DUMP=1) is
# by (op, body, occurrence-in-stream) per `diff_streams.py`, so step/stage
# on the JEOD side are emitted as 0 — they're metadata only.
#
# Invocation (from astrodyn root):
#   docker run --rm \
#     -v "$(pwd)/trick/audit_560:/audit" \
#     jeod-trick:latest bash /audit/run_audit.sh
set -euo pipefail

CC=/jeod/models/interactions/contact/src/point_contact_pair.cc
test -f "$CC" || { echo "missing $CC"; exit 1; }
SP=/jeod/models/interactions/contact/src/spring_pair_interaction.cc
test -f "$SP" || { echo "missing $SP"; exit 1; }
FA=/jeod/models/interactions/contact/src/point_contact_facet.cc
test -f "$FA" || { echo "missing $FA"; exit 1; }

cp "$CC" /tmp/point_contact_pair.cc.orig
cp "$SP" /tmp/spring_pair_interaction.cc.orig
cp "$FA" /tmp/point_contact_facet.cc.orig

python3 - <<'PYEOF'
# Patch point_contact_pair.cc: emit `op=rel_pos`, `op=rel_vel`,
# `op=contact_arm_inertial body=0`, `op=geom_*` and `op=force_penetration_vec`
# from inside in_contact(). All in the `[#560/FULL]` format expected by
# diff_streams.py.
src = open('/jeod/models/interactions/contact/src/point_contact_pair.cc').read()
src = src.replace(
    '#include "utils/named_item/include/named_item.hh"',
    '#include "utils/named_item/include/named_item.hh"\n\n/* #560 audit */\n#include <cstdio>',
)

# Insert before `// calculate the forces on the facets` inside in_contact.
# Note JEOD's `rel_state.trans.position` is r_target - r_subject in SUBJECT
# body frame; the Rust side emits r_A - r_B in inertial (= rel_pos_a_wrt_b).
# For SIM_contact (identity attitudes) subject-frame = inertial and the two
# are related by sign: rel_pos_rust = -trans.position_jeod. We dump the
# **negated** values so the (op=rel_pos) name aligns numerically.
audit = r'''        /* #560/FULL audit instrumentation */
        {
            double rust_rel_pos[3] = { -rel_state.rel_state.trans.position[0],
                                       -rel_state.rel_state.trans.position[1],
                                       -rel_state.rel_state.trans.position[2] };
            double rust_rel_vel[3] = { -rel_state.rel_state.trans.velocity[0],
                                       -rel_state.rel_state.trans.velocity[1],
                                       -rel_state.rel_state.trans.velocity[2] };
            /* Subject's contact arm IN subject body frame = subject_contact_point.
               For identity attitudes this matches our inertial contact_arm_a. */
            fprintf(stderr,
                "[#560/FULL] step=0 stage=0 body=0 op=rel_pos x=%.17e y=%.17e z=%.17e\n",
                rust_rel_pos[0], rust_rel_pos[1], rust_rel_pos[2]);
            fprintf(stderr,
                "[#560/FULL] step=0 stage=0 body=0 op=rel_vel x=%.17e y=%.17e z=%.17e\n",
                rust_rel_vel[0], rust_rel_vel[1], rust_rel_vel[2]);
            fprintf(stderr,
                "[#560/FULL] step=0 stage=0 body=0 op=contact_arm_inertial x=%.17e y=%.17e z=%.17e\n",
                subject_contact_point[0], subject_contact_point[1], subject_contact_point[2]);
            /* target arm in subject frame = target_contact_point − rel_state.trans.position
               = -radius_b * normalize(rel_state.trans.position). In our inertial frame
               this is `normal * radius_b` where normal points from B to A. */
            double tgt_arm[3] = { target_contact_point[0] - rel_state.rel_state.trans.position[0],
                                  target_contact_point[1] - rel_state.rel_state.trans.position[1],
                                  target_contact_point[2] - rel_state.rel_state.trans.position[2] };
            /* But ours emits contact_arm_inertial for body=1 as (b->surface point) - r_B,
               which equals normal * radius_b. JEOD's tgt_arm above is exactly the negative.
               Negate to match. */
            fprintf(stderr,
                "[#560/FULL] step=0 stage=0 body=1 op=contact_arm_inertial x=%.17e y=%.17e z=%.17e\n",
                -tgt_arm[0], -tgt_arm[1], -tgt_arm[2]);
            /* Geometry-style dumps. Our `sep` = r_A - r_B = -trans.position;
               our `sep_len` = |sep|; our `normal` = sep / sep_len. */
            double sep_jeod[3] = { rust_rel_pos[0], rust_rel_pos[1], rust_rel_pos[2] };
            double sep_len_jeod = Vector3::vmag(sep_jeod);
            double normal_jeod[3] = { 0.0, 0.0, 0.0 };
            if (sep_len_jeod > 1.0e-20) {
                normal_jeod[0] = sep_jeod[0] / sep_len_jeod;
                normal_jeod[1] = sep_jeod[1] / sep_len_jeod;
                normal_jeod[2] = sep_jeod[2] / sep_len_jeod;
            }
            double sum_radii_jeod = point_subject->radius + point_target->radius;
            double penetration_depth_jeod = sum_radii_jeod - sep_len_jeod;
            fprintf(stderr,
                "[#560/FULL] step=0 stage=0 body=0 op=geom_sep x=%.17e y=%.17e z=%.17e\n",
                sep_jeod[0], sep_jeod[1], sep_jeod[2]);
            fprintf(stderr,
                "[#560/FULL] step=0 stage=0 body=0 op=geom_sep_len v=%.17e\n", sep_len_jeod);
            fprintf(stderr,
                "[#560/FULL] step=0 stage=0 body=0 op=geom_normal x=%.17e y=%.17e z=%.17e\n",
                normal_jeod[0], normal_jeod[1], normal_jeod[2]);
            fprintf(stderr,
                "[#560/FULL] step=0 stage=0 body=0 op=geom_penetration_depth v=%.17e\n",
                penetration_depth_jeod);
            /* Our `contact_point_on_a` = -normal * radius_a; matches -subject_cp in JEOD's
               subject-frame coords (since subject_cp = +n_AtoB*radius_a = -normal*radius_a). */
            double cp_a_rust[3] = { -subject_contact_point[0],
                                    -subject_contact_point[1],
                                    -subject_contact_point[2] };
            double cp_b_rust[3] = { -tgt_arm[0], -tgt_arm[1], -tgt_arm[2] };
            fprintf(stderr,
                "[#560/FULL] step=0 stage=0 body=0 op=geom_contact_point_on_a x=%.17e y=%.17e z=%.17e\n",
                cp_a_rust[0], cp_a_rust[1], cp_a_rust[2]);
            fprintf(stderr,
                "[#560/FULL] step=0 stage=0 body=1 op=geom_contact_point_on_b x=%.17e y=%.17e z=%.17e\n",
                cp_b_rust[0], cp_b_rust[1], cp_b_rust[2]);
            /* Rust's rel_vel formula: (v_A - v_B) + ω_A × cp_A - ω_B × cp_B. Equivalent to
               JEOD's `rel_velocity` (in subject frame, identity attitude) modulo sign. */
            fprintf(stderr,
                "[#560/FULL] step=0 stage=0 body=0 op=rel_vel_jeod_raw x=%.17e y=%.17e z=%.17e\n",
                rel_velocity[0], rel_velocity[1], rel_velocity[2]);
            fflush(stderr);
        }
'''
src = src.replace(
    '        // calculate the forces on the facets',
    audit + '        // calculate the forces on the facets',
)
open('/jeod/models/interactions/contact/src/point_contact_pair.cc', 'w').write(src)

# Patch spring_pair_interaction.cc: emit force component dumps in new format.
sp = open('/jeod/models/interactions/contact/src/spring_pair_interaction.cc').read()
sp = sp.replace(
    '/* JEOD includes */\n',
    '/* JEOD includes */\n\n/* #560 audit */\n#include <cstdio>\n',
)
sp_audit = r'''    /* #560/FULL audit — emit each force component in [#560/FULL] format. */
    {
        double spring_force_v[3];
        Vector3::scale(penetration_vector, this->spring_k, spring_force_v);
        /* The Rust side computes spring/damping/friction/total in INERTIAL frame.
           JEOD computes them in SUBJECT body frame. For SIM_contact identity
           attitudes the two are equal. */
        fprintf(stderr,
            "[#560/FULL] step=0 stage=0 body=0 op=force_penetration_vec x=%.17e y=%.17e z=%.17e\n",
            penetration_vector[0], penetration_vector[1], penetration_vector[2]);
        fprintf(stderr,
            "[#560/FULL] step=0 stage=0 body=0 op=force_spring x=%.17e y=%.17e z=%.17e\n",
            spring_force_v[0], spring_force_v[1], spring_force_v[2]);
        fprintf(stderr,
            "[#560/FULL] step=0 stage=0 body=0 op=force_damping x=%.17e y=%.17e z=%.17e\n",
            damping_force[0], damping_force[1], damping_force[2]);
        fprintf(stderr,
            "[#560/FULL] step=0 stage=0 body=0 op=force_friction x=%.17e y=%.17e z=%.17e\n",
            friction_force[0], friction_force[1], friction_force[2]);
        fprintf(stderr,
            "[#560/FULL] step=0 stage=0 body=0 op=force_total x=%.17e y=%.17e z=%.17e\n",
            force[0], force[1], force[2]);
        /* v_normal_mag = rel_velocity · nvec. JEOD computes this as `mag` (line 75).
           We emit it as a scalar named `force_v_normal_mag` matching Rust. */
        double v_normal_mag_jeod = Vector3::dot(rel_velocity, nvec);
        fprintf(stderr,
            "[#560/FULL] step=0 stage=0 body=0 op=force_v_normal_mag v=%.17e\n",
            v_normal_mag_jeod);
        fflush(stderr);
    }
'''
sp = sp.replace(
    '    /* add Force and Torque to Subject Facet */',
    sp_audit + '    /* add Force and Torque to Subject Facet */',
)
open('/jeod/models/interactions/contact/src/spring_pair_interaction.cc', 'w').write(sp)

# Patch point_contact_facet.cc::calculate_torque to dump torque outputs.
fa = open('/jeod/models/interactions/contact/src/point_contact_facet.cc').read()
fa = fa.replace(
    '/* JEOD includes */',
    '/* JEOD includes */\n\n/* #560 audit */\n#include <cstdio>',
)
# Find the cross-product line in calculate_torque and inject afterward.
# Look for `Vector3::cross(cm, tmp_force, tmp_torque);`
fa = fa.replace(
    'Vector3::cross(cm, tmp_force, tmp_torque);',
    'Vector3::cross(cm, tmp_force, tmp_torque);\n'
    '    /* #560/FULL audit — torque components per facet. */\n'
    '    {\n'
    '        /* Determine body=0 or =1 from this facet\'s ownership. Coarse:\n'
    '           every call dumps as body=0 first, then body=1 on the next call,\n'
    '           because JEOD calls subject->calculate_torque before target->calculate_torque\n'
    '           in spring_pair_interaction.cc. Use a toggling static. */\n'
    '        static int body_toggle = 0;\n'
    '        int b = body_toggle & 1;\n'
    '        body_toggle++;\n'
    '        fprintf(stderr,\n'
    '            "[#560/FULL] step=0 stage=0 body=%d op=arm_inertial x=%.17e y=%.17e z=%.17e\\n",\n'
    '            b, cm[0], cm[1], cm[2]);\n'
    '        fprintf(stderr,\n'
    '            "[#560/FULL] step=0 stage=0 body=%d op=torque_inertial x=%.17e y=%.17e z=%.17e\\n",\n'
    '            b, tmp_torque[0], tmp_torque[1], tmp_torque[2]);\n'
    '        fflush(stderr);\n'
    '    }',
)
open('/jeod/models/interactions/contact/src/point_contact_facet.cc', 'w').write(fa)

print('patched OK (contact files)')
PYEOF

# Patch dyn_body_integration.cc — dump body state at integrate() entry/exit.
python3 - <<'PYEOF'
path = '/jeod/models/dynamics/dyn_body/src/dyn_body_integration.cc'
src = open(path).read()
src = src.replace(
    '#include "../include/dyn_body.hh"',
    '#include "../include/dyn_body.hh"\n\n/* #560 audit */\n#include <cstdio>\n#include <cstring>',
)
# DynBody::integrate(double dyn_dt, unsigned int target_stage) — dump body
# state at function entry (= state going into stage K) and at exit (= state
# after stage K's advance).
entry_marker = 'er7_utils::IntegratorResult DynBody::integrate(double dyn_dt, unsigned int target_stage)\n{\n    er7_utils::IntegratorResult status(false);'
entry_replacement = entry_marker + r'''

    /* #560/FULL audit — dump per-stage body state at integrate() entry. */
    {
        const char* body_name = this->name.get_name().c_str();
        int b = (strstr(body_name, "veh1") != NULL) ? 0 : 1;
        const double* p = this->composite_body.state.trans.position;
        const double* v = this->composite_body.state.trans.velocity;
        double q0v = this->composite_body.state.rot.Q_parent_this.scalar;
        const double* qv = this->composite_body.state.rot.Q_parent_this.vector;
        const double* w = this->composite_body.state.rot.ang_vel_this;
        fprintf(stderr,
            "[#560/FULL] step=0 stage=%u body=%d op=integrate_in_pos x=%.17e y=%.17e z=%.17e\n",
            target_stage, b, p[0], p[1], p[2]);
        fprintf(stderr,
            "[#560/FULL] step=0 stage=%u body=%d op=integrate_in_vel x=%.17e y=%.17e z=%.17e\n",
            target_stage, b, v[0], v[1], v[2]);
        fprintf(stderr,
            "[#560/FULL] step=0 stage=%u body=%d op=integrate_in_q q0=%.17e q1=%.17e q2=%.17e q3=%.17e\n",
            target_stage, b, q0v, qv[0], qv[1], qv[2]);
        fprintf(stderr,
            "[#560/FULL] step=0 stage=%u body=%d op=integrate_in_omega x=%.17e y=%.17e z=%.17e\n",
            target_stage, b, w[0], w[1], w[2]);
        fflush(stderr);
    }
'''
src = src.replace(entry_marker, entry_replacement)

# Dump body state at integrate() exit (just before `return status`).
exit_marker = '    // Propagate the integrated state to other state descriptions.\n    propagate_state();\n\n    return status;\n}'
exit_replacement = r'''    // Propagate the integrated state to other state descriptions.
    propagate_state();

    /* #560/FULL audit — dump per-stage body state at integrate() exit. */
    {
        const char* body_name = this->name.get_name().c_str();
        int b = (strstr(body_name, "veh1") != NULL) ? 0 : 1;
        const double* p = this->composite_body.state.trans.position;
        const double* v = this->composite_body.state.trans.velocity;
        double q0v = this->composite_body.state.rot.Q_parent_this.scalar;
        const double* qv = this->composite_body.state.rot.Q_parent_this.vector;
        const double* w = this->composite_body.state.rot.ang_vel_this;
        fprintf(stderr,
            "[#560/FULL] step=0 stage=%u body=%d op=integrate_out_pos x=%.17e y=%.17e z=%.17e\n",
            target_stage, b, p[0], p[1], p[2]);
        fprintf(stderr,
            "[#560/FULL] step=0 stage=%u body=%d op=integrate_out_vel x=%.17e y=%.17e z=%.17e\n",
            target_stage, b, v[0], v[1], v[2]);
        fprintf(stderr,
            "[#560/FULL] step=0 stage=%u body=%d op=integrate_out_q q0=%.17e q1=%.17e q2=%.17e q3=%.17e\n",
            target_stage, b, q0v, qv[0], qv[1], qv[2]);
        fprintf(stderr,
            "[#560/FULL] step=0 stage=%u body=%d op=integrate_out_omega x=%.17e y=%.17e z=%.17e\n",
            target_stage, b, w[0], w[1], w[2]);
        fflush(stderr);
    }

    return status;
}'''
src = src.replace(exit_marker, exit_replacement)
open(path, 'w').write(src)
print('patched OK (dyn_body_integration.cc)')
PYEOF

# Patch ER7 rk4_second_order_ode_integrator.cc to dump velocity & accel at
# the top of each `integrate()` call — these are the k_v / k_a (or k_qdot /
# k_alpha for rotation) values for the current stage.
python3 - <<'PYEOF'
path = '/trick/trick_source/er7_utils/integration/rk4/src/rk4_second_order_ode_integrator.cc'
src = open(path).read()
src = src.replace(
    '#include "er7_utils/integration/core/include/integ_utils.hh"',
    '#include "er7_utils/integration/core/include/integ_utils.hh"\n\n/* #560 audit */\n#include <cstdio>',
)
# Inject at the top of RK4SimpleSecondOrderODEIntegrator::integrate (right
# after the `double step_factor;` line) — emits the translational derivatives
# at this stage. We can't know `body` here; dump it as -1 to indicate the
# JEOD `integrate_bodies` loop position needs to be inferred from
# occurrence order.
trans_marker = '   double step_factor;\n\n   /**\n    * ### Overview'
trans_replacement = r'''   double step_factor;

   /* #560/FULL audit — translational k_v/k_a at stage K input. */
   if (getenv("ASTRODYN_560_FULL_DUMP") != NULL) {
       /* state_size[0] is position size; [1] is velocity size. Both = 3 for
          a typical DynBody trans state. We emit only the first 3 components. */
       static int trans_call = 0;
       int b = (trans_call++) % 2;
       fprintf(stderr,
           "[#560/FULL] step=0 stage=%u body=%d op=k_v_trans x=%.17e y=%.17e z=%.17e\n",
           target_stage, b, velocity[0], velocity[1], velocity[2]);
       fprintf(stderr,
           "[#560/FULL] step=0 stage=%u body=%d op=k_a_trans x=%.17e y=%.17e z=%.17e\n",
           target_stage, b, accel[0], accel[1], accel[2]);
       fflush(stderr);
   }

   /**
    * ### Overview'''
src = src.replace(trans_marker, trans_replacement)
# Inject similar at top of RK4GeneralizedStepSecondOrderODEIntegrator::integrate
# for rotational (k_qdot derived from velocity = ω, k_alpha = accel).
rot_marker = 'IntegratorResult\nRK4GeneralizedStepSecondOrderODEIntegrator::integrate (\n   double dyn_dt,\n   unsigned int target_stage,\n   double const * ER7_UTILS_RESTRICT accel,\n   double * ER7_UTILS_RESTRICT velocity,\n   double * ER7_UTILS_RESTRICT position)\n{\n   double step_factor;'
rot_replacement = rot_marker + r'''

   /* #560/FULL audit — rotational k_qdot_omega/k_alpha at stage K input.
      `velocity` here is ω (angular velocity, body frame); `accel` is α.
      The Rust side emits `k_alpha` directly; `k_qdot` is derived from ω in
      compute_left_quat_deriv. We dump ω as `k_qdot_omega` for alignment
      and `k_alpha` as `k_alpha`. */
   if (getenv("ASTRODYN_560_FULL_DUMP") != NULL) {
       static int rot_call = 0;
       int b = (rot_call++) % 2;
       fprintf(stderr,
           "[#560/FULL] step=0 stage=%u body=%d op=k_qdot_omega x=%.17e y=%.17e z=%.17e\n",
           target_stage, b, velocity[0], velocity[1], velocity[2]);
       fprintf(stderr,
           "[#560/FULL] step=0 stage=%u body=%d op=k_alpha x=%.17e y=%.17e z=%.17e\n",
           target_stage, b, accel[0], accel[1], accel[2]);
       fflush(stderr);
   }'''
src = src.replace(rot_marker, rot_replacement)
open(path, 'w').write(src)
print('patched OK (rk4_second_order_ode_integrator.cc)')
PYEOF

# Rebuild Trick lib (needed for the ER7 patches to take effect).
cd /trick
make -j"$(nproc)" 2>&1 | tee /audit/run_output/build_trick.log | tail -10 || echo "trick rebuild warning (continuing)"

# Rebuild JEOD lib (contact + dyn_body patches).
cd /jeod/build
mkdir -p /audit/run_output
make -j"$(nproc)" 2>&1 | tee /audit/run_output/build_jeod.log | tail -10

# Build SIM_contact.
cd /jeod/models/interactions/contact/verif/SIM_contact
trick-CP 2>&1 | tee /audit/run_output/build_sim_cp.log | tail -10
make -j"$(nproc)" 2>&1 | tee /audit/run_output/build_sim.log | tail -10

# Run RUN_point_off_center from SIM root.
mkdir -p /audit/run_output
S_main_bin=$(ls /jeod/models/interactions/contact/verif/SIM_contact/S_main_*.exe 2>/dev/null | head -1)
echo "S_main binary: $S_main_bin"
ASTRODYN_560_FULL_DUMP=1 "$S_main_bin" SET_test/RUN_point_off_center/input.py > /audit/run_output/stdout.log 2> /audit/run_output/audit_stderr.log
echo "---audit lines---"
wc -l /audit/run_output/audit_stderr.log
echo "---first 10---"
head -10 /audit/run_output/audit_stderr.log
echo "---last 5---"
tail -5 /audit/run_output/audit_stderr.log
