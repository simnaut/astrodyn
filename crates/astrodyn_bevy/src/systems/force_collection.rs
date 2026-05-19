// JEOD_INV: TS.01 — `<SelfRef>` / `<SelfPlanet>` are runtime-resolved storage-boundary wildcards; see `docs/JEOD_invariants.md` row TS.01 and the lint at `tests/self_ref_self_planet_discipline.rs`.
//! Bevy system for [`AstrodynSet::ForceCollection`](crate::AstrodynSet::ForceCollection).
//!
//! Aggregates per-step force / torque contributions from the interaction
//! systems into [`TotalForceC`] / [`FrameDerivativesC`] for the integrator.

use astrodyn::{
    Acceleration, AngularAcceleration, BodyFrame, Force, RootInertial, SelfRef, Torque,
};
use bevy::prelude::*;
use glam::DVec3;

use crate::components::*;

/// Collects non-gravity forces and all torques into `TotalForceC`.
///
/// Delegates to [`astrodyn::collect_and_resolve_forces`] for frame-aware
/// force/torque aggregation and frame derivative computation.
///
/// Gravity is intentionally **excluded** because the integration system
/// recomputes it at each RK4 stage for 4th-order accuracy. Non-gravity
/// forces (aero, SRP) are approximately constant over one timestep and
/// are added to the per-stage gravity inside the integrator.
// JEOD_INV: DB.28 — forces collected in structural frame, rotated to inertial at root
// JEOD_INV: DB.29 — torques collected in structural frame, rotated to body at root
#[allow(clippy::type_complexity)]
pub fn force_collection_system(
    // JEOD_INV: DB.21 — detached subtrees coast ballistically; their
    // `TotalForceC` / `FrameDerivativesC` are no longer consumed by
    // any integrator. Skip them so downstream consumers don't see
    // stale aggregated forces on bodies that aren't reacting to them.
    mut query: Query<
        (
            &mut TotalForceC,
            Option<&mut FrameDerivativesC>,
            Option<&GravityAccelerationC>,
            Option<&RotationalStateC>,
            Option<&MassPropertiesC>,
            Option<&AerodynamicForceC>,
            Option<&RadiationForceC>,
            Option<&GravityTorqueC>,
            Option<&StructuralTransformC>,
            Option<&ExternalForceC>,
            Option<&ExternalTorqueC>,
            Option<&ExternalForceStructC>,
            Option<&ExternalTorqueStructC>,
        ),
        Without<crate::DetachedSubtreeStateC>,
    >,
) {
    for (
        mut total,
        derivs,
        grav,
        rot_state,
        mass,
        aero,
        srp,
        grav_torque,
        struct_xform,
        ext_force,
        ext_torque,
        ext_force_struct,
        ext_torque_struct,
    ) in &mut query
    {
        let t_struct_body = struct_xform.map_or(glam::DMat3::IDENTITY, |s| *s.0.matrix_ref());
        // `GravityAccelerationC` stores `Acceleration<RootInertial>`; the
        // existing `collect_and_resolve_forces` kernel takes a raw
        // `DVec3`, so drop the phantom here. The kernel's frame
        // contract (gravity in inertial) matches the component's
        // phantom by construction.
        let grav_accel = grav.map_or(DVec3::ZERO, |g| g.grav_accel.raw_si());

        // Map Bevy component references to astrodyn_interactions types for astrodyn.
        let aero_ref = aero.map(|a| astrodyn::AerodynamicForce {
            force: a.force,
            torque: a.torque,
        });
        let srp_ref = srp.map(|s| astrodyn::RadiationForce {
            force: s.force,
            torque: s.torque,
        });
        // GravityTorqueC stores `Torque<BodyFrame<SelfRef>>`; the
        // untyped `collect_and_resolve_forces` boundary still expects a
        // raw `DVec3` in the body frame — drop the phantom at the call
        // site only.
        let gravity_torque_val = grav_torque.map(|gt| gt.0.raw_si());

        // RotationalStateC and MassPropertiesC now wrap typed siblings;
        // convert to untyped at the kernel boundary. (The kernel
        // signature still takes the untyped form. Migrating the kernel
        // signature itself is out of scope for the ECS-surface typing;
        // the win here is at the ECS surface where mission code
        // interacts.)
        // allowed: typed↔raw kernel boundary
        let rot_untyped = rot_state.map(|r| astrodyn::typed_bridge::rot_typed_to_raw(&r.0));
        let mass_untyped = mass.map(|m| astrodyn::typed_bridge::mass_typed_to_raw(&m.0));

        let (collected, frame_derivs_raw) = astrodyn::collect_and_resolve_forces(
            aero_ref.as_ref(),
            srp_ref.as_ref(),
            gravity_torque_val,
            rot_untyped.as_ref(),
            t_struct_body,
            mass_untyped.as_ref(),
            grav_accel,
        );

        // The kernel returns untyped TotalForce / FrameDerivatives;
        // re-wrap as the component's typed form. The `RootInertial` and
        // `BodyFrame<SelfRef>` phantoms match the kernel's documented
        // frame contracts (force inertial, torque body).
        total.0 =
            // allowed: typed↔untyped kernel boundary; the kernel signature in
            // astrodyn is still untyped, so re-wrapping is the canonical
            // adapter pattern (analogous to the From<Untyped> impls in
            // src/components.rs).
            astrodyn::TotalForceTyped::<astrodyn::SelfRef, RootInertial>::from_untyped_unchecked(
                &collected,
            );
        let mut frame_derivs =
            // allowed: typed↔untyped kernel boundary, see TotalForceTyped comment above
            astrodyn::FrameDerivativesTyped::<RootInertial, astrodyn::SelfRef>::from_untyped_unchecked(
                &frame_derivs_raw,
            );

        // Apply external force/torque (set by caller between steps).
        // Matches simulation.rs:846-855 logic. ExternalForceC and
        // ExternalTorqueC carry typed phantoms; the totals are typed
        // too, so the accumulator stays in typed land throughout.
        if let Some(ef) = ext_force {
            if ef.0.raw_si() != DVec3::ZERO {
                total.0.force += ef.0;
                if let Some(mass) = mass {
                    // `Force<RootInertial> / Mass → Acceleration<RootInertial>`
                    // is the typed identity here; we go through raw_si
                    // for the scalar inverse_mass multiply (it's an
                    // untyped f64 by design — see astrodyn_dynamics::mass
                    // doc on why inverse_mass stays untyped).
                    let accel_contrib = ef.0.raw_si() * mass.0.inverse_mass;
                    frame_derivs.trans_accel +=
                        // allowed: scalar inverse_mass is untyped by design; rewrap.
                        Acceleration::<RootInertial>::from_raw_si(accel_contrib);
                }
            }
        }
        if let Some(et) = ext_torque {
            if et.0.raw_si() != DVec3::ZERO {
                total.0.torque += et.0;
                if let Some(mass) = mass {
                    let alpha_contrib = mass.0.inverse_inertia * et.0.raw_si();
                    frame_derivs.rot_accel +=
                        // allowed: same untyped inverse_inertia boundary as above.
                        AngularAcceleration::<BodyFrame<SelfRef>>::from_raw_si(alpha_contrib);
                }
            }
        }

        // Struct-frame external force/torque: rotate to inertial / body
        // using the body's current attitude. Mirrors the runner's
        // `simulation/step/integrate.rs:85-105`, which in turn mirrors
        // JEOD's `dyn_body_collect.cc:219-221`
        // (`extern_forc_inrtl = T_inertial_struct^T · extern_forc_struct`).
        if ext_force_struct.is_some() || ext_torque_struct.is_some() {
            let ef_struct_raw = ext_force_struct.map_or(DVec3::ZERO, |c| c.0.raw_si());
            let et_struct_raw = ext_torque_struct.map_or(DVec3::ZERO, |c| c.0.raw_si());
            if ef_struct_raw != DVec3::ZERO || et_struct_raw != DVec3::ZERO {
                let t_inertial_body = rot_state.map_or(glam::DMat3::IDENTITY, |r| {
                    r.0.q_inertial_body
                        .as_witness()
                        .left_quat_to_transformation()
                });
                let t_inertial_struct =
                    astrodyn::compute_t_inertial_struct(&t_struct_body, &t_inertial_body);
                let force_inertial = t_inertial_struct.transpose() * ef_struct_raw;
                let torque_body = t_struct_body * et_struct_raw;
                // allowed: typed re-wrap at the totals accumulator —
                // `force_inertial` is in root-inertial by construction
                // (computed from a structural-frame load and the body's
                // attitude), so the phantom is asserted at the boundary.
                total.0.force += Force::<RootInertial>::from_raw_si(force_inertial);
                // allowed: typed re-wrap — `torque_body` is in the body
                // frame by construction (rotated from structural via
                // `t_struct_body`).
                total.0.torque += Torque::<BodyFrame<SelfRef>>::from_raw_si(torque_body);
                if let Some(mass) = mass {
                    if force_inertial != DVec3::ZERO {
                        let accel_contrib = force_inertial * mass.0.inverse_mass;
                        frame_derivs.trans_accel +=
                            // allowed: scalar inverse_mass is untyped by design; rewrap.
                            Acceleration::<RootInertial>::from_raw_si(accel_contrib);
                    }
                    if torque_body != DVec3::ZERO {
                        let alpha_contrib = mass.0.inverse_inertia * torque_body;
                        frame_derivs.rot_accel +=
                            // allowed: same untyped inverse_inertia boundary as above.
                            AngularAcceleration::<BodyFrame<SelfRef>>::from_raw_si(alpha_contrib);
                    }
                }
            }
        }

        if let Some(mut derivs) = derivs {
            derivs.0 = frame_derivs;
        }
    }
}
