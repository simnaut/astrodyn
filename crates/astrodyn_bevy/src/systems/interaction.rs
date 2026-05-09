// JEOD_INV: TS.01 — `<SelfRef>` / `<SelfPlanet>` are runtime-resolved storage-boundary wildcards; see `docs/JEOD_invariants.md` row TS.01 and the lint at `tests/self_ref_self_planet_discipline.rs`.
//! Bevy systems for [`AstrodynSet::Interaction`](crate::AstrodynSet::Interaction).
//!
//! Aerodynamic drag, gravity-gradient torque, solar radiation pressure
//! (flat-plate + cannonball), and Earth lighting (eclipse / albedo).
//!
//! Note: `earth_lighting_system` is grouped with the other interaction-flavor
//! kernels (it shares the shadow / illumination math used by SRP) but is
//! registered into [`AstrodynSet::DerivedState`](crate::AstrodynSet::DerivedState)
//! by [`crate::AstrodynPlugin`] / [`crate::register_planet_systems`] because
//! its output is consumed downstream of force collection.

use astrodyn::{Planet, RootInertial, SelfRef};
use bevy::prelude::*;
use glam::DVec3;

use crate::components::*;
use crate::frame_param::FrameOrigin;

use super::util::body_integ_origin_in_root;

/// Compute aerodynamic drag for entities with all required components.
///
/// Placed in `AstrodynSet::Interaction`.
// JEOD_INV: IN.03 — AerodynamicDrag.active gates computation (structural: no DragConfigC -> no drag)
#[allow(clippy::type_complexity)]
pub fn aero_drag_system<P: Planet>(
    // JEOD_INV: DB.21 — detached subtrees coast ballistically; skip
    // drag so `AerodynamicForceC` doesn't hold stale values that no
    // integrator consumes (the runner's split between `bodies` and
    // `detached_subtrees` only evaluates drag on the integrated set).
    mut query: Query<
        (
            &DragConfigC,
            &AtmosphericStateC<P>,
            &TranslationalStateC<P>,
            &RotationalStateC,
            Option<&StructuralTransformC>,
            &mut AerodynamicForceC,
        ),
        Without<crate::DetachedSubtreeStateC>,
    >,
) {
    for (drag_config, atmos, state, rot, struct_xform, mut aero_force) in &mut query {
        let t_struct_body = struct_xform.map_or(glam::DMat3::IDENTITY, |s| *s.0.matrix_ref());

        // `DragConfigC` and `TranslationalStateC` both store typed values;
        // the system reads them directly. The result carries
        // `StructuralFrame<SelfRef>` phantoms, which the structural-frame
        // `AerodynamicForceC` unwraps via `.raw_si()` for storage (the
        // structural-frame Component still uses raw DVec3; that's a
        // remaining typed-storage boundary).
        // allowed: typed↔raw kernel boundary
        let rot_untyped = astrodyn::typed_bridge::rot_typed_to_raw(&rot.0);
        let t_inertial_body = rot_untyped.quaternion.left_quat_to_transformation();
        let t_inertial_struct =
            astrodyn::compute_t_inertial_struct(&t_struct_body, &t_inertial_body);
        // The body velocity and atmospheric state both carry the
        // concrete planet `<P>` at the type level (matching the
        // system instantiation's `<P>` parameter, gated by the bodies
        // query filter), so they pass straight into the typed kernel
        // without a relabel.
        let result = astrodyn::compute_ballistic_drag_typed::<P, SelfRef>(
            &drag_config.0,
            &atmos.0,
            state.velocity,
            &t_inertial_struct,
        );

        aero_force.force = result.force.raw_si();
        aero_force.torque = result.torque.raw_si();
    }
}

/// Compute gravity gradient torque.
///
/// Placed in `AstrodynSet::Interaction`.
// JEOD_INV: IN.01 — GravityTorque.subject_body required (structural: query requires all components)
// JEOD_INV: IN.02 — GravityTorque.active gates computation (structural: no GravityTorqueC -> no torque)
pub fn gravity_torque_system(
    // JEOD_INV: DB.21 — detached subtrees coast ballistically; their
    // gravity gradient torque is no longer consumed by any integrator
    // and would otherwise hold stale values. Skip them.
    mut query: Query<
        (
            &GravityAccelerationC,
            &RotationalStateC,
            &MassPropertiesC,
            &mut GravityTorqueC,
        ),
        Without<crate::DetachedSubtreeStateC>,
    >,
) {
    for (grav, rot, mass, mut torque) in &mut query {
        // MassPropertiesC stores `InertiaTensor<BodyFrame<SelfRef>>`
        // directly; read it without lifting. Same for the rotational
        // state — it's already typed.
        // allowed: typed↔raw kernel boundary
        let rot_untyped = astrodyn::typed_bridge::rot_typed_to_raw(&rot.0);
        let t_parent_this = rot_untyped.quaternion.left_quat_to_transformation();
        torque.0 = astrodyn::compute_gravity_torque_typed::<SelfRef>(
            &grav.grav_grad,
            &t_parent_this,
            mass.0.inertia,
        );
    }
}

/// Compute illumination factor from all shadow-casting bodies.
fn compute_illum_factor<P: Planet>(
    vehicle_pos: DVec3,
    sun_pos: DVec3,
    shadow_bodies: &Query<(&TranslationalStateC<P>, &ShadowBodyC), Without<SunMarker>>,
) -> f64 {
    let mut illum = 1.0_f64;
    for (body_state, shadow) in shadow_bodies.iter() {
        let factor = astrodyn::compute_shadow_fraction(
            vehicle_pos,
            sun_pos,
            body_state.position.raw_si(),
            shadow.radius,
            astrodyn::SOLAR_RADIUS,
        );
        illum = illum.min(factor);
    }
    illum
}

/// Compute earth lighting (eclipse/albedo) for entities with `EarthLightingConfigC`.
///
/// Requires `SunMarker` and `MoonMarker` entities in the world.
///
/// Generic over `P: Planet` so the body's planet-inertial state and the
/// Sun / Moon `TranslationalStateC<P>` (which by convention store the
/// solar-system body positions in the body's planet-inertial frame for
/// the single-planet pipeline) match at the type level.
///
/// Placed in `AstrodynSet::DerivedState`.
#[allow(clippy::type_complexity)]
pub fn earth_lighting_system<P: Planet>(
    frame_origin: FrameOrigin,
    root_frame_entity: Res<crate::RootFrameEntityR>,
    parents: Query<&ChildOf>,
    mut query: Query<
        (
            &TranslationalStateC<P>,
            Option<&FrameEntityC>,
            &EarthLightingConfigC,
            &mut EarthLightingStateC,
        ),
        (Without<SunMarker>, Without<MoonMarker>),
    >,
    sun_query: Query<&TranslationalStateC<P>, With<SunMarker>>,
    moon_query: Query<&TranslationalStateC<P>, With<MoonMarker>>,
) {
    let sun_state = match sun_query.single() {
        Ok(s) => s,
        Err(bevy::ecs::query::QuerySingleError::NoEntities(_)) => {
            // No SunMarker present: clear stale earth lighting values
            for (_, _, _, mut lighting) in &mut query {
                lighting.0 = Default::default();
            }
            return;
        }
        Err(bevy::ecs::query::QuerySingleError::MultipleEntities(_)) => {
            panic!(
                "Multiple entities with SunMarker found in earth_lighting_system. \
                 JEOD assumes exactly one Sun body."
            );
        }
    };
    let moon_state = match moon_query.single() {
        Ok(s) => s,
        Err(bevy::ecs::query::QuerySingleError::NoEntities(_)) => {
            // No MoonMarker present: clear stale earth lighting values
            for (_, _, _, mut lighting) in &mut query {
                lighting.0 = Default::default();
            }
            return;
        }
        Err(bevy::ecs::query::QuerySingleError::MultipleEntities(_)) => {
            panic!(
                "Multiple entities with MoonMarker found in earth_lighting_system. \
                 JEOD assumes exactly one Moon body."
            );
        }
    };
    for (state, body_frame, config, mut lighting) in &mut query {
        // Earth lighting is a root-inertial-shift consumer (RF.10):
        // the kernel mixes the body position with the Sun and Moon
        // positions, all expected in absolute root-inertial
        // coordinates. For non-root-integrated bodies the body's
        // `<PlanetInertial<P>>` storage is integ-frame-
        // relative; lift it to absolute root-inertial via the integ-
        // origin shift before passing to the typed kernel. Sun and
        // Moon are root-integrated by the SunBundle / MoonBundle
        // construction (their frame entities are children of the
        // root frame), so their positions need no shift — only a
        // boundary relabel from `<PlanetInertial<P>>` to
        // `<RootInertial>` to satisfy the typed entry's frame contract.
        let (integ_origin, _integ_origin_vel) =
            body_integ_origin_in_root(body_frame, &parents, root_frame_entity.0, &frame_origin);
        let body_pos_rel = state.position.relabel_to::<RootInertial>();
        let body_pos = body_pos_rel + integ_origin;
        // Sun / Moon are root-integrated by SunBundle / MoonBundle
        // (their frame entity's parent is the root frame, integ
        // origin = zero); the relabel here is the consumer-boundary
        // step that pins the framing convention at the call site.
        let sun_pos = sun_state.position.relabel_to::<RootInertial>();
        let moon_pos = moon_state.position.relabel_to::<RootInertial>();
        lighting.0 = astrodyn::compute_earth_lighting_typed(
            body_pos,
            sun_pos,
            moon_pos,
            config.sun_radius,
            config.earth_radius,
            config.moon_radius,
        );
    }
}

/// Compute flat-plate SRP with thermal emission and shadow detection.
///
// JEOD_INV: IN.06 — RadiationPressure.active gates computation (structural: no FlatPlateConfigC → no SRP)
// JEOD_INV: IN.09 — RadiationSource planet must exist (SunMarker required; panics on multiple)
/// For entities with `FlatPlateConfigC`. Handles:
/// - Solar flux at vehicle distance
/// - Conical shadow from `ShadowBodyC` entities
/// - Per-plate absorption, diffuse/specular reflection, thermal emission
/// - Temperature integration (forward Euler)
/// - Force is rotated from structural to inertial by this system before writing `RadiationForceC`
///
/// Kinematic children of a `MassChildOf` chain (entities carrying
/// [`KinematicChildC`]) are excluded from this system. Until the
/// kinematic-propagation system (design-doc Section 15.3
/// `propagate_state_from_root_system`) lands, a kinematic child's
/// own `TranslationalStateC` / `RotationalStateC` are not advanced
/// in lock-step with the chain root — they stay frozen at whatever
/// the world had when the chain was assembled. Reading those stale
/// states to compute solar pressure here would silently produce SRP
/// for a position the body is no longer at. Excluding kinematic
/// children entirely (rather than feeding them stale state) is the
/// fail-loud-but-conservative choice: kinematic-child appendages get
/// no SRP this PR, and the follow-up that introduces propagated
/// child state will route SRP through the live composite-derived
/// values.
///
/// Placed in `AstrodynSet::Interaction`.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn flat_plate_srp_system<P: Planet>(
    frame_origin: FrameOrigin,
    root_frame_entity: Res<crate::RootFrameEntityR>,
    parents: Query<&ChildOf>,
    // Filter excludes both kinematic-chain children (their
    // `TranslationalStateC` / `RotationalStateC` stay frozen until
    // the kinematic-propagation system lands; computing SRP from
    // stale state would produce solar pressure at the wrong
    // location) and detached subtrees (they coast ballistically;
    // `RadiationForceC` and the per-stage thermal cache stay
    // zeroed because no integrator consumes their forces).
    // JEOD_INV: DB.21 — detached subtrees skip SRP.
    mut query: Query<
        (
            &mut FlatPlateConfigC,
            &TranslationalStateC<P>,
            Option<&RotationalStateC>,
            Option<&MassPropertiesC>,
            Option<&StructuralTransformC>,
            Option<&FrameEntityC>,
            &mut RadiationForceC,
        ),
        (
            Without<SunMarker>,
            Without<CannonballSrpC>,
            Without<crate::DetachedSubtreeStateC>,
            Without<KinematicChildC>,
        ),
    >,
    // Cleanup query for kinematic children: drop any prior-tick
    // `RadiationForceC` / `stage_inputs` left over from when the
    // entity was last in the main query (i.e. before it became a
    // chain member). Without this clear, `force_collection_system`
    // would still accumulate the stale SRP into the child's
    // `TotalForceC`, and `wrench_aggregation_system` would shift
    // that stale wrench up to the parent — silently producing SRP
    // for a position the body is no longer at.
    mut kinematic_cleanup: Query<
        (&mut FlatPlateConfigC, &mut RadiationForceC),
        (
            With<KinematicChildC>,
            Without<SunMarker>,
            Without<CannonballSrpC>,
        ),
    >,
    sun_query: Query<&TranslationalStateC<P>, With<SunMarker>>,
    shadow_bodies: Query<(&TranslationalStateC<P>, &ShadowBodyC), Without<SunMarker>>,
    time: Res<Time<Fixed>>,
) {
    // Drop stale state for any kinematic-child SRP body. Runs first
    // so a transition from non-kinematic → kinematic this tick
    // never carries a leftover SRP force into the wrench-aggregation
    // walk.
    for (mut flat_config, mut srp_force) in &mut kinematic_cleanup {
        flat_config.stage_inputs = None;
        srp_force.force = DVec3::ZERO;
        srp_force.torque = DVec3::ZERO;
    }

    let sun_state = match sun_query.single() {
        Ok(s) => Some(s),
        Err(bevy::ecs::query::QuerySingleError::NoEntities(_)) => None,
        Err(bevy::ecs::query::QuerySingleError::MultipleEntities(_)) => {
            panic!(
                "Multiple entities with SunMarker found. In JEOD, RadiationPressure \
                 has exactly one RadiationSource (value member). Ensure exactly one \
                 Sun entity exists."
            );
        }
    };

    let dt = time.delta_secs_f64();

    for (mut flat_config, state, rot, mass, struct_xform, body_frame, mut srp_force) in &mut query {
        // Clear per-step SRP state unconditionally (before the Sun check)
        // so derivative-mode entities don't retain stale `stage_inputs` or
        // force/torque if the Sun entity is removed between steps — which
        // would otherwise incorrectly drive the coupled RK4 path. Mirrors
        // the unconditional clearing in `astrodyn_runner::Simulation`.
        flat_config.stage_inputs = None;
        srp_force.force = DVec3::ZERO;
        srp_force.torque = DVec3::ZERO;

        let Some(sun_state) = sun_state else {
            continue;
        };

        // SRP is a root-inertial-shift consumer (RF.10): `sun_to_vehicle`
        // and the conical-shadow geometry both mix the body position
        // with the Sun / shadow-body positions, which are tagged
        // `<RootInertial>` (they integrate in root). For non-root-
        // integrated bodies the body's `<PlanetInertial<P>>`
        // storage is integ-frame-relative, so passing it raw to the
        // SRP / shadow kernels would compute `sun_to_vehicle` off by
        // the Earth–planet separation distance — wrong flux direction
        // and wrong illumination factor. Lift the body position to
        // absolute root-inertial via the integ-origin shift before
        // mixing. Both the scheduled-class and derivative-class
        // branches read `pos_raw` for `sun_to_vehicle`, distance, and
        // `compute_illum_factor`, so the shift applies to both — only
        // the temperature integration cadence differs between them.
        let (integ_origin, _integ_origin_vel) =
            body_integ_origin_in_root(body_frame, &parents, root_frame_entity.0, &frame_origin);
        let pos_raw = state.position.raw_si() + integ_origin.raw_si();
        // Sun is registered through `SunBundle` and integrates in the
        // root frame, so its `<PlanetInertial<P>>` storage is
        // numerically root-inertial; no integ-origin shift needed for
        // the Sun position.
        let sun_pos_raw = sun_state.position.raw_si();

        let sun_to_vehicle = pos_raw - sun_pos_raw;
        let distance = sun_to_vehicle.length();
        if distance < 1.0 {
            // Too close to the Sun to compute flux: force/torque/
            // stage_inputs already zeroed above.
            continue;
        }
        let flux_inertial_hat = sun_to_vehicle / distance;
        let flux_mag = astrodyn::solar_flux_at_distance(distance);

        // Shadow fraction (step-constant; matches JEOD's scheduled-class
        // shadow evaluation across all three integration orders).
        let illum_factor = compute_illum_factor(pos_raw, sun_pos_raw, &shadow_bodies);
        // The CoM is in the vehicle's structural frame; tag the typed
        // wildcard at this boundary so `FlatPlateStageInputs.center_grav`
        // (also typed) accepts it without a raw `DVec3` mismatch. Inner
        // SRP kernels go back through `.raw_si()`.
        let center_grav_raw = mass.map_or(DVec3::ZERO, |m| m.0.center_of_mass.raw_si());
        let center_grav = astrodyn::Vec3Ext::m_at::<astrodyn::StructuralFrame<astrodyn::SelfRef>>(
            center_grav_raw,
        );

        match flat_config.integration_order {
            astrodyn::ThermalIntegrationOrder::Scheduled => {
                // Scheduled-class (SIM_3_ORBIT): SRP force + Euler T once
                // per step. Force fed to the orbital integrator is
                // step-constant.
                let t_inertial_body = rot.map_or(glam::DMat3::IDENTITY, |r| {
                    r.0.q_inertial_body
                        .as_witness()
                        .left_quat_to_transformation()
                });
                let t_struct_body =
                    struct_xform.map_or(glam::DMat3::IDENTITY, |s| *s.0.matrix_ref());
                let t_inertial_struct =
                    astrodyn::compute_t_inertial_struct(&t_struct_body, &t_inertial_body);
                let flux_struct_hat = t_inertial_struct * flux_inertial_hat;

                let srp_result = astrodyn::compute_flat_plate_srp_thermal(
                    &flat_config.plates,
                    &flat_config.t_pow4_cached,
                    flux_struct_hat,
                    flux_mag,
                    // Drop typed wildcard for the kernel's raw-DVec3 contract.
                    center_grav.raw_si(),
                    illum_factor,
                );

                let force_inertial = t_inertial_struct.transpose() * srp_result.force;
                srp_force.force = force_inertial;
                srp_force.torque = srp_result.torque;

                // Integrate plate temperatures (forward Euler) — shared with
                // `Simulation` runner via `FlatPlateState::integrate_temperatures`.
                if dt > 0.0 {
                    flat_config.integrate_temperatures(&srp_result.temp_dots, dt);
                }
            }
            astrodyn::ThermalIntegrationOrder::DerivativeFirstOrder
            | astrodyn::ThermalIntegrationOrder::DerivativeRk4 => {
                // Derivative-class: SRP force (and optionally T) recomputed
                // per RK4 stage by the integration system. Cache the
                // step-start inputs on the plate state here; `RadiationForceC`
                // stays at the zero cleared above — the integration system
                // writes a representative final-stage value.
                // `sun_state.position` is stored as
                // `<PlanetInertial<P>>`; the SRP derivative
                // closure expects a root-inertial Sun position (RF.10
                // shift-site). Relabel at the boundary —
                // bit-identical numerics; the Sun's ephemeris-driven
                // inertial position numerically coincides with the
                // root frame's representation.
                let sun_pos_root = sun_state.position.relabel_to::<astrodyn::RootInertial>();
                flat_config.stage_inputs = Some(astrodyn::FlatPlateStageInputs {
                    sun_position: sun_pos_root,
                    illum_factor,
                    center_grav,
                });
            }
        }
    }
}

/// Compute cannonball SRP using JEOD's `RadiationDefaultSurface` formula.
///
/// Force = (flux/c) * cx_area * [1 + albedo*diffuse*(4/9)] * flux_hat * illum_factor.
///
/// For entities with `CannonballSrpC`. Requires `SunMarker` entity in the world.
/// Optional shadow detection via `ShadowBodyC` entities.
/// Writes force to `RadiationForceC` (torque is always zero for cannonball).
///
/// Placed in `AstrodynSet::Interaction`.
#[allow(clippy::type_complexity)]
pub fn cannonball_srp_system<P: Planet>(
    frame_origin: FrameOrigin,
    root_frame_entity: Res<crate::RootFrameEntityR>,
    parents: Query<&ChildOf>,
    // JEOD_INV: DB.21 — detached subtrees coast ballistically; skip
    // cannonball SRP so `RadiationForceC` doesn't hold stale values
    // that no integrator consumes.
    mut query: Query<
        (
            &CannonballSrpC,
            &TranslationalStateC<P>,
            Option<&FrameEntityC>,
            &mut RadiationForceC,
        ),
        (
            Without<SunMarker>,
            Without<FlatPlateConfigC>,
            Without<crate::DetachedSubtreeStateC>,
        ),
    >,
    sun_query: Query<&TranslationalStateC<P>, With<SunMarker>>,
    shadow_bodies: Query<(&TranslationalStateC<P>, &ShadowBodyC), Without<SunMarker>>,
) {
    let sun_state = match sun_query.single() {
        Ok(s) => s,
        Err(bevy::ecs::query::QuerySingleError::NoEntities(_)) => return,
        Err(bevy::ecs::query::QuerySingleError::MultipleEntities(_)) => {
            panic!(
                "Multiple entities with SunMarker found. \
                 Ensure exactly one Sun entity exists."
            );
        }
    };

    for (config, state, body_frame, mut srp_force) in &mut query {
        // Cannonball SRP is a root-inertial-shift consumer (RF.10):
        // the kernel mixes the body position with the Sun position
        // (expected root-inertial). Lift the body's
        // `<PlanetInertial<P>>` storage to absolute root-
        // inertial via the integ-origin shift before mixing — same
        // boundary discipline as the flat-plate / solar-beta sites.
        let (integ_origin, _integ_origin_vel) =
            body_integ_origin_in_root(body_frame, &parents, root_frame_entity.0, &frame_origin);
        let pos_raw = state.position.raw_si() + integ_origin.raw_si();
        let sun_pos_raw = sun_state.position.raw_si();
        let illum_factor = compute_illum_factor(pos_raw, sun_pos_raw, &shadow_bodies);

        srp_force.force = astrodyn::compute_cannonball_srp(
            pos_raw,
            sun_pos_raw,
            config.cx_area,
            config.albedo,
            config.diffuse,
            illum_factor,
        );
        srp_force.torque = DVec3::ZERO;
    }
}
