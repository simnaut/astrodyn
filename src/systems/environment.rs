//! Bevy systems for [`JeodSet::Environment`](crate::JeodSet::Environment).
//!
//! Gravity computation (point-mass + spherical harmonics + relativistic
//! corrections + tidal ΔC20) and atmosphere evaluation.

use bevy::prelude::*;
use glam::DVec3;
use jeod_sim::{IntegOrigin, IntegrationFrame, Planet, Position, RootInertial, Velocity};

use crate::components::*;
use crate::frame_param::FrameOrigin;
use crate::{AtmosphereModelR, SimulationTimeR};

use super::util::body_integ_origin_in_root;

/// Pre-computes gravity for each dynamic body.
///
/// Gravity is precomputed here in the Environment stage but is recomputed at
/// each integrator stage by the integration system for multi-stage accuracy.
///
/// Delegates to [`jeod_sim::accumulate_gravity`] for the per-body accumulation
/// loop, providing a closure that resolves Bevy entity references.
///
/// Bodies whose frame entity is a child of a non-root integration frame
/// have their integration-frame origin (relative to root inertial)
/// added to `body.position` to recover the absolute inertial position
/// for the gravity field; the same origin is passed to
/// [`jeod_sim::accumulate_gravity_typed`] so the differential gravity
/// correction subtracts the integ frame's own acceleration toward each
/// source. The integration frame is determined from the body's
/// `FrameEntityC` parent via `Query<&ChildOf>` (no explicit
/// integration-frame handle component), and the origin is queried via
/// the [`FrameOrigin`] SystemParam — typed
/// `(Position<RootInertial>, Velocity<RootInertial>)` directly, so no
/// `from_raw_si` lift is needed at the boundary.
#[allow(clippy::type_complexity)]
pub fn gravity_computation_system<P: Planet>(
    frame_origin: FrameOrigin,
    root_frame_entity: Res<crate::RootFrameEntityR>,
    parents: Query<&ChildOf>,
    // Filter excludes detached subtrees: only attached bodies participate
    // in gravity / force-collection / integration. Detached subtrees coast
    // ballistically (no force, no torque) via `step_detached_system`, so
    // populating `GravityAccelerationC` on them is wasted work and would
    // expose stale values to diagnostics / logging consumers. Mirrors the
    // runner's split between `Simulation::bodies` and
    // `Simulation::detached_subtrees` — gravity is only evaluated on the
    // integrated set.
    // JEOD_INV: DB.21 — detached subtrees skip gravity evaluation.
    mut bodies: Query<
        (
            Entity,
            &TranslationalStateC<P>,
            &GravityControlsC,
            &mut GravityAccelerationC,
            Option<&FrameEntityC>,
        ),
        Without<crate::DetachedSubtreeStateC>,
    >,
    sources: Query<(
        &GravitySourceC,
        Option<&PlanetFixedRotationC<P>>,
        &SourceInertialPositionC,
        Option<&SourceInertialVelocityC>,
        Option<&TidalDeltaC20C>,
        Option<&TidalConfigC>,
    )>,
) {
    // Project each `(entity, state, controls, accel, body_frame)` row
    // from `bodies.iter_mut()` into `(entity, GravityBodyInputs,
    // store_closure)`. The closure captures the row's
    // `Mut<'_, GravityAccelerationC>` by move and writes back to it
    // when the driver invokes it — Bevy change detection fires on the
    // deref-mut assignment as it would in a manual loop. Relabeling
    // `Position<PlanetInertial<P>>` → `Position<IntegrationFrame>` is
    // the boundary lift; the RF.10 shift to root inertial happens
    // inside `evaluate_body_gravity_typed` via the `IntegOrigin`.
    let body_iter = bodies
        .iter_mut()
        .map(|(entity, state, controls, accel, body_frame)| {
            let (integ_pos, integ_vel) =
                body_integ_origin_in_root(body_frame, &parents, root_frame_entity.0, &frame_origin);
            let inputs = jeod_sim::GravityBodyInputs {
                position: Position::<IntegrationFrame>::from_raw_si(state.position.raw_si()), // allowed: gravity RF.10 boundary — relabel `PlanetInertial<P>` → `IntegrationFrame` so the kernel's `IntegOrigin` shift is the structurally-guarded transition to `RootInertial`
                velocity: Velocity::<IntegrationFrame>::from_raw_si(state.velocity.raw_si()), // allowed: same gravity RF.10 boundary
                integ_origin: IntegOrigin {
                    position: integ_pos,
                    velocity: integ_vel,
                },
                controls: &controls.0,
            };
            let mut accel = accel;
            let store = move |result: jeod_sim::GravityAccelerationTyped<RootInertial>| {
                accel.0 = result;
            };
            (entity, inputs, store)
        });

    jeod_sim::run_gravity_stage(
        body_iter,
        |entity, source_entity| match sources.get(source_entity) {
            Ok((source, rot, pos, _, tidal, tidal_config)) => Some(jeod_sim::ResolvedSource {
                source: &source.0,
                rotation: rot.map(|r| r.0.matrix_ref()),
                position: pos.0.raw_si(),
                delta_c20: tidal.map_or(0.0, |t| t.0.value),
                // JEOD gates on n_deltacoeffs > 0 (tidal config
                // present), not on whether ΔC20 component exists.
                has_delta_coeffs: tidal_config.is_some(),
            }),
            Err(_) => {
                panic!(
                    "Entity {entity:?}: GravityControl references source \
                     {source_entity:?} which does not exist or lacks \
                     GravitySourceC + SourceInertialPositionC."
                );
            }
        },
        // Source velocity flows through the planet-agnostic
        // `SourceInertialVelocityC` (`Velocity<RootInertial>`). It is
        // opt-in: `PlanetBundle`, `SunBundle`, and `MoonBundle` do not
        // insert it, and `ephemeris_update_system` only writes through
        // it when it is already present (no auto-insert from
        // `EphemerisBodyC`). Sources without the component coast at
        // zero velocity for the relativistic correction — callers who
        // want PPN to see source motion must attach
        // `SourceInertialVelocityC` explicitly.
        |_entity, source_entity| {
            sources.get(source_entity).ok().map(|(s, _, p, v, _, _)| {
                let velocity = v.map(|v| v.0.raw_si()).unwrap_or(DVec3::ZERO);
                jeod_sim::ResolvedRelativisticSource {
                    mu: s.mu,
                    position: p.0.raw_si(),
                    velocity,
                }
            })
        },
    );
}

// JEOD_INV: AT.01 — active flag gates computation (no AtmosphericStateC component = no computation)
// JEOD_INV: AT.02 — atmosphere model pointer non-null for update (AtmosphereModelR resource checked)
/// Update atmospheric state for entities that have `AtmosphericStateC`.
///
/// Delegates to [`jeod_sim::evaluate_atmosphere`] for the per-body evaluation
/// pipeline (planet-fixed rotation, geodetic conversion, model dispatch, wind).
pub fn atmosphere_update_system<P: Planet>(
    atmos_model: Option<Res<AtmosphereModelR>>,
    sim_time: Option<Res<SimulationTimeR>>,
    planet_query: Query<&PlanetFixedRotationC<P>>,
    mut query: Query<(&TranslationalStateC<P>, &mut AtmosphericStateC<P>)>,
) {
    // JEOD_INV: AT.02 — early return if no atmosphere model resource
    let Some(model) = atmos_model else {
        return;
    };

    // JEOD_INV: AT.03 — planet-fixed position required for geodetic altitude
    let t_inertial_pfix = if let Some(entity) = model.planet_entity {
        let Ok(r) = planet_query.get(entity) else {
            panic!(
                "AtmosphereModelR.planet_entity is set ({entity:?}) but entity has no \
                 PlanetFixedRotationC. In JEOD, the planet-fixed frame is always \
                 available for atmosphere computation. Add PlanetFixedRotationC to \
                 the planet entity or set planet_entity to None for spherical fallback."
            );
        };
        Some(*r.0.matrix_ref())
    } else {
        None
    };

    let tai_tjt = sim_time.as_ref().map(|t| t.tai_tjt);
    for (state, mut atmos) in &mut query {
        // MET atmosphere requires time for seasonal variation. Check only when
        // entities with AtmosphericStateC actually exist (avoids panic when MET
        // is configured but no bodies need atmosphere yet).
        if tai_tjt.is_none() {
            if let jeod_sim::AtmosphereModel::Met(_) = &model.config.model {
                panic!(
                    "MET atmosphere requires SimulationTimeR resource for TJT. \
                     Ensure JeodPlugin is added (it provides SimulationTimeR)."
                );
            }
        }
        **atmos = jeod_sim::evaluate_atmosphere_typed::<P>(
            &model.config,
            state.position,
            t_inertial_pfix.as_ref(),
            tai_tjt,
        );
    }
}
