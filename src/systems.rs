//! Bevy `Systems` that delegate per-body work to `jeod_sim` per-body
//! orchestration functions. Each system queries the relevant components,
//! calls into `jeod_sim`, and writes the result back. No physics
//! algorithms live here.

use bevy::prelude::*;
use glam::DVec3;
use jeod_sim::{
    Acceleration, AngularAcceleration, BodyFrame, Force, Position, RootInertial, SelfPlanet,
    SelfRef, Torque, Velocity,
};

use crate::components::*;
use crate::AtmosphereModelR;
use crate::FrameTreeR;
use crate::SimulationTimeR;

// ── Frame-tree source registration ──

/// Auto-register every gravity-source entity (carrying [`GravitySourceC`])
/// into [`FrameTreeR`] at startup, then insert [`SourceFrameIdC`] on
/// the entity. Sources are added as children of the existing root
/// inertial frame; their initial position comes from
/// [`SourceInertialPositionC`] and (when present) initial velocity from
/// [`SourceInertialVelocityC`].
///
/// A [`SourcePfixFrameIdC`] is additionally inserted iff the source
/// also carries [`PlanetFixedRotationC`] — that's the indicator
/// `planet_fixed_rotation_system` filters on; without it the source
/// never rotates and a pfix node would be a permanent identity that
/// `source_pfix_rotation()` would mis-report as `Some(identity)`. When
/// `PlanetFixedRotationC` is present and `RotationModelC` is omitted,
/// the same `EarthRNP` default applies as in
/// `planet_fixed_rotation_system`.
///
/// This is the Bevy analog of `jeod_runner::Simulation::add_source` —
/// it makes the lifted source-mutation helpers (issue #71 item 5)
/// usable directly via [`crate::SourceMutator`].
///
/// **Divergence from jeod_runner**: every source becomes a child of
/// the root frame, including the central body. `jeod_runner` renames
/// the root frame to `<central>.inertial` and reuses it. The Bevy
/// adapter keeps a generic root and treats all sources uniformly so
/// the registration order doesn't matter and so adding a body in a
/// non-Earth-central simulation doesn't require special-casing
/// "central" sources. Frame-switch parity (issue #71 items 2-4) lives
/// at the orchestration layer, where this divergence is invisible.
#[allow(clippy::type_complexity)]
pub fn register_source_frames_system(
    mut commands: Commands,
    mut frame_tree: ResMut<FrameTreeR>,
    root: Res<crate::RootFrameIdR>,
    sources: Query<
        (
            Entity,
            Option<&Name>,
            &SourceInertialPositionC,
            Option<&SourceInertialVelocityC>,
            Option<&RotationModelC>,
            Option<&PlanetFixedRotationC>,
        ),
        (With<GravitySourceC>, Without<SourceFrameIdC>),
    >,
) {
    for (entity, name, pos, vel, rotation_model, pfix_rot) in &sources {
        let label = name
            .map(|n| n.as_str().to_string())
            .unwrap_or_else(|| format!("source{:?}", entity));
        // Initialize the source frame node from the entity's current
        // typed state. Reading both Position and (optional) Velocity
        // lets sources that already carry a non-zero
        // `SourceInertialVelocityC` start with the right velocity in the
        // tree; sources without the velocity component get zero, matching
        // their ECS state. (Phase B PR #260 review fixup.)
        let init_pos = pos.0.raw_si();
        let init_vel = vel.map_or(glam::DVec3::ZERO, |v| v.0.raw_si());
        let inertial_id = frame_tree.0.add_child(
            root.0,
            format!("{label}.inertial"),
            jeod_sim::RefFrameKind::Inertial,
            jeod_sim::RefFrameState {
                trans: jeod_sim::RefFrameTrans {
                    position: init_pos,
                    velocity: init_vel,
                },
                rot: jeod_sim::RefFrameRot::default(),
            },
        );
        let mut entity_cmds = commands.entity(entity);
        entity_cmds.insert(SourceFrameIdC(inertial_id));

        // Create a pfix child frame only if this source actually rotates.
        // The presence of `PlanetFixedRotationC` is the indicator —
        // `planet_fixed_rotation_system` queries `&mut PlanetFixedRotationC`,
        // so an entity without it never rotates, and a pfix node would be
        // a permanent identity that `source_pfix_rotation()` would
        // mis-report as `Some(identity)` instead of `None`. Plain
        // point-mass sources spawned without `PlanetFixedRotationC` get no
        // pfix node, matching `jeod_runner` for the same case (PR #260
        // round-2 review fixup). When rotation IS present and
        // `RotationModelC` is omitted, the EarthRNP default applies —
        // same default as `planet_fixed_rotation_system`.
        if pfix_rot.is_some() {
            let default_model = jeod_sim::RotationModel::EarthRNP;
            let model_value = rotation_model.map_or(default_model, |m| m.0);
            if !matches!(model_value, jeod_sim::RotationModel::None) {
                let pfix_id = frame_tree.0.add_child(
                    inertial_id,
                    format!("{label}.pfix"),
                    jeod_sim::RefFrameKind::PlanetFixed,
                    jeod_sim::RefFrameState::default(),
                );
                entity_cmds.insert(SourcePfixFrameIdC(pfix_id));
            }
        }
    }
}

/// Register a [`SourcePfixFrameIdC`] for sources that were registered
/// without [`PlanetFixedRotationC`] and acquired it later.
/// [`register_source_frames_system`] filters by `Without<SourceFrameIdC>`,
/// so it cannot pick up an entity that gained `PlanetFixedRotationC`
/// after its initial registration. Without this pass,
/// `planet_fixed_rotation_system` would update the ECS rotation each
/// step but the frame tree would never get a pfix child, leaving
/// `source_pfix_rotation()` and any frame-tree consumer reporting "no
/// planet-fixed frame" for a source that is in fact rotating.
///
/// Same registration semantics as [`register_source_frames_system`]'s
/// pfix branch: gated on [`PlanetFixedRotationC`], `EarthRNP` default
/// when [`RotationModelC`] is absent, no node when the rotation model
/// is explicitly [`jeod_sim::RotationModel::None`].
///
/// **Reuse path**: when an entity carries a [`RetiredPfixFrameIdC`]
/// (the planet just toggled back from `RotationModel::None` to a
/// rotating model), this system reuses the stashed
/// [`jeod_sim::FrameId`] instead of allocating a fresh node — the
/// orphan is renamed back to the canonical `<label>.pfix` and its
/// state reset to identity. This bounds frame-tree growth at one
/// orphan per source regardless of toggle-cycle count and keeps
/// [`jeod_sim::FrameTree::find_by_name`] returning the live frame.
#[allow(clippy::type_complexity)]
pub fn register_pfix_frames_system(
    mut commands: Commands,
    mut frame_tree: ResMut<FrameTreeR>,
    sources: Query<
        (
            Entity,
            Option<&Name>,
            &SourceFrameIdC,
            Option<&RotationModelC>,
            Option<&RetiredPfixFrameIdC>,
        ),
        (
            With<GravitySourceC>,
            With<PlanetFixedRotationC>,
            Without<SourcePfixFrameIdC>,
        ),
    >,
) {
    for (entity, name, source_fid, rotation_model, retired) in &sources {
        let default_model = jeod_sim::RotationModel::EarthRNP;
        let model_value = rotation_model.map_or(default_model, |m| m.0);
        if matches!(model_value, jeod_sim::RotationModel::None) {
            continue;
        }
        let label = name
            .map(|n| n.as_str().to_string())
            .unwrap_or_else(|| format!("source{:?}", entity));
        let canonical_name = format!("{label}.pfix");
        let pfix_id = if let Some(retired_id) = retired {
            // Reuse the orphan node from the previous toggle cycle:
            // restore its canonical name, reset its state to identity,
            // and drop the marker. The node already has the right
            // parent (`source_fid.0`) since `planet_fixed_rotation_system`
            // does not move it on retirement.
            let node = frame_tree.0.get_mut(retired_id.0);
            node.name = canonical_name;
            node.state = jeod_sim::RefFrameState::default();
            commands.entity(entity).remove::<RetiredPfixFrameIdC>();
            retired_id.0
        } else {
            frame_tree.0.add_child(
                source_fid.0,
                canonical_name,
                jeod_sim::RefFrameKind::PlanetFixed,
                jeod_sim::RefFrameState::default(),
            )
        };
        commands.entity(entity).insert(SourcePfixFrameIdC(pfix_id));
    }
}

/// Sync each gravity source's typed state from the ECS components
/// (`SourceInertialPositionC` + optional `SourceInertialVelocityC`) into
/// its [`FrameTreeR`] inertial frame node each step. Mirrors
/// `jeod_runner::Simulation::update_ephemeris`'s post-DE4xx writeback to
/// the frame tree — required so frame-tree consumers
/// (`compute_relative_state`, `frame_origin`, frame-switch evaluation,
/// per-stage source interpolation in [`integration_system`]) see the
/// current source state rather than the registration-time snapshot.
///
/// Velocity source-of-truth precedence (PR #260 round-3 review fixup):
///
/// 1. [`SourceInertialVelocityC`] when present — the explicit
///    per-source velocity component.
/// 2. Otherwise [`TranslationalStateC`]'s velocity —
///    `ephemeris_update_system` populates it for ephemeris-driven
///    sources that don't carry the standalone velocity component
///    (Sun / Moon entities used by SRP / earth-lighting are typically
///    spawned this way via `SunBundle` / `MoonBundle`).
/// 3. Otherwise leave the frame-tree node's velocity unchanged.
///
/// Round 2 only consulted `SourceInertialVelocityC`, which left
/// ephemeris-only sources stuck at zero velocity in the frame tree.
///
/// Runs in `JeodSet::EphemerisUpdate` after `ephemeris_update_system`
/// (which writes the ECS components from DE4xx) so the FrameTreeR sync
/// sees the latest values.
#[allow(clippy::type_complexity)]
pub fn sync_source_to_frame_system(
    mut frame_tree: ResMut<FrameTreeR>,
    sources: Query<(
        &SourceFrameIdC,
        &SourceInertialPositionC,
        Option<&SourceInertialVelocityC>,
        Option<&TranslationalStateC>,
    )>,
) {
    for (fid, pos, vel, trans) in &sources {
        let node = frame_tree.0.get_mut(fid.0);
        node.state.trans.position = pos.0.raw_si();
        let velocity = vel
            .map(|v| v.0.raw_si())
            .or_else(|| trans.map(|t| t.0.velocity.raw_si()));
        if let Some(v) = velocity {
            node.state.trans.velocity = v;
        }
    }
}

/// Auto-register every vehicle entity (carrying [`TranslationalStateC`])
/// into [`FrameTreeR`] at startup, attaching [`BodyFrameIdC`] +
/// [`IntegFrameIdC`]. The body's integration frame is determined by:
///
/// 1. `IntegSourceC(Some(source_entity))` — child of that source's
///    `SourceFrameIdC` node (panics if the source isn't yet registered).
/// 2. Otherwise — child of the root inertial frame
///    ([`crate::RootFrameIdR`]).
///
/// The body's initial state is read from [`TranslationalStateC`] and
/// written into the new frame node so the tree is consistent from the
/// first step.
///
/// Runs at `Startup` and again before `JeodSet::EphemerisUpdate` to
/// catch dynamically-spawned bodies. Filters by
/// `Without<BodyFrameIdC>` so the registration is one-time per body.
/// Issue #71 items 2 and 4.
#[allow(clippy::type_complexity)]
pub fn register_body_frames_system(
    mut commands: Commands,
    mut frame_tree: ResMut<FrameTreeR>,
    root: Res<crate::RootFrameIdR>,
    sources: Query<&SourceFrameIdC>,
    bodies: Query<
        (
            Entity,
            Option<&Name>,
            &TranslationalStateC,
            Option<&IntegSourceC>,
        ),
        (
            With<TranslationalStateC>,
            With<DynamicsConfigC>,
            Without<BodyFrameIdC>,
        ),
    >,
) {
    for (entity, name, trans, integ_source) in &bodies {
        let label = name
            .map(|n| n.as_str().to_string())
            .unwrap_or_else(|| format!("body{:?}", entity));

        // Resolve the integration frame ID. Default: root inertial.
        let integ_frame_id = match integ_source.and_then(|c| c.0) {
            Some(source_entity) => sources
                .get(source_entity)
                .map(|c| c.0)
                .unwrap_or_else(|err| {
                    panic!(
                        "register_body_frames_system: body {entity:?} has \
                         IntegSourceC pointing at {source_entity:?}, but that \
                         entity is not a registered gravity source (missing \
                         SourceFrameIdC). Spawn the source via PlanetBundle \
                         before the body, or remove IntegSourceC. \
                         Underlying error: {err:?}"
                    )
                }),
            None => root.0,
        };

        // Body frame node carries the body's current state relative to its
        // integ frame. For root-integrated bodies this is the absolute
        // inertial state (matches existing Bevy behavior); for non-root
        // bodies the body's TranslationalStateC is interpreted as
        // already in integ-frame coordinates (mission code is
        // responsible for supplying state in the integ-frame).
        let body_state = jeod_sim::RefFrameState {
            trans: jeod_sim::RefFrameTrans {
                position: trans.0.position.raw_si(),
                velocity: trans.0.velocity.raw_si(),
            },
            rot: jeod_sim::RefFrameRot::default(),
        };
        let body_fid = frame_tree.0.add_child(
            integ_frame_id,
            format!("{label}.body"),
            jeod_sim::RefFrameKind::Body,
            body_state,
        );
        commands
            .entity(entity)
            .insert((BodyFrameIdC(body_fid), IntegFrameIdC(integ_frame_id)));
    }
}

/// Sync each vehicle's [`TranslationalStateC`] into its
/// [`BodyFrameIdC`] node in [`FrameTreeR`]. Mirrors
/// `jeod_runner::Simulation::step_internal`'s post-integration sync
/// (`step/integrate.rs:540-544`). Required so [`frame_switch_system`]
/// sees current body state when evaluating switch distances.
///
/// Runs in `JeodSet::Integration` after `integration_system` and
/// before `frame_switch_system`. Issue #71 item 2.
pub fn sync_body_to_frame_system(
    mut frame_tree: ResMut<FrameTreeR>,
    bodies: Query<(&TranslationalStateC, &BodyFrameIdC)>,
) {
    for (trans, body_fid) in &bodies {
        let node = frame_tree.0.get_mut(body_fid.0);
        node.state.trans.position = trans.0.position.raw_si();
        node.state.trans.velocity = trans.0.velocity.raw_si();
    }
}

/// Evaluate distance-based [`FrameSwitchesC`] entries for each body. On
/// trigger, the lifted [`jeod_sim::evaluate_and_apply_frame_switch`]
/// helper reparents the body in [`FrameTreeR`], rewrites the body's
/// translational state into the new integration frame's coordinates,
/// updates [`IntegFrameIdC`], and flips
/// [`GravityControlsC`]'s `differential` flags so the new central
/// source becomes non-differential.
///
/// Runs in `JeodSet::Integration` after [`sync_body_to_frame_system`].
/// Bodies without [`FrameSwitchesC`] are skipped. Issue #71 item 3.
///
/// JEOD reference: `dyn_body_frame_switch.cc:173-182`. The Bevy adapter
/// borrows the same logic via the lifted helper, so behavior is
/// bit-identical to `jeod_runner::Simulation` for the same scenario.
///
/// Phase C4: `FrameSwitchConfig<Entity>` and `GravityControls<Entity>`
/// flow into the generic helper directly via a closure-based source
/// lookup; there is no longer a `usize`-keyed bridge.
#[allow(clippy::type_complexity)]
pub fn frame_switch_system(
    mut frame_tree: ResMut<FrameTreeR>,
    root: Res<crate::RootFrameIdR>,
    sources: Query<&SourceFrameIdC>,
    mut bodies: Query<(
        Entity,
        &mut TranslationalStateC,
        &BodyFrameIdC,
        &mut IntegFrameIdC,
        &mut FrameSwitchesC,
        &mut GravityControlsC,
    )>,
) {
    // Snapshot the count of registered sources for error diagnostics.
    let num_sources = sources.iter().count();

    for (body_entity, mut trans, body_fid, mut integ_fid, mut switches, mut gravity_controls) in
        &mut bodies
    {
        if switches.0.is_empty() {
            continue;
        }
        // Translate TranslationalStateC into a raw struct the lifted
        // helper can write in place. The (DVec3 read × 2) + (DVec3 write × 2)
        // cost only fires when a switch actually triggers.
        let mut raw_trans = jeod_sim::TranslationalState {
            position: trans.0.position.raw_si(),
            velocity: trans.0.velocity.raw_si(),
        };
        let body_idx = body_entity.index().index() as usize;

        let switched = jeod_sim::evaluate_and_apply_frame_switch(
            &mut frame_tree.0,
            root.0,
            body_fid.0,
            &mut integ_fid.0,
            &mut raw_trans,
            &mut switches.0,
            &mut gravity_controls.0,
            // Closure: maps a target `Entity` to its source-inertial
            // FrameId in the tree. Returns `None` if the entity isn't a
            // registered source — the helper turns that into
            // `FrameSwitchTargetMissing`.
            |entity| sources.get(*entity).ok().map(|c| c.0),
            num_sources,
            body_idx,
        )
        .unwrap_or_else(|err| {
            panic!(
                "frame_switch_system: body {body_entity:?} switch evaluation failed: \
                 target source {target:?} is not a registered gravity source \
                 ({num} source entit{plural} currently registered). Spawn the source \
                 via PlanetBundle (which inserts SourceFrameIdC) before referencing \
                 it from a FrameSwitchConfig.",
                target = err.target_source,
                num = err.num_sources,
                plural = if err.num_sources == 1 { "y" } else { "ies" },
            )
        });

        if switched {
            // Re-wrap raw mutated state from the lifted helper (which
            // takes an untyped `TranslationalState`); boundary analogous
            // to `integrate_body`'s untyped kernel API.
            let pos_typed =
                jeod_sim::Position::<jeod_sim::RootInertial>::from_raw_si(raw_trans.position); // allowed: lifted-helper boundary
            let vel_typed =
                jeod_sim::Velocity::<jeod_sim::RootInertial>::from_raw_si(raw_trans.velocity); // allowed: lifted-helper boundary
            trans.0.position = pos_typed;
            trans.0.velocity = vel_typed;
            // `IntegFrameIdC` was updated in place by the helper;
            // `GravityControlsC.0` had its `differential` flags flipped
            // in place using `Entity` equality. `IntegSourceC` (the
            // config-time intent) is intentionally untouched — the
            // live truth lives in `IntegFrameIdC`.
        }
    }
}

// ── Time ──

/// Advance every JEOD-tracked time scale by the Bevy `Time<Fixed>` delta
/// each step (TAI/UTC/UT1/TDB/TT/GMST). Runs in
/// [`JeodSet::TimeUpdate`](crate::JeodSet::TimeUpdate).
// JEOD_INV: TM.03 — time types updated in dependency order (delegates to SimulationTime::advance)
pub fn time_advance_system(mut sim_time: ResMut<SimulationTimeR>, time: Res<Time<Fixed>>) {
    let dt = time.delta_secs_f64();
    sim_time.advance(dt);
}

// ── Ephemeris / Frames ──

/// Computes the inertial-to-planet-fixed rotation matrix for each entity
/// that carries a `PlanetFixedRotationC` component.
///
/// Dispatches per-entity via `RotationModelC`:
///
/// - `EarthRNP`: IAU 2000A precession-nutation + GAST + optional polar motion
/// - `MarsIAU`: IAU pole + spin + nutation Fourier series
/// - `MoonIAU`: IAU 2009 pole + prime meridian
/// - `MoonDE421`: DE421 BPC libration (requires `EphemerisR`)
/// - `None`: skip (leaves `PlanetFixedRotationC` unchanged)
///
/// When `RotationModelC` is absent, defaults to `EarthRNP`.
///
/// Earth RNP is lazy-computed once per step and reused across all `EarthRNP`
/// entities.
#[allow(clippy::type_complexity)]
pub fn planet_fixed_rotation_system(
    mut commands: Commands,
    sim_time: Res<SimulationTimeR>,
    polar: Option<Res<crate::PolarMotionR>>,
    ephemeris: Option<Res<crate::EphemerisR>>,
    mut frame_tree: ResMut<FrameTreeR>,
    mut query: Query<(
        Entity,
        &mut PlanetFixedRotationC,
        Option<&RotationModelC>,
        Option<&PlanetOmegaC>,
        Option<&mut PlanetAngularVelocityC>,
        Option<&SourcePfixFrameIdC>,
    )>,
) {
    let polar_params = polar.map(|p| (p.xp, p.yp));
    // Lazy-compute Earth RNP only if needed (most common case). Cache the
    // already-typed `FrameTransform` rather than the bare matrix so the
    // expensive `from_matrix` work (matrix→quat extraction + renormalization)
    // happens once per tick total, not once per EarthRNP entity per tick —
    // all EarthRNP entities share the same rotation each step.
    type EarthRot =
        jeod_sim::FrameTransform<jeod_sim::RootInertial, jeod_sim::PlanetFixed<SelfPlanet>>;
    let mut earth_rotation: Option<EarthRot> = Option::None;
    let mut earth_rotation_raw: Option<glam::DMat3> = Option::None;
    for (entity, mut rot, model, omega, ang_vel, pfix_fid) in &mut query {
        let default_model = jeod_sim::RotationModel::EarthRNP;
        let rotation_model = model.map_or(&default_model, |m| &m.0);
        // Track whether we wrote a rotation this tick — controls
        // `PlanetAngularVelocityC` and FrameTreeR pfix-node writes.
        let rotated = !matches!(rotation_model, jeod_sim::RotationModel::None);
        // Capture the raw DMat3 too so we can sync the FrameTreeR pfix node
        // via the lifted `sync_pfix_rotation` helper (which takes the matrix
        // and the planet omega — same data the rotation matrix carries).
        let mut raw_matrix: Option<glam::DMat3> = None;
        match rotation_model {
            jeod_sim::RotationModel::None => {}
            jeod_sim::RotationModel::EarthRNP => {
                let mat = *earth_rotation_raw.get_or_insert_with(|| {
                    jeod_sim::compute_t_parent_this_from_tjt_with_polar(
                        sim_time.gmst_seconds,
                        sim_time.tt_tjt(),
                        polar_params,
                    )
                });
                let rotation = *earth_rotation.get_or_insert_with(|| {
                    // allowed: matrix is JEOD's RNP-derived rotation; the
                    // RootInertial → PlanetFixed<SelfPlanet> phantoms match the kernel
                    // by construction
                    jeod_sim::FrameTransform::from_matrix(mat)
                });
                rot.0 = rotation;
                raw_matrix = Some(mat);
            }
            jeod_sim::RotationModel::MarsIAU => {
                let tt_s_since_j2000 =
                    (sim_time.tt_tjt() - jeod_sim::J2000_TT_TJT) * jeod_sim::SECONDS_PER_DAY;
                let mat = jeod_sim::rotation_mars::compute_mars_rotation(tt_s_since_j2000);
                // allowed: matrix from JEOD-ported IAU Mars rotation formula
                rot.0 = jeod_sim::FrameTransform::from_matrix(mat);
                raw_matrix = Some(mat);
            }
            jeod_sim::RotationModel::MoonIAU => {
                let tdb_jd = sim_time.tdb_julian_date();
                let tdb_s_since_j2000 =
                    (tdb_jd - jeod_sim::J2000_TT_JD) * jeod_sim::SECONDS_PER_DAY;
                let mat = jeod_sim::rotation_moon::compute_moon_rotation(tdb_s_since_j2000);
                // allowed: matrix from JEOD-ported IAU Moon rotation formula
                rot.0 = jeod_sim::FrameTransform::from_matrix(mat);
                raw_matrix = Some(mat);
            }
            jeod_sim::RotationModel::MoonDE421 => {
                let eph = ephemeris.as_ref().expect(
                    "RotationModel::MoonDE421 requires the EphemerisR resource with a BPC \
                     loaded. Insert EphemerisR before stepping the simulation, or switch the \
                     body to RotationModel::MoonIAU.",
                );
                let tdb_jd = sim_time.tdb_julian_date();
                let mat = eph
                    .get_body_rotation(jeod_sim::EphemerisBody::Moon, tdb_jd)
                    .unwrap_or_else(|err| {
                        panic!(
                            "Moon DE421 BPC rotation query failed at TDB JD {tdb_jd}: {err:?}. \
                             The loaded BPC kernel does not cover this epoch; load a kernel \
                             whose coverage includes the simulation epoch."
                        )
                    });
                // allowed: matrix from NASA SPICE BPC kernel (DE421 / Moon-PA)
                rot.0 = jeod_sim::FrameTransform::from_matrix(mat);
                raw_matrix = Some(mat);
            }
        }

        // ── Planet angular velocity ──
        // JEOD `planet_rnp.cc` writes `ang_vel_this = [0, 0, planet_omega]`
        // on the pfix frame node. Mirror that on (a) the `PlanetAngularVelocityC`
        // ECS component and (b) the FrameTreeR pfix node so velocity
        // composition both via the typed component and via the lifted
        // `compute_relative_state` reads the correct rate. Issue #71 item 1
        // + Copilot review (PR #260): the pfix-node sync via
        // `jeod_sim::sync_pfix_rotation` is what closes the frame-tree
        // half of the gap.
        if rotated {
            // Falling back to `0.0` for a rotating planet (`RotationModelC`
            // present but `PlanetOmegaC` absent) silently misreports the
            // pfix angular velocity as zero, which leaves issue #71 item 1
            // broken for manual-spawn call sites that include
            // `PlanetFixedRotationC` + `RotationModelC` but not
            // `PlanetOmegaC`. Map the rotation model to the canonical
            // `PlanetConfig::omega` when the explicit override is absent
            // (PR #260 round-2 review fixup).
            let default_omega = match rotation_model {
                jeod_sim::RotationModel::None => 0.0,
                jeod_sim::RotationModel::EarthRNP => jeod_sim::EARTH.omega,
                jeod_sim::RotationModel::MarsIAU => jeod_sim::MARS.omega,
                jeod_sim::RotationModel::MoonIAU | jeod_sim::RotationModel::MoonDE421 => {
                    jeod_sim::MOON.omega
                }
            };
            let omega_value = omega.map(|o| o.0).unwrap_or(default_omega);
            if let Some(mut ang_vel_c) = ang_vel {
                // Mint `AngularVelocity<PlanetFixed<SelfPlanet>>` from the
                // scalar `PlanetOmegaC`. JEOD's `planet_rnp.cc` writes
                // [0, 0, omega] in the pfix frame; this is the typed-API
                // boundary for that scalar → typed-vector lift.
                type PlanetAngVel = jeod_sim::AngularVelocity<jeod_sim::PlanetFixed<SelfPlanet>>;
                let raw = glam::DVec3::new(0.0, 0.0, omega_value);
                ang_vel_c.0 = PlanetAngVel::from_raw_si(raw); // allowed: scalar omega → typed AngularVelocity boundary
            }
            // Sync the FrameTreeR pfix node from the same data via the
            // lifted helper. This is what `jeod_runner` does in
            // `update_ephemeris`; the Bevy adapter previously skipped it,
            // so frame-tree consumers (`compute_relative_state` through
            // pfix) saw stale identity rotation and zero angular velocity.
            if let (Some(matrix), Some(pfix_fid)) = (raw_matrix, pfix_fid) {
                jeod_sim::sync_pfix_rotation(&mut frame_tree.0, pfix_fid.0, matrix, omega_value);
            }
        } else {
            // `RotationModel::None`: actively clear the rotation matrix,
            // angular velocity, and pfix-tree state. Without this, a
            // runtime toggle from a rotating model to `None` would leave
            // the last-tick rotation matrix on `PlanetFixedRotationC`,
            // the last-tick omega on `PlanetAngularVelocityC`, and the
            // last-tick `(matrix, omega)` on the FrameTreeR pfix node —
            // so frame-tree queries would still report a rotating
            // planet-fixed frame even though the source is configured
            // as non-rotating. PR #260 round-9 review fixup.
            // allowed: explicit identity clear when rotation model toggles to None;
            // the RootInertial → PlanetFixed<SelfPlanet> phantoms are correct by
            // construction (same shape as the rotating-branch from_matrix sites).
            rot.0 = jeod_sim::FrameTransform::from_matrix(glam::DMat3::IDENTITY);
            if let Some(mut ang_vel_c) = ang_vel {
                type PlanetAngVel = jeod_sim::AngularVelocity<jeod_sim::PlanetFixed<SelfPlanet>>;
                ang_vel_c.0 = PlanetAngVel::from_raw_si(glam::DVec3::ZERO); // allowed: zero-omega clear → typed AngularVelocity boundary
            }
            if let Some(pfix_fid) = pfix_fid {
                // Sync the pfix node to identity / zero so any consumer
                // still holding the FrameId reads a consistent state on
                // the toggle tick (before the component removal below
                // takes effect — Commands buffer until the next sync
                // point).
                jeod_sim::sync_pfix_rotation(
                    &mut frame_tree.0,
                    pfix_fid.0,
                    glam::DMat3::IDENTITY,
                    0.0,
                );
                // Clearing the pfix node's matrix/omega isn't enough
                // on its own — consumers that branch on the *presence*
                // of `SourcePfixFrameIdC` would keep treating the
                // source as rotating-capable, reintroducing the
                // `Some(identity)` vs `None` ambiguity. Mirror the
                // registration symmetry: `register_pfix_frames_system`
                // inserts the component when a source gains a
                // non-`None` rotation model; this branch removes it
                // when the model toggles back to `None`.
                //
                // The orphan tree node is kept alive (the frame tree
                // has no removal API since arena indices are stable),
                // but renamed to a sentinel so
                // `FrameTree::find_by_name("<label>.pfix")` won't
                // shadow a future live frame, and its FrameId is
                // stashed in `RetiredPfixFrameIdC` so the next toggle
                // back to a rotating model reuses this node instead
                // of allocating a fresh one. Without the rename +
                // reuse, every `None → rotating → None …` cycle
                // would leak an additional pfix node.
                let node = frame_tree.0.get_mut(pfix_fid.0);
                node.name = format!("{}.retired", node.name);
                commands
                    .entity(entity)
                    .remove::<SourcePfixFrameIdC>()
                    .insert(RetiredPfixFrameIdC(pfix_fid.0));
            }
        }
    }
}

/// Computes tidal ΔC20 for each gravity source that has a `TidalConfigC`.
///
/// Runs after `planet_fixed_rotation_system` so the rotation matrix is current.
/// Sources without `TidalConfigC` keep their default `TidalDeltaC20C::default()`
/// (a zero-valued [`jeod_sim::Ratio`]).
pub fn tidal_update_system(
    mut query: Query<(&TidalConfigC, &PlanetFixedRotationC, &mut TidalDeltaC20C)>,
) {
    for (config, rotation, mut delta) in &mut query {
        // `TidalConfigC` already wraps `TidalConfigTyped` — the dimensional
        // lift happened once at insertion (`TidalConfigC::from_untyped`),
        // so the system reads the typed value directly with no per-tick
        // `Vec` allocation or per-body f64 → typed conversion.
        // `compute_delta_c20_typed` returns `Ratio`, matching
        // `TidalDeltaC20C`'s storage type.
        delta.0 = jeod_sim::compute_delta_c20_typed(&config.0, rotation.0.matrix_ref());
    }
}

/// Updates source positions from DE4xx ephemeris each step.
///
/// Queries entities with `EphemerisBodyC` + `SourceInertialPositionC` and
/// looks up the current position/velocity from the `EphemerisR` resource.
/// Also updates `SourceInertialVelocityC` and `TranslationalStateC` when
/// present (velocity for relativistic corrections; translational state for
/// Sun/Moon entities used by SRP, solar beta, and earth lighting systems).
///
/// Placed in `JeodSet::EphemerisUpdate`.
pub fn ephemeris_update_system(
    ephemeris: Option<Res<crate::EphemerisR>>,
    sim_time: Res<SimulationTimeR>,
    mut query: Query<(
        &EphemerisBodyC,
        &mut SourceInertialPositionC,
        Option<&mut SourceInertialVelocityC>,
        Option<&mut TranslationalStateC>,
    )>,
) {
    let Some(eph) = ephemeris else {
        return;
    };
    let tdb_jd = sim_time.tdb_julian_date();
    for (ephem_body, mut source_pos, source_vel, trans_state) in &mut query {
        // Typed sibling: returns `(Position<RootInertial>, Velocity<RootInertial>)`
        // directly, matching the typed component storage. Bit-identical to
        // the deprecated f64 path — the kernel itself extracts SI base
        // values from ANISE and re-wraps them.
        let (pos_typed, vel_typed) = eph
            .get_state_typed(ephem_body.target, ephem_body.observer, tdb_jd)
            .unwrap_or_else(|e| {
                panic!(
                    "Ephemeris lookup failed for {:?} wrt {:?} at TDB JD {tdb_jd}: {e}",
                    ephem_body.target, ephem_body.observer,
                )
            });
        source_pos.0 = pos_typed;
        if let Some(mut sv) = source_vel {
            sv.0 = vel_typed;
        }
        if let Some(mut ts) = trans_state {
            // TranslationalStateC now wraps TranslationalStateTyped<RootInertial>;
            // assign the typed values directly. The frame phantom is
            // checked at the type level — pos_typed is Position<RootInertial>
            // by construction, matching the storage's RootInertial frame.
            ts.0.position = pos_typed;
            ts.0.velocity = vel_typed;
        }
    }
}

// ── Dynamics ──

/// Recompute derived mass quantities (`inverse_mass`, `inverse_inertia`) each step.
///
/// Port of JEOD's `(DYNAMICS, "scheduled") dyn_body.mass.update_mass_properties()`.
/// JEOD runs this every timestep so that runtime mass changes (fuel burn,
/// staging, attach/detach) are reflected in the dynamics before the next
/// derivative computation.
///
/// Placed before `JeodSet::EphemerisUpdate` so gravity and force collection
/// see current mass properties.
pub fn mass_update_system(mut query: Query<&mut MassPropertiesC>) {
    for mut mass in &mut query {
        mass.recompute_derived();
    }
}

/// Collects non-gravity forces and all torques into `TotalForceC`.
///
/// Delegates to [`jeod_sim::collect_and_resolve_forces`] for frame-aware
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
    mut query: Query<(
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
    )>,
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
    ) in &mut query
    {
        let t_struct_body = struct_xform.map_or(glam::DMat3::IDENTITY, |s| *s.0.matrix_ref());
        // `GravityAccelerationC` stores `Acceleration<RootInertial>`; the
        // existing `collect_and_resolve_forces` kernel takes a raw
        // `DVec3`, so drop the phantom here. The kernel's frame
        // contract (gravity in inertial) matches the component's
        // phantom by construction.
        let grav_accel = grav.map_or(DVec3::ZERO, |g| g.grav_accel.raw_si());

        // Map Bevy component references to jeod_interactions types for jeod_sim.
        let aero_ref = aero.map(|a| jeod_sim::AerodynamicForce {
            force: a.force,
            torque: a.torque,
        });
        let srp_ref = srp.map(|s| jeod_sim::RadiationForce {
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
        // signature itself is out of scope for #172 H1; the win here
        // is at the ECS surface where mission code interacts.)
        let rot_untyped = rot_state.map(|r| r.0.to_untyped());
        let mass_untyped = mass.map(|m| m.0.to_untyped());

        let (collected, frame_derivs_raw) = jeod_sim::collect_and_resolve_forces(
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
            // jeod_sim is still untyped, so re-wrapping is the canonical
            // adapter pattern (analogous to the From<Untyped> impls in
            // src/components.rs).
            jeod_sim::TotalForceTyped::<jeod_sim::SelfRef, RootInertial>::from_untyped_unchecked(
                &collected,
            );
        let mut frame_derivs =
            // allowed: typed↔untyped kernel boundary, see TotalForceTyped comment above
            jeod_sim::FrameDerivativesTyped::<RootInertial, jeod_sim::SelfRef>::from_untyped_unchecked(
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
                    // untyped f64 by design — see jeod_dynamics::mass
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

        if let Some(mut derivs) = derivs {
            derivs.0 = frame_derivs;
        }
    }
}

/// Advances translational (and optionally rotational) state by one timestep.
///
/// Delegates to [`jeod_sim::integrate_body`] for 6-DOF/3-DOF routing and
/// integration stepping. Gravity is recomputed at each intermediate state
/// for proper multi-stage accuracy.
///
/// The integration method is determined by the optional `IntegratorTypeC`
/// component (RK4, RKF45, GaussJackson, Abm4). When absent, RK4 is used.
/// GaussJackson requires `GaussJacksonStateC`; ABM4 requires `Abm4StateC`.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn integration_system(
    frame_tree: Res<FrameTreeR>,
    root: Res<crate::RootFrameIdR>,
    mut bodies: Query<(
        Entity,
        &DynamicsConfigC,
        &mut TranslationalStateC,
        Option<&mut RotationalStateC>,
        Option<&MassPropertiesC>,
        &GravityControlsC,
        &mut TotalForceC,
        Option<&IntegratorTypeC>,
        Option<&mut GaussJacksonStateC>,
        Option<&mut Abm4StateC>,
        Option<&mut FlatPlateConfigC>,
        Option<&StructuralTransformC>,
        Option<&mut RadiationForceC>,
        Option<&mut FrameDerivativesC>,
        Option<&IntegFrameIdC>,
    )>,
    sources: Query<
        (
            &GravitySourceC,
            Option<&PlanetFixedRotationC>,
            &SourceInertialPositionC,
            Option<&SourceInertialVelocityC>,
            Option<&TidalDeltaC20C>,
            Option<&TidalConfigC>,
            // Fallback velocity source for ephemeris-driven sources (Sun /
            // Moon via SunBundle / MoonBundle) that don't carry
            // SourceInertialVelocityC. PR #260 round-3 review.
            Option<&TranslationalStateC>,
        ),
        // Static disjointness vs. the `bodies` query's `&mut
        // TranslationalStateC`: no integrated body is also a gravity
        // source. Without this filter Bevy can't prove the queries
        // don't alias and panics with `assert_component_access_compatibility`.
        Without<DynamicsConfigC>,
    >,
    time: Res<Time<Fixed>>,
    sim_time: Res<SimulationTimeR>,
) {
    let dt = time.delta_secs_f64();
    if dt == 0.0 {
        return;
    }
    // Dynamic timestep matches `jeod_runner::run_integration`'s
    // `integ_dt = sim_dt * time_scale_factor` so reversed/scaled time
    // produces consistent gravity at RK sub-stages.
    let integ_dt = dt * sim_time.0.time_scale_factor;

    // Helper closure for gravity at an intermediate state — reused by both
    // the standard and coupled dispatch branches. The integrator passes
    // raw `DVec3` per-stage states (the integrator internals are not
    // yet typed); we wrap into `Position<RootInertial>` / `Velocity<RootInertial>`
    // for the typed `*_typed` kernels and unwrap before returning.
    //
    // `integ_origin_pos` / `integ_origin_vel` are the per-body integration
    // frame's translational state (relative to root) at step start. For
    // root-integrated bodies both are zero — the original behavior. For
    // non-root bodies the integ frame may itself be moving, so each
    // RK sub-stage advances the origin linearly by `time_frac * integ_dt`,
    // matching `jeod_runner::run_integration`. Source positions are
    // similarly interpolated when the integ frame moves, so the Newtonian
    // gravity field stays consistent across stages. PPN (relativistic)
    // corrections use step-start source state — runner does the same
    // (`step/integrate.rs:199-202`). Issue #71 item 4 + PR #260 review.
    let eval_gravity = |entity: Entity,
                        controls: &GravityControlsC,
                        pos: DVec3,
                        vel: DVec3,
                        integ_origin_pos: DVec3,
                        integ_origin_vel: DVec3,
                        time_frac: f64|
     -> DVec3 {
        // Per-stage interpolation of the integration frame's origin and
        // each source's position, mirroring jeod_runner's pattern in
        // `step/integrate.rs:172-184`. `sub_dt` is gated on the integ
        // frame actually moving so root-integrated bodies stay
        // bit-identical to the pre-N3 path.
        let stage_dt = time_frac * integ_dt;
        let stage_origin_pos = integ_origin_pos + integ_origin_vel * stage_dt;
        let sub_dt = if integ_origin_vel != DVec3::ZERO {
            stage_dt
        } else {
            0.0
        };
        // The standard `integrate_body` (and `integrate_body_coupled`
        // for the thermal-SRP path) accept a `gravity_fn` closure
        // that receives raw `DVec3` per-stage state. These lifts are
        // inside `jeod_sim` boundary territory, not at the Bevy ECS
        // surface that #172 H1 was specifically about.
        let typed_abs_pos = Position::<RootInertial>::from_raw_si(pos + stage_origin_pos); // allowed: integrator-kernel boundary
        let typed_abs_vel = Velocity::<RootInertial>::from_raw_si(vel + integ_origin_vel); // allowed: integrator-kernel boundary
        let typed_origin = Position::<RootInertial>::from_raw_si(stage_origin_pos); // allowed: integrator-kernel boundary

        // Helper: resolve a source's effective velocity, falling back to
        // `TranslationalStateC.velocity` when the explicit
        // `SourceInertialVelocityC` component is absent. PR #260 round-3
        // fix — without the fallback, ephemeris-driven Sun/Moon sources
        // (spawned via SunBundle/MoonBundle, which include
        // `TranslationalStateC` but not `SourceInertialVelocityC`) get
        // treated as stationary at every RK sub-stage.
        let source_vel =
            |v: Option<&SourceInertialVelocityC>, ts: Option<&TranslationalStateC>| -> DVec3 {
                v.map(|v| v.0.raw_si())
                    .or_else(|| ts.map(|t| t.0.velocity.raw_si()))
                    .unwrap_or(DVec3::ZERO)
            };

        let typed_accel = jeod_sim::accumulate_gravity_typed(
            typed_abs_pos,
            &controls.0,
            typed_origin,
            |source_entity| match sources.get(source_entity) {
                Ok((s, r, p, v, tidal, tidal_config, ts)) => {
                    let base_pos = p.0.raw_si();
                    let stage_pos = if sub_dt != 0.0 {
                        base_pos + source_vel(v, ts) * sub_dt
                    } else {
                        base_pos
                    };
                    Some(jeod_sim::ResolvedSource {
                        source: &s.0,
                        rotation: r.map(|r| r.0.matrix_ref()),
                        position: stage_pos,
                        delta_c20: tidal.map_or(0.0, |t| t.0.value),
                        has_delta_coeffs: tidal_config.is_some(),
                    })
                }
                Err(_) => {
                    panic!(
                        "Entity {entity:?}: GravityControl references source \
                         {source_entity:?} which does not exist or lacks \
                         GravitySourceC + SourceInertialPositionC."
                    );
                }
            },
        );
        let mut accel = typed_accel.grav_accel.raw_si();

        // PPN (relativistic) corrections use step-start source positions
        // and velocities — `jeod_runner::run_integration` snapshots both
        // outside the per-stage closure (`step/integrate.rs:199-202`),
        // so per-stage interpolation here would drift from runner.
        // (Round-2 PR #260 introduced the per-stage interpolation; round
        // 3 review R1 caught the divergence.)
        let rel = jeod_sim::accumulate_relativistic_corrections_typed(
            typed_abs_pos,
            typed_abs_vel,
            &controls.0,
            |source_entity| {
                sources
                    .get(source_entity)
                    .ok()
                    .map(|(s, _, p, v, _, _, ts)| {
                        // Step-start values for PPN — runner does the
                        // same (snapshots `src_pos`/`src_vel` outside
                        // the per-stage closure).
                        jeod_sim::ResolvedRelativisticSource {
                            mu: s.mu,
                            position: p.0.raw_si(),
                            velocity: source_vel(v, ts),
                        }
                    })
            },
        );
        accel += rel.raw_si();

        accel
    };

    for (
        entity,
        config,
        mut state,
        mut rot_state,
        mass,
        controls,
        mut total_force,
        integrator,
        mut gj_state,
        mut abm4_state,
        mut flat_config,
        struct_xform,
        mut srp_force,
        mut frame_derivs,
        integ_frame,
    ) in &mut bodies
    {
        // Per-body integration-frame origin (relative to root). Computed
        // once per step — the integ frame doesn't move during a single
        // integration step, so the multi-stage RK4 sub-evaluations
        // reuse the same value. Issue #71 item 4.
        let (integ_origin_pos, integ_origin_vel) = match integ_frame {
            Some(c) if c.0 != root.0 => jeod_sim::frame_origin(&frame_tree.0, root.0, c.0),
            _ => (DVec3::ZERO, DVec3::ZERO),
        };
        let integrator_type = integrator.map_or(jeod_sim::IntegratorType::Rk4, |c| c.0);
        if matches!(integrator_type, jeod_sim::IntegratorType::GaussJackson(..)) {
            assert!(
                gj_state.is_some(),
                "Entity {entity:?}: IntegratorTypeC is GaussJackson but \
                 GaussJacksonStateC component is missing. Create the state \
                 from the same config used in IntegratorTypeC, e.g.: \
                 GaussJacksonStateC(GaussJacksonState::new(config))"
            );
        }
        if matches!(integrator_type, jeod_sim::IntegratorType::Abm4) {
            assert!(
                abm4_state.is_some(),
                "Entity {entity:?}: IntegratorTypeC is Abm4 but \
                 Abm4StateC component is missing. Add \
                 Abm4StateC(Abm4State::new()) to the entity."
            );
        }

        // Derivative-class thermal fork: the SRP system cached step-start
        // inputs into `flat_config.stage_inputs`. Recompute SRP force +
        // temperature derivatives per RK4 stage through
        // `integrate_body_coupled`. See `jeod_runner::Simulation::step_internal`
        // for the sister implementation.
        let stage_inputs_and_order = flat_config
            .as_ref()
            .and_then(|fc| fc.stage_inputs.map(|si| (si, fc.integration_order)));
        if let Some((srp_inputs, thermal_order)) = stage_inputs_and_order {
            assert!(
                matches!(integrator_type, jeod_sim::IntegratorType::Rk4),
                "Entity {entity:?}: derivative-class ThermalIntegrationOrder \
                 requires RK4 integrator; use Scheduled or switch integrator.",
            );
            let t_struct_body = struct_xform.map_or(glam::DMat3::IDENTITY, |s| *s.0.matrix_ref());
            // Drop typed phantoms at the kernel boundary. `total_force`
            // accumulators are typed (`Force<RootInertial>` / `Torque<BodyFrame>`);
            // the integrator API still consumes raw `DVec3`.
            let non_grav_non_srp_force = total_force.force.raw_si();
            let constant_torque = total_force.torque.raw_si();
            let mut final_srp_inertial_force = DVec3::ZERO;
            let mut final_srp_torque = DVec3::ZERO;
            let mut k1_temp_dots: Option<Vec<f64>> = None;
            // Convert typed state to the untyped form the kernel wants.
            // After `integrate_body_coupled` mutates the untyped copies
            // we re-wrap as typed for storage.
            let mass_copy_untyped = mass.map(|m| m.0.to_untyped());
            let mut state_untyped = state.0.to_untyped();
            let mut rot_state_untyped = rot_state.as_ref().map(|r| r.0.to_untyped());
            let thermal = flat_config
                .as_mut()
                .expect("stage_inputs_and_order => flat_config present");
            jeod_sim::integrate_body_coupled(
                config,
                &mut state_untyped,
                rot_state_untyped.as_mut(),
                mass_copy_untyped.as_ref(),
                |stage_trans, stage_rot, stage_thermal, time_frac| {
                    let gravity_accel = eval_gravity(
                        entity,
                        controls,
                        stage_trans.position,
                        stage_trans.velocity,
                        integ_origin_pos,
                        integ_origin_vel,
                        time_frac,
                    );
                    let t_inertial_body = stage_rot.map_or(glam::DMat3::IDENTITY, |r| {
                        r.quaternion.left_quat_to_transformation()
                    });
                    let t_inertial_struct =
                        jeod_sim::compute_t_inertial_struct(&t_struct_body, &t_inertial_body);
                    // Per-stage flux recompute from intermediate vehicle
                    // position — matches JEOD's derivative-class
                    // `RadiationSource::calculate_flux`. Sun position is
                    // step-constant (ephemeris is scheduled-class).
                    //
                    // RF.10: `stage_trans.position` is the integrator's
                    // intermediate `DVec3` in the body's *integration*
                    // frame, which equals root inertial only when
                    // `IntegFrameIdC == root`. For `IntegFrameIdC != root`
                    // (issue #71 item 4) we shift via the per-stage origin
                    // before differencing against `srp_inputs.sun_position`
                    // (which is typed `Position<RootInertial>`). Mirrors
                    // `jeod_runner::run_integration`'s coupled SRP path
                    // (`crates/jeod_runner/src/simulation/step/integrate.rs:299-305`).
                    use jeod_sim::{Position, RootInertial};
                    let stage_dt = time_frac * integ_dt;
                    let stage_origin = if integ_origin_vel != DVec3::ZERO {
                        integ_origin_pos + integ_origin_vel * stage_dt
                    } else {
                        integ_origin_pos
                    };
                    let stage_pos_root: Position<RootInertial> =
                        // allowed: typed-API boundary — `stage_trans.position`
                        // arrives as the integrator's untyped intermediate
                        // DVec3; `stage_pos_root` is the root-inertial value
                        // after the integ-origin shift, ready for the typed
                        // `srp_inputs.sun_position` subtraction.
                        Position::<RootInertial>::from_raw_si(stage_trans.position + stage_origin);
                    let sun_to_vehicle: Position<RootInertial> =
                        stage_pos_root - srp_inputs.sun_position;
                    let sun_to_vehicle = sun_to_vehicle.raw_si();
                    let distance = sun_to_vehicle.length().max(1.0);
                    let stage_flux_inertial_hat = sun_to_vehicle / distance;
                    let stage_flux_mag = jeod_sim::solar_flux_at_distance(distance);
                    let flux_struct_hat = t_inertial_struct * stage_flux_inertial_hat;
                    let srp_result = jeod_sim::compute_flat_plate_srp_thermal(
                        &stage_thermal.plates,
                        &stage_thermal.t_pow4_cached,
                        flux_struct_hat,
                        stage_flux_mag,
                        srp_inputs.center_grav,
                        srp_inputs.illum_factor,
                    );
                    let srp_force_inertial = t_inertial_struct.transpose() * srp_result.force;
                    final_srp_inertial_force = srp_force_inertial;
                    final_srp_torque = srp_result.torque;
                    let temp_dots = match thermal_order {
                        jeod_sim::ThermalIntegrationOrder::DerivativeRk4 => srp_result.temp_dots,
                        jeod_sim::ThermalIntegrationOrder::DerivativeFirstOrder => {
                            if time_frac == 0.0 {
                                k1_temp_dots = Some(srp_result.temp_dots.clone());
                                srp_result.temp_dots
                            } else {
                                k1_temp_dots
                                    .as_ref()
                                    .expect("stage 1 runs before stages 2-4")
                                    .clone()
                            }
                        }
                        jeod_sim::ThermalIntegrationOrder::Scheduled => {
                            unreachable!("Scheduled bodies do not enter the coupled path")
                        }
                    };
                    // `srp_result.torque` is structural-frame per
                    // `FlatPlateSrpResult` docs; `constant_torque` is
                    // body-frame (from `collect_and_resolve_forces`).
                    // Rotate to body frame before summing so the coupled
                    // integrator's rotational dynamics are correct when
                    // `t_struct_body` != IDENTITY.
                    let srp_torque_body = t_struct_body * srp_result.torque;
                    jeod_sim::CoupledStageEval {
                        gravity_accel,
                        non_grav_force: non_grav_non_srp_force + srp_force_inertial,
                        torque: constant_torque + srp_torque_body,
                        temp_dots,
                    }
                },
                &mut thermal.0,
                dt,
                sim_time.0.time_scale_factor,
            );

            // Re-wrap kernel-mutated untyped state back into typed
            // components. The frame phantoms are unchanged (the typed
            // storage's `RootInertial` / `BodyFrame<SelfRef>` are the same
            // frames the kernel was operating in).
            // allowed: typed↔untyped kernel boundary (integrate_body_coupled
            // signature is untyped); analogous to From<Untyped> impls.
            state.0 = jeod_sim::TranslationalStateTyped::<RootInertial>::from_untyped_unchecked(
                &state_untyped,
            );
            if let (Some(rs), Some(ru)) = (rot_state.as_mut(), rot_state_untyped) {
                // allowed: same typed↔untyped kernel boundary as above.
                rs.0 = jeod_sim::RotationalStateTyped::<SelfRef>::from_untyped_unchecked(&ru);
            }

            // Write representative `RadiationForceC` from stage 4 so
            // `VehicleOutput`-equivalent observers still see the SRP force.
            if let Some(ref mut srp_force) = srp_force {
                srp_force.force = final_srp_inertial_force;
                srp_force.torque = final_srp_torque;
            }

            // Backfill `TotalForceC` and `FrameDerivativesC` with the
            // final-stage SRP contribution so downstream observers see
            // SRP-inclusive values, matching the Scheduled-mode invariant
            // that `TotalForceC` / `FrameDerivativesC` reflect every
            // applied force / resulting acceleration. In derivative modes
            // this is a "representative stage" (stage 4) snapshot, same
            // as `RadiationForceC` above.
            // allowed: SRP kernel returns DVec3; re-wrap into the typed
            // accumulators (`Force<RootInertial>` / `Torque<BodyFrame<SelfRef>>`).
            total_force.force += Force::<RootInertial>::from_raw_si(final_srp_inertial_force);
            let final_srp_torque_body = t_struct_body * final_srp_torque;
            // allowed: same SRP-kernel boundary.
            total_force.torque += Torque::<BodyFrame<SelfRef>>::from_raw_si(final_srp_torque_body);
            if let (Some(ref mut fd), Some(mass_p)) = (frame_derivs.as_mut(), mass_copy_untyped) {
                // allowed: typed↔untyped acceleration accumulator boundary.
                fd.trans_accel += Acceleration::<RootInertial>::from_raw_si(
                    final_srp_inertial_force * mass_p.inverse_mass,
                );
                // allowed: typed↔untyped angular-acceleration boundary.
                fd.rot_accel += AngularAcceleration::<BodyFrame<SelfRef>>::from_raw_si(
                    mass_p.inverse_inertia * final_srp_torque_body,
                );
            }
            continue;
        }

        // Standard (Scheduled or no-SRP) path. Same typed↔untyped
        // bridging as the coupled path: extract untyped at entry,
        // re-wrap typed at exit.
        let mut state_untyped = state.0.to_untyped();
        let mut rot_state_untyped = rot_state.as_ref().map(|r| r.0.to_untyped());
        let mass_untyped = mass.map(|m| m.0.to_untyped());
        jeod_sim::integrate_body(
            config,
            &mut state_untyped,
            rot_state_untyped.as_mut(),
            mass_untyped.as_ref(),
            |pos, vel, time_frac| {
                eval_gravity(
                    entity,
                    controls,
                    pos,
                    vel,
                    integ_origin_pos,
                    integ_origin_vel,
                    time_frac,
                )
            },
            total_force.force.raw_si(),
            total_force.torque.raw_si(),
            dt,
            sim_time.0.time_scale_factor,
            integrator_type,
            gj_state.as_mut().map(|g| &mut g.0),
            abm4_state.as_mut().map(|a| &mut a.0),
        );
        // Re-wrap kernel-mutated state back into typed components;
        // integrate_body signature is untyped, so re-wrapping is the
        // canonical adapter step (analogous to From<Untyped> impls).
        state.0 =
            // allowed: typed↔untyped kernel boundary
            jeod_sim::TranslationalStateTyped::<RootInertial>::from_untyped_unchecked(&state_untyped);
        if let (Some(rs), Some(ru)) = (rot_state.as_mut(), rot_state_untyped) {
            // allowed: typed↔untyped kernel boundary
            rs.0 = jeod_sim::RotationalStateTyped::<SelfRef>::from_untyped_unchecked(&ru);
        }
    }
}

// ── Gravity ──

/// Pre-computes gravity for each dynamic body.
///
/// Gravity is precomputed here in the Environment stage but is recomputed at
/// each integrator stage by the integration system for multi-stage accuracy.
///
/// Delegates to [`jeod_sim::accumulate_gravity`] for the per-body accumulation
/// loop, providing a closure that resolves Bevy entity references.
///
/// Bodies with [`IntegFrameIdC`] pointing at a non-root frame have their
/// integration-frame origin (relative to root inertial) added to
/// `body.position` to recover the absolute inertial position for the
/// gravity field; the same origin is passed to
/// [`jeod_sim::accumulate_gravity_typed`] so the differential gravity
/// correction subtracts the integ frame's own acceleration toward each
/// source. Issue #71 item 4. Bodies without [`IntegFrameIdC`] continue
/// to use the root inertial frame as before.
#[allow(clippy::type_complexity)]
pub fn gravity_computation_system(
    frame_tree: Res<FrameTreeR>,
    root: Res<crate::RootFrameIdR>,
    mut bodies: Query<(
        Entity,
        &TranslationalStateC,
        &GravityControlsC,
        &mut GravityAccelerationC,
        Option<&IntegFrameIdC>,
    )>,
    sources: Query<(
        &GravitySourceC,
        Option<&PlanetFixedRotationC>,
        &SourceInertialPositionC,
        Option<&SourceInertialVelocityC>,
        Option<&TidalDeltaC20C>,
        Option<&TidalConfigC>,
        // Fallback velocity source for ephemeris-driven sources that
        // don't carry SourceInertialVelocityC. PR #260 round-3.
        Option<&TranslationalStateC>,
    )>,
) {
    for (entity, state, controls, mut accel, integ_frame) in &mut bodies {
        // TranslationalStateC stores typed `Position<IntegrationFrame>` /
        // `Velocity<IntegrationFrame>` (issue #71 item 4 + #255). For
        // root-integrated bodies the integ frame numerically equals
        // root inertial, so the raw values match what gravity wants.
        // For non-root bodies we shift to absolute root-inertial
        // coordinates below via `IntegFrameIdC` + `frame_origin_typed`.
        // (Pre-#172-H1 the system extracted raw DVec3 here and called
        // `from_raw_si` to mint typed values; that bypass is gone.)
        let body_pos = state.position;
        let body_vel = state.velocity;

        // Integration-frame origin (relative to root). Zero for
        // root-integrated bodies. Issue #71 item 4 + Phase C5: typed
        // `frame_origin_typed::<RootInertial>` returns `Position<RootInertial>`
        // directly, so no `from_raw_si` lift is needed at the boundary.
        let (integ_origin, integ_origin_vel) = match integ_frame {
            Some(c) if c.0 != root.0 => {
                jeod_sim::frame_origin_typed::<RootInertial>(&frame_tree.0, root.0, c.0)
            }
            _ => (
                Position::<RootInertial>::zero(),
                Velocity::<RootInertial>::zero(),
            ),
        };
        let abs_pos = body_pos + integ_origin;

        let typed_accel = jeod_sim::accumulate_gravity_typed(
            abs_pos,
            &controls.0,
            integ_origin,
            |source_entity| match sources.get(source_entity) {
                Ok((source, rot, pos, _, tidal, tidal_config, _)) => {
                    Some(jeod_sim::ResolvedSource {
                        source: &source.0,
                        rotation: rot.map(|r| r.0.matrix_ref()),
                        position: pos.0.raw_si(),
                        delta_c20: tidal.map_or(0.0, |t| t.0.value),
                        // JEOD gates on n_deltacoeffs > 0 (tidal config
                        // present), not on whether ΔC20 component exists.
                        has_delta_coeffs: tidal_config.is_some(),
                    })
                }
                Err(_) => {
                    panic!(
                        "Entity {entity:?}: GravityControl references source \
                         {source_entity:?} which does not exist or lacks \
                         GravitySourceC + SourceInertialPositionC."
                    );
                }
            },
        );
        accel.0 = typed_accel;

        // Apply relativistic (post-Newtonian PPN) corrections after Newtonian
        // gravity, matching Simulation::step() stage 4b ordering. PPN depends
        // on |r_body - r_source| and v_body in inertial frame, so for
        // non-root-integrated bodies we lift `body_pos`/`body_vel` from
        // integ-frame coords into absolute inertial coords first.
        // Reuse the typed origin computed above: (integ_origin,
        // integ_origin_vel) are already Position/Velocity<Inertial>.
        let abs_body_pos = body_pos + integ_origin;
        let abs_body_vel = body_vel + integ_origin_vel;
        let rel_accel = jeod_sim::accumulate_relativistic_corrections_typed(
            abs_body_pos,
            abs_body_vel,
            &controls.0,
            |source_entity| {
                sources
                    .get(source_entity)
                    .ok()
                    .map(|(s, _, p, v, _, _, ts)| {
                        // Fall back to TranslationalStateC.velocity when
                        // SourceInertialVelocityC is absent — same precedence
                        // as `sync_source_to_frame_system`. PR #260 round-3.
                        let velocity = v
                            .map(|v| v.0.raw_si())
                            .or_else(|| ts.map(|t| t.0.velocity.raw_si()))
                            .unwrap_or(DVec3::ZERO);
                        jeod_sim::ResolvedRelativisticSource {
                            mu: s.mu,
                            position: p.0.raw_si(),
                            velocity,
                        }
                    })
            },
        );
        accel.grav_accel += rel_accel;
    }
}

// ── Atmosphere ──

// JEOD_INV: AT.01 — active flag gates computation (no AtmosphericStateC component = no computation)
// JEOD_INV: AT.02 — atmosphere model pointer non-null for update (AtmosphereModelR resource checked)
/// Update atmospheric state for entities that have `AtmosphericStateC`.
///
/// Delegates to [`jeod_sim::evaluate_atmosphere`] for the per-body evaluation
/// pipeline (planet-fixed rotation, geodetic conversion, model dispatch, wind).
pub fn atmosphere_update_system(
    atmos_model: Option<Res<AtmosphereModelR>>,
    sim_time: Option<Res<SimulationTimeR>>,
    planet_query: Query<&PlanetFixedRotationC>,
    mut query: Query<(&TranslationalStateC, &mut AtmosphericStateC)>,
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
        **atmos = jeod_sim::evaluate_atmosphere(
            &model.config,
            state.position.raw_si(),
            t_inertial_pfix.as_ref(),
            tai_tjt,
        );
    }
}

// ── Interactions ──

/// Compute aerodynamic drag for entities with all required components.
///
/// Placed in `JeodSet::Interaction`.
// JEOD_INV: IN.03 — AerodynamicDrag.active gates computation (structural: no DragConfigC -> no drag)
#[allow(clippy::type_complexity)]
pub fn aero_drag_system(
    mut query: Query<(
        &DragConfigC,
        &AtmosphericStateC,
        &TranslationalStateC,
        &RotationalStateC,
        Option<&StructuralTransformC>,
        &mut AerodynamicForceC,
    )>,
) {
    for (drag_config, atmos, state, rot, struct_xform, mut aero_force) in &mut query {
        let t_struct_body = struct_xform.map_or(glam::DMat3::IDENTITY, |s| *s.0.matrix_ref());

        // `DragConfigC` and `TranslationalStateC` both store typed values;
        // the system reads them directly. The result carries
        // `StructuralFrame<SelfRef>` phantoms, which the structural-frame
        // `AerodynamicForceC` unwraps via `.raw_si()` for storage (the
        // structural-frame Component still uses raw DVec3; that's the
        // remaining boundary inside the H1 migration).
        let rot_untyped = rot.0.to_untyped();
        // Bevy adapter stores body velocity as `Velocity<RootInertial>`
        // (current sims have root=Earth.inertial). Drag's typed sibling
        // expects `Velocity<PlanetInertial<P>>`; relabel via from_raw_si is
        // bit-identical and asserts the Earth-orbit assumption.
        use jeod_sim::{Earth, PlanetInertial, Velocity};
        // allowed: RootInertial → PlanetInertial<Earth> relabel for the
        // typed sibling; bit-identical (no arithmetic). Documented at #255.
        let drag_velocity = Velocity::<PlanetInertial<Earth>>::from_raw_si(state.velocity.raw_si());
        let result = jeod_sim::compute_drag_typed::<Earth, SelfRef>(
            &drag_config.0,
            atmos,
            drag_velocity,
            Some(&rot_untyped),
            t_struct_body,
        );

        aero_force.force = result.force.raw_si();
        aero_force.torque = result.torque.raw_si();
    }
}

/// Compute gravity gradient torque.
///
/// Placed in `JeodSet::Interaction`.
// JEOD_INV: IN.01 — GravityTorque.subject_body required (structural: query requires all components)
// JEOD_INV: IN.02 — GravityTorque.active gates computation (structural: no GravityTorqueC -> no torque)
pub fn gravity_torque_system(
    mut query: Query<(
        &GravityAccelerationC,
        &RotationalStateC,
        &MassPropertiesC,
        &mut GravityTorqueC,
    )>,
) {
    for (grav, rot, mass, mut torque) in &mut query {
        // MassPropertiesC stores `InertiaTensor<BodyFrame<SelfRef>>`
        // directly; read it without lifting. Same for the rotational
        // state — it's already typed.
        let rot_untyped = rot.0.to_untyped();
        torque.0 = jeod_sim::compute_gravity_torque_typed::<SelfRef>(
            &grav.grav_grad,
            &rot_untyped,
            mass.0.inertia,
        );
    }
}

/// Compute illumination factor from all shadow-casting bodies.
fn compute_illum_factor(
    vehicle_pos: DVec3,
    sun_pos: DVec3,
    shadow_bodies: &Query<(&TranslationalStateC, &ShadowBodyC), Without<SunMarker>>,
) -> f64 {
    let mut illum = 1.0_f64;
    for (body_state, shadow) in shadow_bodies.iter() {
        let factor = jeod_sim::compute_shadow_fraction(
            vehicle_pos,
            sun_pos,
            body_state.position.raw_si(),
            shadow.radius,
            jeod_sim::SOLAR_RADIUS,
        );
        illum = illum.min(factor);
    }
    illum
}

// ── Derived States ──

/// Compute orbital elements for entities with `OrbitalElementsConfigC`.
///
/// Placed in `JeodSet::DerivedState`.
pub fn orbital_elements_system(
    mut query: Query<(
        &TranslationalStateC,
        &OrbitalElementsConfigC,
        &mut OrbitalElementsC,
    )>,
    sources: Query<&GravitySourceC>,
) {
    for (state, config, mut elements) in &mut query {
        let Ok(source) = sources.get(config.gravity_source) else {
            elements.0 = Default::default();
            continue;
        };
        // Position and velocity are typed `Position<RootInertial>` on the
        // Bevy adapter (current sims have root=Earth.inertial). Relabel
        // to `Position<PlanetInertial<Earth>>` for the typed sibling —
        // bit-identical relabel that asserts the documented assumption.
        use jeod_sim::{Earth, PlanetInertial, Position, Velocity};
        let mu_typed = jeod_sim::F64Ext::m3_per_s2(source.mu);
        // allowed: RootInertial → PlanetInertial<Earth> relabel for the
        // typed sibling; bit-identical (no arithmetic). Documented at #255.
        let pos = Position::<PlanetInertial<Earth>>::from_raw_si(state.position.raw_si());
        // allowed: same relabel as `pos` above.
        let vel = Velocity::<PlanetInertial<Earth>>::from_raw_si(state.velocity.raw_si());
        match jeod_sim::compute_orbital_elements_typed::<Earth>(mu_typed, pos, vel) {
            Ok(oe) => elements.0 = oe,
            Err(_) => elements.0 = Default::default(),
        }
    }
}

/// Compute Euler angles for entities with `EulerAnglesConfigC`.
///
/// Placed in `JeodSet::DerivedState`.
pub fn euler_angles_system(
    mut query: Query<(
        Option<&RotationalStateC>,
        &EulerAnglesConfigC,
        &mut EulerAnglesC,
    )>,
) {
    for (rot_opt, config, mut angles) in &mut query {
        if let Some(rot) = rot_opt {
            // The "_typed" function takes untyped input but returns
            // typed `[Angle; 3]` (the typed-output naming convention
            // documented in jeod_sim::derived). Convert at the call.
            let rot_untyped = rot.0.to_untyped();
            angles.0 = jeod_sim::compute_body_euler_angles_typed(&rot_untyped, config.sequence);
        } else {
            angles.0 = Default::default();
        }
    }
}

/// Compute LVLH frame for entities with `LvlhFrameC`.
///
/// Presence of `LvlhFrameC` alone enables computation (no separate config needed).
///
/// Placed in `JeodSet::DerivedState`.
pub fn lvlh_system(mut query: Query<(&TranslationalStateC, &mut LvlhFrameC)>) {
    for (state, mut lvlh) in &mut query {
        // Typed throughout — TranslationalStateC carries `RootInertial`
        // for the Bevy adapter (current sims have root=Earth.inertial).
        // The typed sibling expects `PlanetInertial<P>`; relabel via
        // `from_raw_si` is bit-identical and asserts the documented
        // assumption that root coincides with Earth.inertial here.
        use jeod_sim::{Earth, PlanetInertial};
        // allowed: RootInertial → PlanetInertial<Earth> relabel for the
        // typed sibling; bit-identical (no arithmetic). Documented at #255.
        let pos = jeod_sim::Position::<PlanetInertial<Earth>>::from_raw_si(state.position.raw_si());
        // allowed: same relabel as `pos` above.
        let vel = jeod_sim::Velocity::<PlanetInertial<Earth>>::from_raw_si(state.velocity.raw_si());
        lvlh.0 = jeod_sim::compute_body_lvlh_frame_typed::<Earth>(pos, vel);
    }
}

/// Compute geodetic state for entities with `GeodeticConfigC`.
///
/// Placed in `JeodSet::DerivedState`.
pub fn geodetic_system(
    mut query: Query<(&TranslationalStateC, &GeodeticConfigC, &mut GeodeticStateC)>,
    planets: Query<(&PlanetFixedRotationC, &PlanetC)>,
) {
    for (state, config, mut geodetic) in &mut query {
        let Ok((rot, planet)) = planets.get(config.planet) else {
            geodetic.0 = Default::default();
            continue;
        };
        // Position is already typed; only the ellipsoid radii lift
        // remains, which is the typed-units boundary on planet shape
        // (a config-time conversion, not a per-step bypass).
        use jeod_sim::F64Ext;
        use jeod_sim::{Earth, PlanetInertial};
        // allowed: RootInertial → PlanetInertial<Earth> relabel for the
        // typed sibling; bit-identical (no arithmetic). Documented at #255.
        let pos = jeod_sim::Position::<PlanetInertial<Earth>>::from_raw_si(state.position.raw_si());
        geodetic.0 = jeod_sim::compute_body_geodetic_typed::<Earth>(
            pos,
            rot.0.matrix_ref(),
            planet.r_eq.m(),
            planet.r_pol.m(),
        );
    }
}

/// Compute solar beta angle for entities with `SolarBetaC`.
///
/// Requires a `SunMarker` entity to exist in the world.
///
/// Placed in `JeodSet::DerivedState`.
pub fn solar_beta_system(
    mut query: Query<(&TranslationalStateC, &mut SolarBetaC), Without<SunMarker>>,
    sun_query: Query<&TranslationalStateC, With<SunMarker>>,
) {
    let sun_state = match sun_query.single() {
        Ok(s) => s,
        Err(bevy::ecs::query::QuerySingleError::NoEntities(_)) => {
            // No SunMarker present: clear stale solar beta values
            for (_, mut beta) in &mut query {
                beta.0 = Default::default();
            }
            return;
        }
        Err(bevy::ecs::query::QuerySingleError::MultipleEntities(_)) => {
            panic!(
                "Multiple entities with SunMarker found in solar_beta_system. \
                 JEOD assumes exactly one Sun body; ensure exactly one SunMarker entity exists."
            );
        }
    };
    for (state, mut beta) in &mut query {
        // Typed throughout — the kernel returns a typed `Angle`; unwrap
        // to radians for the (still f64) `SolarBetaC` storage.
        // `Angle.value` reads the SI base value (radian), matching
        // `Angle::get::<radian>()` — f64-equality is preserved.
        beta.0 = jeod_sim::compute_body_solar_beta_typed(
            state.position,
            state.velocity,
            sun_state.position,
        )
        .value;
    }
}

/// Compute earth lighting (eclipse/albedo) for entities with `EarthLightingConfigC`.
///
/// Requires `SunMarker` and `MoonMarker` entities in the world.
///
/// Placed in `JeodSet::DerivedState`.
#[allow(clippy::type_complexity)]
pub fn earth_lighting_system(
    mut query: Query<
        (
            &TranslationalStateC,
            &EarthLightingConfigC,
            &mut EarthLightingStateC,
        ),
        (Without<SunMarker>, Without<MoonMarker>),
    >,
    sun_query: Query<&TranslationalStateC, With<SunMarker>>,
    moon_query: Query<&TranslationalStateC, With<MoonMarker>>,
) {
    let sun_state = match sun_query.single() {
        Ok(s) => s,
        Err(bevy::ecs::query::QuerySingleError::NoEntities(_)) => {
            // No SunMarker present: clear stale earth lighting values
            for (_, _, mut lighting) in &mut query {
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
            for (_, _, mut lighting) in &mut query {
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
    for (state, config, mut lighting) in &mut query {
        lighting.0 = jeod_sim::compute_earth_lighting(
            state.position.raw_si(),
            sun_state.position.raw_si(),
            moon_state.position.raw_si(),
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
/// Placed in `JeodSet::Interaction`.
#[allow(clippy::type_complexity)]
pub fn flat_plate_srp_system(
    mut query: Query<
        (
            &mut FlatPlateConfigC,
            &TranslationalStateC,
            Option<&RotationalStateC>,
            Option<&MassPropertiesC>,
            Option<&StructuralTransformC>,
            &mut RadiationForceC,
        ),
        (Without<SunMarker>, Without<CannonballSrpC>),
    >,
    sun_query: Query<&TranslationalStateC, With<SunMarker>>,
    shadow_bodies: Query<(&TranslationalStateC, &ShadowBodyC), Without<SunMarker>>,
    time: Res<Time<Fixed>>,
) {
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

    for (mut flat_config, state, rot, mass, struct_xform, mut srp_force) in &mut query {
        // Clear per-step SRP state unconditionally (before the Sun check)
        // so derivative-mode entities don't retain stale `stage_inputs` or
        // force/torque if the Sun entity is removed between steps — which
        // would otherwise incorrectly drive the coupled RK4 path. Mirrors
        // the unconditional clearing in `jeod_runner::Simulation`.
        flat_config.stage_inputs = None;
        srp_force.force = DVec3::ZERO;
        srp_force.torque = DVec3::ZERO;

        let Some(sun_state) = sun_state else {
            continue;
        };

        // The SRP kernel (`compute_flat_plate_srp_thermal`) and shadow
        // helpers all consume raw DVec3. Extract once at the top so the
        // rest of the body matches the kernel's untyped surface.
        let pos_raw = state.position.raw_si();
        let sun_pos_raw = sun_state.position.raw_si();

        let sun_to_vehicle = pos_raw - sun_pos_raw;
        let distance = sun_to_vehicle.length();
        if distance < 1.0 {
            // Too close to the Sun to compute flux: force/torque/
            // stage_inputs already zeroed above.
            continue;
        }
        let flux_inertial_hat = sun_to_vehicle / distance;
        let flux_mag = jeod_sim::solar_flux_at_distance(distance);

        // Shadow fraction (step-constant; matches JEOD's scheduled-class
        // shadow evaluation across all three integration orders).
        let illum_factor = compute_illum_factor(pos_raw, sun_pos_raw, &shadow_bodies);
        let center_grav = mass.map_or(DVec3::ZERO, |m| m.0.center_of_mass.raw_si());

        match flat_config.integration_order {
            jeod_sim::ThermalIntegrationOrder::Scheduled => {
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
                    jeod_sim::compute_t_inertial_struct(&t_struct_body, &t_inertial_body);
                let flux_struct_hat = t_inertial_struct * flux_inertial_hat;

                let srp_result = jeod_sim::compute_flat_plate_srp_thermal(
                    &flat_config.plates,
                    &flat_config.t_pow4_cached,
                    flux_struct_hat,
                    flux_mag,
                    center_grav,
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
            jeod_sim::ThermalIntegrationOrder::DerivativeFirstOrder
            | jeod_sim::ThermalIntegrationOrder::DerivativeRk4 => {
                // Derivative-class: SRP force (and optionally T) recomputed
                // per RK4 stage by the integration system. Cache the
                // step-start inputs on the plate state here; `RadiationForceC`
                // stays at the zero cleared above — the integration system
                // writes a representative final-stage value.
                flat_config.stage_inputs = Some(jeod_sim::FlatPlateStageInputs {
                    // `sun_state.position` is the typed component value;
                    // pass it directly so the typed phantom carries into
                    // the RK4 derivative closure (RF.10 structural guard).
                    sun_position: sun_state.position,
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
/// Placed in `JeodSet::Interaction`.
#[allow(clippy::type_complexity)]
pub fn cannonball_srp_system(
    mut query: Query<
        (&CannonballSrpC, &TranslationalStateC, &mut RadiationForceC),
        (Without<SunMarker>, Without<FlatPlateConfigC>),
    >,
    sun_query: Query<&TranslationalStateC, With<SunMarker>>,
    shadow_bodies: Query<(&TranslationalStateC, &ShadowBodyC), Without<SunMarker>>,
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

    for (config, state, mut srp_force) in &mut query {
        let pos_raw = state.position.raw_si();
        let sun_pos_raw = sun_state.position.raw_si();
        let illum_factor = compute_illum_factor(pos_raw, sun_pos_raw, &shadow_bodies);

        srp_force.force = jeod_sim::compute_cannonball_srp(
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

/// Process mass-tree attach/detach messages and sync composite properties.
///
/// Runs before interactions so that mass changes from staging are
/// reflected in the current step's interaction forces, force collection,
/// and integration.
///
/// Note: [`crate::MassTreeR`] must be present as a resource for attach/detach
/// messages to have any effect.
///
/// # Example
/// ```
/// use bevy::prelude::*;
/// use bevy_jeod::DetachEvent;
///
/// // A user-defined system that emits a DetachEvent for a known booster
/// // entity (e.g. one cached in a Resource).
/// #[derive(Resource)]
/// struct Booster(Entity);
///
/// fn detach_booster(
///     booster: Res<Booster>,
///     mut detach_messages: bevy::ecs::message::MessageWriter<DetachEvent>,
/// ) {
///     detach_messages.write(DetachEvent { child: booster.0 });
/// }
///
/// let mut app = App::new();
/// app.add_message::<DetachEvent>();
/// app.add_systems(Update, detach_booster);
/// ```
pub fn staging_system(
    tree: Option<ResMut<crate::MassTreeR>>,
    mut attach_events: bevy::ecs::message::MessageReader<crate::AttachEvent>,
    mut detach_events: bevy::ecs::message::MessageReader<crate::DetachEvent>,
    mut bodies: Query<(&crate::MassBodyIdC, &mut MassPropertiesC)>,
) {
    // No mass tree resource → drain events and return.
    let Some(mut tree) = tree else {
        attach_events.clear();
        detach_events.clear();
        return;
    };

    let mut changed_ids: Vec<jeod_sim::MassBodyId> = Vec::new();

    for evt in attach_events.read() {
        let child_id = bodies
            .get(evt.child)
            .unwrap_or_else(|_| {
                panic!(
                    "AttachEvent.child = {:?} is not a mass body — entity is missing MassBodyIdC \
                 and/or MassPropertiesC. Spawn the body via the mass-tree API before attaching.",
                    evt.child
                )
            })
            .0
             .0;
        let parent_id = bodies
            .get(evt.parent)
            .unwrap_or_else(|_| {
                panic!(
                    "AttachEvent.parent = {:?} is not a mass body — entity is missing MassBodyIdC \
                 and/or MassPropertiesC. Spawn the parent via the mass-tree API before attaching.",
                    evt.parent
                )
            })
            .0
             .0;
        tree.attach(child_id, parent_id, evt.offset, evt.t_parent_child);
        changed_ids.push(child_id);
        changed_ids.push(parent_id);
    }

    for evt in detach_events.read() {
        let child_id = bodies
            .get(evt.child)
            .unwrap_or_else(|_| {
                panic!(
                    "DetachEvent.child = {:?} is not a mass body — entity is missing MassBodyIdC \
                 and/or MassPropertiesC.",
                    evt.child
                )
            })
            .0
             .0;
        if let Some(parent_id) = tree.parent(child_id) {
            changed_ids.push(parent_id);
        }
        tree.detach(child_id);
        changed_ids.push(child_id);
    }

    // Sync composite mass properties for all affected nodes.
    // Walk up from each changed node to the root to capture cascading updates.
    if !changed_ids.is_empty() {
        let mut sync_ids: Vec<jeod_sim::MassBodyId> = Vec::new();
        for &id in &changed_ids {
            let mut current = id;
            sync_ids.push(current);
            while let Some(parent) = tree.parent(current) {
                sync_ids.push(parent);
                current = parent;
            }
        }
        sync_ids.sort_unstable();
        sync_ids.dedup();

        for (body_id, mut mass) in &mut bodies {
            if sync_ids.binary_search(&body_id.0).is_ok() {
                *mass = MassPropertiesC::from(tree.get(body_id.0).composite_properties);
            }
        }
    }
}
