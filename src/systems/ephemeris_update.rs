//! Bevy systems for [`JeodSet::EphemerisUpdate`](crate::JeodSet::EphemerisUpdate).
//!
//! Planet-fixed rotation (RNP), DE4xx ephemeris updates, and tidal ΔC20.

use bevy::prelude::*;
use jeod_sim::Planet;

use crate::components::*;
use crate::SimulationTimeR;

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
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn planet_fixed_rotation_system<P: Planet>(
    mut commands: Commands,
    sim_time: Res<SimulationTimeR>,
    polar: Option<Res<crate::PolarMotionR>>,
    ephemeris: Option<Res<crate::EphemerisR>>,
    mut query: Query<(
        Entity,
        &mut PlanetFixedRotationC<P>,
        Option<&RotationModelC>,
        Option<&PlanetOmegaC>,
        Option<&mut PlanetAngularVelocityC<P>>,
        Option<&PfixFrameEntityC>,
    )>,
    mut frame_rots: Query<&mut FrameRotC>,
    mut frame_ang_vels: Query<&mut FrameAngVelC>,
) {
    let polar_params = polar.map(|p| (p.xp, p.yp));
    // Lazy-compute Earth RNP once per system invocation when an
    // `EarthRNP` rotation-model entity is matched. Cache the
    // already-typed `FrameTransform` rather than the bare matrix so the
    // expensive `from_matrix` work (matrix→quat extraction + renormalization)
    // happens once per tick total, not once per EarthRNP entity per tick —
    // all matched entities share the same rotation each step.
    type PlanetRot<P> = jeod_sim::FrameTransform<jeod_sim::RootInertial, jeod_sim::PlanetFixed<P>>;
    let mut earth_rotation: Option<PlanetRot<P>> = Option::None;
    let mut earth_rotation_raw: Option<glam::DMat3> = Option::None;
    for (entity, mut rot, model, omega, ang_vel, pfix_frame_entity) in &mut query {
        let default_model = jeod_sim::RotationModel::EarthRNP;
        let rotation_model = model.map_or(&default_model, |m| &m.0);
        // Track whether we wrote a rotation this tick — controls
        // `PlanetAngularVelocityC` and pfix frame-entity writes.
        let rotated = !matches!(rotation_model, jeod_sim::RotationModel::None);
        // Capture the raw DMat3 too so we can sync the pfix frame
        // entity (FrameRotC + FrameAngVelC) from the same data.
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
                    // RootInertial → PlanetFixed<P> phantoms match the
                    // kernel by construction (system instantiation pins P).
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
        // on the pfix frame node. Mirror that on (a) the
        // `PlanetAngularVelocityC` ECS component and (b) the pfix
        // frame entity's `FrameRotC` / `FrameAngVelC` so velocity
        // composition both via the typed component and via
        // [`crate::frame_param::RelativeFrameState`] reads the
        // correct rate.
        if rotated {
            // Falling back to `0.0` for a rotating planet (`RotationModelC`
            // present but `PlanetOmegaC` absent) silently misreports the
            // pfix angular velocity as zero for manual-spawn call sites
            // that include `PlanetFixedRotationC` + `RotationModelC` but
            // not `PlanetOmegaC`. Map the rotation model to the
            // canonical `PlanetConfig::omega` when the explicit
            // override is absent.
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
                // Mint `AngularVelocity<PlanetFixed<P>>` from the
                // scalar `PlanetOmegaC`. JEOD's `planet_rnp.cc` writes
                // [0, 0, omega] in the pfix frame; this is the typed-API
                // boundary for that scalar → typed-vector lift.
                let raw = glam::DVec3::new(0.0, 0.0, omega_value);
                let typed = jeod_sim::AngularVelocity::<jeod_sim::PlanetFixed<P>>::from_raw_si(raw); // allowed: scalar omega → typed AngularVelocity boundary
                ang_vel_c.0 = typed;
            }
            // Write the pfix frame entity's FrameRotC / FrameAngVelC.
            // When `PfixFrameEntityC` is present the referenced entity
            // must be alive with FrameRotC / FrameAngVelC intact
            // (spawned by `register_pfix_frames_system`, torn down in
            // lockstep with the marker by the despawn observers and
            // the rotation-toggle retirement path). A stale handle
            // here would silently desync the pfix-frame state from
            // the rotation matrix on `PlanetFixedRotationC`.
            if let (Some(matrix), Some(pfix_fe)) = (raw_matrix, pfix_frame_entity) {
                let mut frame_rot = frame_rots.get_mut(pfix_fe.0).unwrap_or_else(|err| {
                    panic!(
                        "planet_fixed_rotation_system: source {entity:?} has \
                         PfixFrameEntityC({:?}) but that entity has no \
                         FrameRotC ({err:?}). The pfix frame entity must be \
                         alive with FrameRotC / FrameAngVelC attached \
                         (spawned by register_pfix_frames_system). Either \
                         remove the stale PfixFrameEntityC marker before \
                         despawning the pfix frame entity, or ensure the \
                         pfix frame entity stays alive for as long as the \
                         source carries the handle.",
                        pfix_fe.0
                    )
                });
                frame_rot.q_parent_this =
                    jeod_sim::JeodQuat::left_quat_from_transformation(&matrix);
                frame_rot.t_parent_this = matrix;
                let mut frame_av = frame_ang_vels.get_mut(pfix_fe.0).unwrap_or_else(|err| {
                    panic!(
                        "planet_fixed_rotation_system: source {entity:?} has \
                         PfixFrameEntityC({:?}) but that entity has no \
                         FrameAngVelC ({err:?}). The pfix frame entity must \
                         be alive with FrameRotC / FrameAngVelC attached \
                         (spawned by register_pfix_frames_system). Either \
                         remove the stale PfixFrameEntityC marker before \
                         despawning the pfix frame entity, or ensure the \
                         pfix frame entity stays alive for as long as the \
                         source carries the handle.",
                        pfix_fe.0
                    )
                });
                frame_av.0 = glam::DVec3::new(0.0, 0.0, omega_value);
            }
        } else {
            // `RotationModel::None`: actively clear the rotation
            // matrix, angular velocity, and pfix-frame state. Without
            // this, a runtime toggle from a rotating model to `None`
            // would leave the last-tick rotation matrix on
            // `PlanetFixedRotationC`, the last-tick omega on
            // `PlanetAngularVelocityC`, and the last-tick state on
            // the pfix frame entity — so frame-tree queries would
            // still report a rotating planet-fixed frame even though
            // the source is configured as non-rotating.
            // allowed: explicit identity clear when rotation model toggles to None;
            // the RootInertial → PlanetFixed<P> phantoms are correct by
            // construction (same shape as the rotating-branch from_matrix sites).
            rot.0 = jeod_sim::FrameTransform::from_matrix(glam::DMat3::IDENTITY);
            if let Some(mut ang_vel_c) = ang_vel {
                // allowed: zero-omega clear → typed AngularVelocity boundary
                ang_vel_c.0 = jeod_sim::AngularVelocity::<jeod_sim::PlanetFixed<P>>::from_raw_si(
                    glam::DVec3::ZERO,
                );
            }
            if let Some(pfix_fe) = pfix_frame_entity {
                // Clear the pfix frame entity's state to identity so
                // any `RelativeFrameState` reader sees the source as
                // non-rotating. When `PfixFrameEntityC` is present,
                // the referenced entity must be alive with
                // FrameRotC / FrameAngVelC intact; silently skipping
                // the clear would leave the pfix rotation/omega
                // frozen at the last rotating-tick value while the
                // toggle-to-`None` removes the public component.
                let mut frame_rot = frame_rots.get_mut(pfix_fe.0).unwrap_or_else(|err| {
                    panic!(
                        "planet_fixed_rotation_system (RotationModel::None \
                         clear): source {entity:?} has PfixFrameEntityC({:?}) \
                         but that entity has no FrameRotC ({err:?}). The \
                         pfix frame entity must be alive with FrameRotC / \
                         FrameAngVelC attached (spawned by \
                         register_pfix_frames_system). Either remove the \
                         stale PfixFrameEntityC marker before despawning \
                         the pfix frame entity, or ensure the pfix frame \
                         entity stays alive for as long as the source \
                         carries the handle.",
                        pfix_fe.0
                    )
                });
                *frame_rot = FrameRotC::default();
                let mut frame_av = frame_ang_vels.get_mut(pfix_fe.0).unwrap_or_else(|err| {
                    panic!(
                        "planet_fixed_rotation_system (RotationModel::None \
                         clear): source {entity:?} has PfixFrameEntityC({:?}) \
                         but that entity has no FrameAngVelC ({err:?}). The \
                         pfix frame entity must be alive with FrameRotC / \
                         FrameAngVelC attached (spawned by \
                         register_pfix_frames_system). Either remove the \
                         stale PfixFrameEntityC marker before despawning \
                         the pfix frame entity, or ensure the pfix frame \
                         entity stays alive for as long as the source \
                         carries the handle.",
                        pfix_fe.0
                    )
                });
                frame_av.0 = glam::DVec3::ZERO;

                // Retire the pfix frame entity for reuse on the next
                // toggle back to a rotating model. The orphan entity
                // stays alive (its `ChildOf(source_frame_entity)`
                // edge is preserved so the `RetiredPfixFrameEntityC`
                // reuse path in `register_pfix_frames_system`
                // doesn't have to re-parent), its `Name` is
                // overwritten with a stable `.retired` sentinel so
                // name-based lookups won't shadow a future live
                // entity, and the source's `PfixFrameEntityC` is
                // removed and replaced with `RetiredPfixFrameEntityC`.
                // This bounds the world's pfix-frame entity count at
                // one per source regardless of toggle-cycle count.
                commands
                    .entity(pfix_fe.0)
                    .insert(Name::new(format!("pfix.retired:{:?}", pfix_fe.0)));
                commands
                    .entity(entity)
                    .remove::<PfixFrameEntityC>()
                    .insert(RetiredPfixFrameEntityC(pfix_fe.0));
            }
        }
    }
}

/// Computes tidal ΔC20 for each gravity source that has a `TidalConfigC`.
///
/// Runs after `planet_fixed_rotation_system` so the rotation matrix is current.
/// Sources without `TidalConfigC` keep their default `TidalDeltaC20C::default()`
/// (a zero-valued [`jeod_sim::Ratio`]).
pub fn tidal_update_system<P: Planet>(
    mut query: Query<(&TidalConfigC, &PlanetFixedRotationC<P>, &mut TidalDeltaC20C)>,
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
/// Also updates `SourceInertialVelocityC` and `TranslationalStateC<P>` when
/// present (velocity for relativistic corrections; translational state for
/// Sun/Moon entities used by SRP, solar beta, and earth lighting systems).
///
/// Generic over `P: Planet` so the relabel from `RootInertial` → `PlanetInertial<P>`
/// matches the planet phantom on the `TranslationalStateC<P>` instance being
/// updated. Each plugin instantiation only matches sources whose
/// translational state carries the matching `<P>` tag — Sun/Moon ephemeris
/// bodies typically lack `TranslationalStateC` (so `Option<&mut ...>` is
/// `None` and the relabel is skipped) or carry a tag matching the planet
/// they orbit. See `register_planet_systems` for downstream multi-planet
/// instantiation.
///
/// Placed in `JeodSet::EphemerisUpdate`.
#[allow(clippy::type_complexity)]
pub fn ephemeris_update_system<P: Planet>(
    ephemeris: Option<Res<crate::EphemerisR>>,
    sim_time: Res<SimulationTimeR>,
    mut query: Query<(
        &EphemerisBodyC,
        &mut SourceInertialPositionC,
        Option<&mut SourceInertialVelocityC>,
        Option<&mut TranslationalStateC<P>>,
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
            // TranslationalStateC wraps `TranslationalStateTyped<PlanetInertial<P>>`;
            // `pos_typed` / `vel_typed` are root-inertial-tagged by the
            // ephemeris API. Relabel via `from_raw_si` to the
            // `<P>`-tagged planet-inertial frame the Component stores.
            // The numeric SI values (m, m/s) are preserved exactly —
            // only the phantom tag changes. The query filter guarantees
            // we only land in this branch when the matched body's
            // `<P>` matches the system instantiation, so the relabel is
            // sound.
            let pos_si = pos_typed.raw_si();
            let vel_si = vel_typed.raw_si();
            // allowed: ephemeris boundary, RootInertial → PlanetInertial<P> relabel
            ts.0.position = jeod_sim::Position::<jeod_sim::PlanetInertial<P>>::from_raw_si(pos_si);
            // allowed: same ephemeris boundary relabel
            ts.0.velocity = jeod_sim::Velocity::<jeod_sim::PlanetInertial<P>>::from_raw_si(vel_si);
        }
    }
}
