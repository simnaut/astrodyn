//! Bevy systems for [`AstrodynSet::EphemerisUpdate`](crate::AstrodynSet::EphemerisUpdate).
//!
//! Planet-fixed rotation (RNP), DE4xx ephemeris updates, and tidal ΔC20.

use astrodyn::Planet;
use bevy::prelude::*;

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
    mut frame_epochs: Query<&mut FrameEpochC>,
) {
    let polar_params = polar.map(|p| (p.xp, p.yp));
    // Frame-epoch stamp for every pfix frame this system rotates
    // (RFS-603) — mirrors the runner's `update_ephemeris` stamping the
    // pfix node each step a rotation model runs; `RotationModel::None`
    // sources keep their registration-time epoch, matching the runner's
    // heterogeneous epochs for cross-backend document equality.
    let frame_epoch = FrameEpochC(sim_time.tdb());
    // Lazy-compute Earth RNP once per system invocation when an
    // `EarthRNP` rotation-model entity is matched. Cache the
    // already-typed `FrameTransform` rather than the bare matrix so the
    // expensive `from_matrix` work (matrix→quat extraction + renormalization)
    // happens once per tick total, not once per EarthRNP entity per tick —
    // all matched entities share the same rotation each step.
    type PlanetRot<P> = astrodyn::FrameTransform<astrodyn::RootInertial, astrodyn::PlanetFixed<P>>;
    let mut earth_rotation: Option<PlanetRot<P>> = Option::None;
    let mut earth_rotation_raw: Option<glam::DMat3> = Option::None;
    // Lazy-build the ANISE `Epoch` for the current TDB instant once.
    // The MoonDE421 branch below is the only consumer; hoisting the
    // `Epoch::from_tdb_seconds` + `hifitime::Epoch::to_time_scale` cost
    // out of the per-entity match keeps multi-Moon scenarios honest and
    // mirrors the runner's `update_ephemeris` epoch-cache pattern.
    let mut tdb_epoch: Option<astrodyn::Epoch> = Option::None;
    for (entity, mut rot, model, omega, ang_vel, pfix_frame_entity) in &mut query {
        let default_model = astrodyn::RotationModel::EarthRNP;
        let rotation_model = model.map_or(&default_model, |m| &m.0);
        // Track whether we wrote a rotation this tick — controls
        // `PlanetAngularVelocityC` and pfix frame-entity writes.
        let rotated = !matches!(rotation_model, astrodyn::RotationModel::None);
        // Capture the raw DMat3 too so we can sync the pfix frame
        // entity (FrameRotC + FrameAngVelC) from the same data.
        let mut raw_matrix: Option<glam::DMat3> = None;
        match rotation_model {
            astrodyn::RotationModel::None => {}
            astrodyn::RotationModel::EarthRNP => {
                let mat = *earth_rotation_raw.get_or_insert_with(|| {
                    astrodyn::compute_t_parent_this_from_tjt_with_polar(
                        sim_time.gmst_seconds,
                        sim_time.tt_tjt(),
                        polar_params,
                    )
                });
                let rotation = *earth_rotation.get_or_insert_with(|| {
                    astrodyn::compute_t_parent_this_from_tjt_with_polar_typed::<P>(
                        sim_time.gmst_seconds,
                        sim_time.tt_tjt(),
                        polar_params,
                    )
                });
                rot.0 = rotation;
                raw_matrix = Some(mat);
            }
            astrodyn::RotationModel::MarsIAU => {
                let tt_s_since_j2000 =
                    (sim_time.tt_tjt() - astrodyn::J2000_TT_TJT) * astrodyn::SECONDS_PER_DAY;
                let mat = astrodyn::rotation_mars::compute_mars_rotation(tt_s_since_j2000);
                // allowed: matrix from JEOD-ported IAU Mars rotation formula; the
                // typed sibling pins `<PlanetFixed<Mars>>`, but the system's
                // storage is `<PlanetFixed<P>>` (P resolved at registration);
                // the lift here is the legitimate cross-P boundary.
                rot.0 = astrodyn::FrameTransform::from_matrix(mat);
                raw_matrix = Some(mat);
            }
            astrodyn::RotationModel::MoonIAU => {
                let tdb_jd = sim_time.tdb_julian_date();
                let tdb_s_since_j2000 =
                    (tdb_jd - astrodyn::J2000_TT_JD) * astrodyn::SECONDS_PER_DAY;
                let mat = astrodyn::rotation_moon::compute_moon_rotation(tdb_s_since_j2000);
                // allowed: same cross-P kernel-output boundary as the Mars branch above.
                rot.0 = astrodyn::FrameTransform::from_matrix(mat);
                raw_matrix = Some(mat);
            }
            astrodyn::RotationModel::MoonDE421 => {
                let eph = ephemeris.as_ref().expect(
                    "RotationModel::MoonDE421 requires the EphemerisR resource with a BPC \
                     loaded. Insert EphemerisR before stepping the simulation, or switch the \
                     body to RotationModel::MoonIAU.",
                );
                let tdb_jd = sim_time.tdb_julian_date();
                let epoch =
                    *tdb_epoch.get_or_insert_with(|| astrodyn::Ephemeris::tdb_jd_to_epoch(tdb_jd));
                let mat = eph
                    .get_body_rotation_epoch(astrodyn::EphemerisBody::Moon, epoch)
                    .unwrap_or_else(|err| {
                        panic!(
                            "Moon DE421 BPC rotation query failed at TDB JD {tdb_jd}: {err:?}. \
                             The loaded BPC kernel does not cover this epoch; load a kernel \
                             whose coverage includes the simulation epoch."
                        )
                    });
                // allowed: matrix from NASA SPICE BPC kernel (DE421 / Moon-PA)
                rot.0 = astrodyn::FrameTransform::from_matrix(mat);
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
                astrodyn::RotationModel::None => 0.0,
                astrodyn::RotationModel::EarthRNP => astrodyn::EARTH.omega,
                astrodyn::RotationModel::MarsIAU => astrodyn::MARS.omega,
                astrodyn::RotationModel::MoonIAU | astrodyn::RotationModel::MoonDE421 => {
                    astrodyn::MOON.omega
                }
            };
            let omega_value = omega.map(|o| o.0).unwrap_or(default_omega);
            if let Some(mut ang_vel_c) = ang_vel {
                // Mint `AngularVelocity<PlanetFixed<P>>` from the
                // scalar `PlanetOmegaC`. JEOD's `planet_rnp.cc` writes
                // [0, 0, omega] in the pfix frame; this is the typed-API
                // boundary for that scalar → typed-vector lift.
                let raw = glam::DVec3::new(0.0, 0.0, omega_value);
                let typed = astrodyn::AngularVelocity::<astrodyn::PlanetFixed<P>>::from_raw_si(raw); // allowed: scalar omega → typed AngularVelocity boundary
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
                    astrodyn::JeodQuat::left_quat_from_transformation(&matrix);
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
                let mut epoch = frame_epochs.get_mut(pfix_fe.0).unwrap_or_else(|err| {
                    panic!(
                        "planet_fixed_rotation_system: source {entity:?} has \
                         PfixFrameEntityC({:?}) but that entity has no \
                         FrameEpochC ({err:?}). Pfix frame entities are \
                         stamped with FrameEpochC at registration (issue \
                         #664); do not strip it.",
                        pfix_fe.0
                    )
                });
                *epoch = frame_epoch;
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
            rot.0 = astrodyn::FrameTransform::from_matrix(glam::DMat3::IDENTITY);
            if let Some(mut ang_vel_c) = ang_vel {
                // allowed: zero-omega clear → typed AngularVelocity boundary
                ang_vel_c.0 = astrodyn::AngularVelocity::<astrodyn::PlanetFixed<P>>::from_raw_si(
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
                    .insert(Name::new(format!("pfix.retired:{:?}", pfix_fe.0)))
                    // A retired pfix frame is no longer a live tree node:
                    // drop its identity (freeing the uid in
                    // `FrameUidIndexR` via the deindex system) and its
                    // epoch — the reuse path re-stamps both. Leaving the
                    // identity would make the retired orphan exportable
                    // as a phantom frame and collide with the re-minted
                    // identity on the next toggle-back (issue #664).
                    .remove::<FrameUidC>()
                    .remove::<FrameEpochC>();
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
/// (a zero-valued [`astrodyn::Ratio`]).
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
        delta.0 = astrodyn::compute_delta_c20_typed(&config.0, rotation.0.matrix_ref());
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
/// Placed in `AstrodynSet::EphemerisUpdate`.
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
    // Build the ANISE `Epoch` once per system run and reuse across all
    // matched entities — every query in the loop evaluates at the same
    // instant, so the `Epoch::from_tdb_seconds` +
    // `hifitime::Epoch::to_time_scale` work only needs to happen once.
    // Mirrors the runner's `update_ephemeris` per-step epoch cache.
    let epoch = astrodyn::Ephemeris::tdb_jd_to_epoch(tdb_jd);
    for (ephem_body, mut source_pos, source_vel, trans_state) in &mut query {
        // Typed sibling: returns `(Position<RootInertial>, Velocity<RootInertial>)`
        // directly, matching the typed component storage. Bit-identical to
        // the deprecated f64 path — the kernel itself extracts SI base
        // values from ANISE and re-wraps them.
        let (pos_typed, vel_typed) = eph
            .get_state_typed_epoch(ephem_body.target, ephem_body.observer, epoch)
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
            ts.0.position = astrodyn::Position::<astrodyn::PlanetInertial<P>>::from_raw_si(pos_si);
            // allowed: same ephemeris boundary relabel
            ts.0.velocity = astrodyn::Velocity::<astrodyn::PlanetInertial<P>>::from_raw_si(vel_si);
        }
    }
}
