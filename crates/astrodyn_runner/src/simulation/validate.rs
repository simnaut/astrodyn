//! Pre-step setup-time validation for [`super::Simulation`].
//!
//! Lifted out of `mod.rs` because the single `validate()` method runs
//! ~325 lines and is a distinct concern from the per-step pipeline.

use astrodyn::validation::ValidationError;

use super::Simulation;

impl Simulation {
    /// Validate all bodies against JEOD invariants and apply auto-corrections.
    ///
    /// Call once before the first `step()`. Returns `Ok(())` if all bodies are
    /// valid, or `Err(errors)` with all validation errors found.
    ///
    /// Also runs `GravityControl::check_validity()` on each control to
    /// auto-correct degree/order (matching JEOD's `initialize_gravity_controls()`
    /// and the Bevy adapter's startup validation).
    // JEOD_INV: GV.03 — check_validity() called at startup (auto-corrections applied in-place)
    pub fn validate(&mut self) -> Result<(), Vec<ValidationError>> {
        let num_sources = self.gravity_data.len();
        let mut all_errors = Vec::new();
        for (body_idx, body) in self.bodies.iter_mut().enumerate() {
            let plate_counts = body.flat_plate_state.as_ref().map(|fps| {
                (
                    fps.plates.len(),
                    fps.temperatures.len(),
                    fps.t_pow4_cached.len(),
                )
            });
            let trans_untyped = body.trans.to_untyped();
            let errors = astrodyn::validate_body(
                &body.config,
                &body.gravity_controls,
                true, // SimBody always has gravity_accel field
                body.mass.as_ref(),
                body.rot.is_some(),
                Some(&trans_untyped),
                |source_id: usize| self.gravity_data.get(source_id).map(|g| &g.source),
                plate_counts,
            );
            all_errors.extend(errors);

            // Validate shadow_body index
            if let Some((idx, _radius)) = body.shadow_body {
                if idx >= num_sources {
                    all_errors.push(ValidationError::ShadowBodyOutOfRange {
                        index: idx,
                        num_sources,
                    });
                }
            }

            // Validate geodetic_planet index
            if let Some((idx, _, _)) = body.geodetic_planet {
                if idx >= num_sources {
                    all_errors.push(ValidationError::GeodeticPlanetOutOfRange {
                        index: idx,
                        num_sources,
                    });
                }
            }

            // Validate orbital_elements_source index
            if let Some(idx) = body.orbital_elements_source {
                if idx >= num_sources {
                    all_errors.push(ValidationError::OrbitalElementsSourceOutOfRange {
                        index: idx,
                        num_sources,
                    });
                }
            }

            // Validate force producers have mass (JEOD_INV: MA.01 — MassBody always present)
            if (body.drag.is_some() || body.flat_plate_state.is_some()) && body.mass.is_none() {
                all_errors.push(ValidationError::ForceProducerWithoutMass { body_idx });
            }

            // GaussJackson is translational-only (6-DOF not yet supported)
            if matches!(body.integrator, astrodyn::IntegratorType::GaussJackson(..))
                && body.config.rotational_dynamics
            {
                all_errors.push(ValidationError::GaussJacksonWith6Dof { body_idx });
            }

            // ABM4 is translational-only (6-DOF not yet supported)
            if matches!(body.integrator, astrodyn::IntegratorType::Abm4)
                && body.config.rotational_dynamics
            {
                all_errors.push(ValidationError::Abm4With6Dof { body_idx });
            }

            // GaussJackson config validation — delegates to GaussJacksonConfig::check()
            // so the predicate is defined in one place.
            if let astrodyn::IntegratorType::GaussJackson(ref config) = body.integrator {
                for detail in config.check() {
                    all_errors
                        .push(ValidationError::GaussJacksonConfigInvalid { body_idx, detail });
                }
            }

            // Drag requires atmosphere config on the simulation. (After
            // the typed-storage migration the per-body atmospheric
            // state is unconditionally present; drag presence is the
            // discriminator that decides whether the atmosphere stage
            // fills it, so the validation key is now `body.drag`.)
            if body.drag.is_some() && self.atmosphere.is_none() {
                all_errors.push(ValidationError::AtmosphericStateWithoutAtmosphere { body_idx });
            }

            // Solar beta requires sun_source on the simulation
            if body.compute_solar_beta && self.sun_source.is_none() {
                all_errors.push(ValidationError::SolarBetaWithoutSunSource { body_idx });
            }

            // Earth lighting requires both sun_source and moon_source
            if body.earth_lighting_config.is_some() {
                if self.sun_source.is_none() {
                    all_errors.push(ValidationError::EarthLightingWithoutSunSource { body_idx });
                }
                if self.moon_source.is_none() {
                    all_errors.push(ValidationError::EarthLightingWithoutMoonSource { body_idx });
                }
            }

            // Gravity torque requires both mass and rotational state
            if body.compute_gravity_torque && (body.mass.is_none() || body.rot.is_none()) {
                all_errors.push(ValidationError::GravityTorqueWithoutMassOrRot { body_idx });
            }

            // Frame switch target_source must be a valid source index AND
            // present in the body's gravity controls (so the post-switch
            // differential flip actually takes effect).
            // Only validate active switches — JEOD only evaluates active switches.
            for sw in &body.frame_switches {
                if sw.active {
                    let target = sw.target_source;
                    if target >= num_sources {
                        all_errors.push(ValidationError::FrameSwitchTargetSourceOutOfRange {
                            body_idx,
                            target_source: target,
                            num_sources,
                        });
                    } else if !body
                        .gravity_controls
                        .controls
                        .iter()
                        .any(|c| c.source_name == target)
                    {
                        all_errors.push(ValidationError::FrameSwitchTargetSourceNotInControls {
                            body_idx,
                            target_source: target,
                        });
                    }
                }
            }

            // Warn when body uses a non-root integration frame with features
            // that assume root-inertial coordinates. JEOD evaluates these
            // derived states in the central-body inertial frame; they will
            // produce incorrect results in other frames.
            {
                let non_root_integ = body.integ_frame_id != self.root_frame_id;
                let non_root_switch = body.frame_switches.iter().any(|sw| {
                    sw.active
                        && self
                            .source_frame_ids
                            .get(sw.target_source)
                            .is_some_and(|frame| frame.inertial != self.root_frame_id)
                });
                if non_root_integ || non_root_switch {
                    let has_root_dependent_feature = body.drag.is_some()
                        || body.flat_plate_state.is_some()
                        || body.cannonball_srp.is_some()
                        || body.orbital_elements_source.is_some()
                        || body.euler_sequence.is_some()
                        || body.compute_lvlh
                        || body.geodetic_planet.is_some()
                        || body.compute_solar_beta
                        || body.earth_lighting_config.is_some();
                    if has_root_dependent_feature {
                        all_errors.push(ValidationError::NonRootFrameWithRootDependentFeatures {
                            body_idx,
                        });
                    }
                }
            }

            // Apply gravity control auto-corrections (degree/order clamping).
            // JEOD_INV: GV.03 — check_validity() auto-corrects out-of-range settings
            for ctrl in &mut body.gravity_controls.controls {
                if let Some(grav) = self.gravity_data.get(ctrl.source_name) {
                    ctrl.check_validity(&grav.source);
                }
            }
        }

        // Validate sun_source index (simulation-level, outside body loop)
        if let Some(idx) = self.sun_source {
            if idx >= num_sources {
                all_errors.push(ValidationError::SunSourceOutOfRange {
                    index: idx,
                    num_sources,
                });
            }
        }

        // Validate moon_source index
        if let Some(idx) = self.moon_source {
            if idx >= num_sources {
                all_errors.push(ValidationError::MoonSourceOutOfRange {
                    index: idx,
                    num_sources,
                });
            }
        }

        // Validate atmosphere_planet_source index
        if let Some(idx) = self.atmosphere_planet_source {
            if idx >= num_sources {
                all_errors.push(ValidationError::AtmospherePlanetOutOfRange {
                    index: idx,
                    num_sources,
                });
            }
        }

        // Ephemeris mapping on root-frame sources — would silently discard position.
        for (i, ephem) in self.source_ephem_bodies.iter().enumerate() {
            if ephem.is_some() && self.source_frame_ids[i].inertial == self.root_frame_id {
                all_errors.push(ValidationError::EphemerisOnRootSource { source_idx: i });
            }
        }

        // Contact-coupled path (inter-body or ground-contact pairs)
        // requires the multi-body coupled RK4 kernel, which drives every
        // body through the same code path. Enforce RK4 + 6-DOF on all
        // bodies (not just those that appear in a pair) so the field
        // docs' "validation error" promise matches reality. Hoist the
        // shared per-body checks out of the inter-body / ground-contact
        // branches so each body emits at most one
        // ContactPairsRequire{Rk4,6Dof} error even when both pair types
        // are registered.
        let any_contact_pairs =
            !self.contact_pairs.is_empty() || !self.ground_contact_pairs.is_empty();
        if any_contact_pairs {
            for (body_idx, body) in self.bodies.iter().enumerate() {
                if !matches!(body.integrator, astrodyn::IntegratorType::Rk4) {
                    all_errors.push(ValidationError::ContactPairsRequireRk4 { body_idx });
                }
                if body.rot.is_none() || body.mass.is_none() {
                    all_errors.push(ValidationError::ContactPairsRequire6Dof { body_idx });
                }
            }
        }

        if !self.contact_pairs.is_empty() {
            // The coupled RK4 contact path evaluates `stage_trans`/`stage_rot`
            // directly, and those stage states are in each body's integration
            // frame. For pair-level consistency (and to match the no-transform
            // convention in `evaluate_contact_pair`), both bodies must share
            // the same integration frame, and that frame must be the root
            // inertial frame.
            for (pair_idx, pair) in self.contact_pairs.iter().enumerate() {
                let frame_a = self.bodies[pair.body_a].integ_frame_id;
                let frame_b = self.bodies[pair.body_b].integ_frame_id;
                if frame_a != frame_b {
                    all_errors.push(ValidationError::ContactPairFrameMismatch {
                        pair_idx,
                        body_a: pair.body_a,
                        body_b: pair.body_b,
                        frame_a,
                        frame_b,
                    });
                    continue;
                }
                if frame_a != self.root_frame_id {
                    // frame_a == frame_b at this point, so both bodies
                    // share the same non-root frame. Emit one error per
                    // body so debuggers see the full set.
                    all_errors.push(ValidationError::ContactPairNonRootFrame {
                        pair_idx,
                        body_idx: pair.body_a,
                        frame: frame_a,
                        root: self.root_frame_id,
                    });
                    all_errors.push(ValidationError::ContactPairNonRootFrame {
                        pair_idx,
                        body_idx: pair.body_b,
                        frame: frame_a,
                        root: self.root_frame_id,
                    });
                }
            }
        }

        // Ground-contact pairs: per-pair frame and central-planet
        // checks. Per-body RK4 + 6-DOF is hoisted above (shared with
        // inter-body contact).
        if !self.ground_contact_pairs.is_empty() {
            for (pair_idx, pair) in self.ground_contact_pairs.iter().enumerate() {
                let frame = self.bodies[pair.body_a].integ_frame_id;
                if frame != self.root_frame_id {
                    all_errors.push(ValidationError::GroundContactPairNonRootFrame {
                        pair_idx,
                        body_idx: pair.body_a,
                        frame,
                        root: self.root_frame_id,
                    });
                }
                if let Some(planet_source) = self.ground_contact_planet_source {
                    let sfids = &self.source_frame_ids[planet_source];
                    if sfids.inertial != self.root_frame_id {
                        all_errors.push(ValidationError::GroundContactNonCentralPlanet {
                            pair_idx,
                            planet_source,
                        });
                    }
                }
            }
        }

        // Separate warnings from fatal errors — warnings are logged, not returned.
        let mut fatal = Vec::new();
        for error in all_errors {
            if error.is_warning() {
                log::warn!("{error}");
            } else {
                fatal.push(error);
            }
        }
        if !fatal.is_empty() {
            return Err(fatal);
        }

        // Auto-initialize Gauss-Jackson state for bodies that need it.
        // Check config consistency for pre-existing states.
        for (body_idx, body) in self.bodies.iter_mut().enumerate() {
            if let astrodyn::IntegratorType::GaussJackson(ref config) = body.integrator {
                match &body.gj_state {
                    None => {
                        body.gj_state = Some(astrodyn::GaussJacksonState::new(*config));
                    }
                    Some(state) if state.config() != *config => {
                        fatal.push(ValidationError::GaussJacksonConfigInvalid {
                            body_idx,
                            detail: format!(
                                "existing gj_state config {:?} does not match \
                                 IntegratorType config {:?}. \
                                 Remove gj_state or recreate from the same config.",
                                state.config(),
                                config,
                            ),
                        });
                    }
                    Some(_) => {} // config matches, keep existing state
                }
            }
            // Auto-initialize ABM4 state for bodies that need it.
            if matches!(body.integrator, astrodyn::IntegratorType::Abm4)
                && body.abm4_state.is_none()
            {
                body.abm4_state = Some(astrodyn::Abm4State::new());
            }
        }
        if !fatal.is_empty() {
            return Err(fatal);
        }

        Ok(())
    }
}
