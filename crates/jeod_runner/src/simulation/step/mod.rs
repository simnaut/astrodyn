//! Per-step integration pipeline for [`super::Simulation`].
//!
//! `step_internal` orchestrates JEOD's 9 numbered integration stages
//! (issue #253 Task B) by delegating each to a dedicated sibling file:
//!
//! - [`ephemeris`] — stages 2 + 2b: planet-fixed rotations, frame-tree
//!   sync, DE4xx source positions, tidal ΔC20.
//! - [`environment`] — stages 4 + 4b + 5: gravity (Newtonian +
//!   post-Newtonian PPN) and atmosphere.
//! - [`interactions`] — stage 6: drag, SRP (flat-plate or cannonball),
//!   gravity-gradient torque.
//! - [`integrate`] — stages 7 + 8 + 8b: force collection, RK4
//!   integration with contact-pair coupling, frame-switch handling.
//!   Holds the bulk of the per-step work (~580 lines).
//! - [`derived`] — stage 9: orbital elements, Euler angles, LVLH,
//!   geodetic state, solar beta, earth lighting.
//!
//! Stages 1 (time advance) and 3 (mass-property recompute) stay inline
//! in `step_internal` because each is < 5 lines. The detached-subtree
//! ballistic propagation at the end of the step also lives here as
//! it's not really "stage 9" but a post-step concern.
//!
//! `step`, `step_n`, `step_until`, and `frame_origin` are the public
//! entry points; they all live in this `mod.rs`.

use glam::DVec3;

use jeod_frames::FrameId;
use jeod_sim::{IntegOrigin, Position, Velocity};

use super::Simulation;
use crate::error::StepError;

mod derived;
mod environment;
mod ephemeris;
mod integrate;
mod interactions;
mod kinematic;

impl Simulation {
    /// Advance the simulation by one timestep.
    ///
    /// Runs the full JEOD pipeline in order:
    /// 1. Time update
    /// 2. Ephemeris update (planet-fixed rotations + frame tree sync)
    /// 3. Mass update (recompute derived quantities)
    /// 4. Gravity computation
    /// 5. Atmosphere evaluation
    /// 6. Interaction computation (drag, SRP, gravity torque)
    /// 7. Force collection and frame derivative computation
    /// 8. State integration (RK4, with sub-stage tree updates)
    /// 9. Derived state computation
    ///
    /// On `Err`, the simulation is **not** recoverable: time has already
    /// advanced and downstream state may be partially updated. To retry,
    /// construct a fresh [`Simulation`].
    pub fn step(&mut self) -> Result<(), StepError> {
        self.step_internal(self.dt)
    }

    /// Get the position and velocity of a frame relative to the root inertial frame.
    pub fn frame_origin(&self, frame_id: FrameId) -> (DVec3, DVec3) {
        if frame_id == self.root_frame_id {
            return (DVec3::ZERO, DVec3::ZERO);
        }
        let state = self
            .frame_tree
            .compute_relative_state(self.root_frame_id, frame_id);
        (state.trans.position, state.trans.velocity)
    }

    /// Internal step with explicit dt (avoids temporary mutation of `self.dt`
    /// in `step_until`).
    fn step_internal(&mut self, dt: f64) -> Result<(), StepError> {
        // Flip before any irreversible mutation. `time.advance(dt)` runs
        // unconditionally below, so once we enter `step_internal` the
        // simulation is no longer in its t=0 registration window —
        // independent of whether downstream fallible work succeeds.
        // Setting this only on success would let a registration call
        // between a failed `step` and the next attempt compute an
        // init-phase impulse against state whose time has already
        // moved. See `Simulation::has_stepped`.
        self.has_stepped = true;

        // ── 1. Time update ──
        self.time.advance(dt);

        // ── 2. Ephemeris update — planet-fixed rotations + frame tree sync ──
        self.update_ephemeris()?;
        // ── 3. Mass update — recompute inverse_mass/inverse_inertia ──
        for body in &mut self.bodies {
            if let Some(ref mut mass) = body.mass {
                mass.recompute_derived();
            }
        }

        // ── 3b. Kinematic state propagation (root → leaves), pre-integration ──
        // Mirrors the Bevy adapter's `propagate_state_from_root_system`
        // schedule placement — `JeodSet::ForceCollection`, after mass
        // recompute and before any consumer that reads child attitudes
        // (the wrench walk in Bevy; defense-in-depth here, since the
        // runner's `collect_and_resolve_forces` is per-body without an
        // upward chain walk yet). The "before integration" placement
        // is one tick stale by construction — every kinematic child's
        // state reflects the *previous* tick's integrated parent. The
        // post-integration sibling call below cleans the end-of-step
        // state up so consumers reading `Simulation::body(idx)` after
        // a `step()` see the freshly-derived value.
        //
        // See `step::kinematic` for the JEOD precedent and the per-call
        // diagnostic invariants.
        self.propagate_kinematic_state();

        // Precompute frame origins from the tree for all body integration
        // frames. The typed `IntegOrigin` is the only safe path from
        // `Position<IntegrationFrame>` to `Position<RootInertial>` — see #255 /
        // RF.10. Returns `IntegOrigin::zero()` when the body integrates in
        // the root frame (the bit-identical no-shift case).
        let body_integ_origins: Vec<IntegOrigin> = self
            .bodies
            .iter()
            .map(|b| {
                let (p, v) = self.frame_origin(b.integ_frame_id);
                IntegOrigin {
                    position: Position::from_raw_si(p),
                    velocity: Velocity::from_raw_si(v),
                }
            })
            .collect();

        self.update_environment(&body_integ_origins);

        // ── 6. Interactions — drag, SRP, gravity torque ──
        let (sun_pos, moon_pos) = self.compute_interactions(dt, &body_integ_origins);

        self.run_integration(dt, &body_integ_origins)?;

        // ── 8c. Kinematic state propagation (root → leaves), post-integration ──
        // The integrator just produced fresh root-body state; rerun
        // the kinematic walk so every non-root child's `body.trans` /
        // `body.rot` reflects the same-tick parent state. Mirrors
        // JEOD's `DynBody::propagate_state_from_structure` invocation
        // at the end of every integration cycle
        // (`models/dynamics/dyn_body/src/dyn_body_propagate_state.cc`).
        // Without this, downstream consumers (`Simulation::body`,
        // derived-state computations, frame-tree sync) would observe
        // a one-tick-stale child state — i.e. the previous tick's
        // parent state composed with the link, not the freshly-
        // integrated one.
        self.propagate_kinematic_state();

        // ── 9. Derived states ──
        self.compute_derived_states(sun_pos, moon_pos, &body_integ_origins);

        // Advance any free-flying detached subtrees ballistically. This
        // matches JEOD's behavior for tree roots whose grav_interaction
        // is empty (the common case for staging — no force applied to
        // separated stages between detach and reattach). Use the dynamic
        // timestep `dt * time_scale_factor` so the ballistic propagation
        // stays consistent with the integrated bodies under time reversal
        // / scaling (matches `integ_dt` used elsewhere in this step).
        if !self.detached_subtrees.is_empty() {
            self.step_detached_subtrees(dt * self.time.time_scale_factor);
        }

        Ok(())
    }

    /// Advance the simulation by `n` timesteps.
    pub fn step_n(&mut self, n: usize) -> Result<(), StepError> {
        for _ in 0..n {
            self.step()?;
        }
        Ok(())
    }

    /// Advance the simulation until `target_time` (in simulation seconds).
    ///
    /// Steps at `self.dt` until the remaining time is less than `dt`,
    /// then takes a final fractional step if the remainder exceeds 1 ms.
    pub fn step_until(&mut self, target_time: f64) -> Result<(), StepError> {
        while self.time.simtime + self.dt <= target_time + 0.001 {
            self.step()?;
        }
        let remainder = target_time - self.time.simtime;
        if remainder > 0.001 {
            // Fractional steps corrupt multi-step history arrays (GJ's
            // Störmer-Cowell coefficients and delinv accumulators, ABM4's
            // Adams history) — both methods assume constant dt.
            let has_multistep = self.bodies.iter().any(|b| {
                matches!(
                    b.integrator,
                    jeod_dynamics::IntegratorType::GaussJackson(..)
                        | jeod_dynamics::IntegratorType::Abm4
                )
            });
            assert!(
                !has_multistep,
                "step_until() would take a fractional step ({remainder:.6}s vs dt={:.6}s). \
                 Multi-step integrators (GaussJackson, ABM4) require constant dt. \
                 Ensure target_time is an integer multiple of dt.",
                self.dt
            );
            self.step_internal(remainder)?;
        }
        Ok(())
    }
}
