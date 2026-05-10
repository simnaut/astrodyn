//! Per-phase wall-clock instrumentation for [`super::super::Simulation`].
//!
//! Compiled only under the `phase_timing` cargo feature. The per-phase
//! `Instant::now()` boundaries live in [`super::super::step::step_internal`];
//! this module only owns the accumulator struct and its summary formatter.
//!
//! Bit-identity is preserved: timing calls touch wall-clock state only
//! and never participate in any f64 arithmetic that the
//! `bevy_parity_*` tests rely on. Even so, the feature is off by default
//! so the production builds pay zero overhead.

use std::time::Duration;

/// Wall-clock time spent in each per-step phase, accumulated across
/// every call to `Simulation::step()`.
///
/// One field per JEOD pipeline stage as labelled in
/// `step_internal`. Sum of all `Duration` fields plus
/// instrumentation overhead == total user time spent inside `step()`.
///
/// The `steps` counter records how many `step_internal` calls have
/// contributed, so the summary can render per-step averages alongside
/// totals.
#[derive(Debug, Default, Clone, Copy)]
pub struct PhaseTimings {
    /// Stage 1: `time.advance(dt)`.
    pub time_advance: Duration,
    /// Stage 2 + 2b: `update_ephemeris` (planet-fixed rotations,
    /// frame-tree sync, DE4xx source positions, tidal ΔC20).
    pub ephemeris: Duration,
    /// Stage 3: per-body `mass.recompute_derived()`.
    pub mass_recompute: Duration,
    /// Pre-integration `body_integ_origins` snapshot from the frame
    /// tree (the typed [`astrodyn::IntegOrigin`] shift used by the
    /// pre-integration kinematic walk).
    pub integ_origins_pre: Duration,
    /// Stage 3a + 3b: pre-integration `propagate_frame_attached_state`
    /// + `propagate_kinematic_state` walks.
    pub kinematic_pre: Duration,
    /// Stage 4 + 4b + 5: `update_environment` (gravity + atmosphere).
    pub environment: Duration,
    /// Stage 6: `compute_interactions` (drag, SRP, gravity-gradient
    /// torque).
    pub interactions: Duration,
    /// Stage 7 + 8 + 8b: `run_integration` (force collection, RK4
    /// integration, frame switching). Dominant phase on most workloads.
    pub integration: Duration,
    /// Post-integration `body_integ_origins_post` snapshot from the
    /// frame tree (recomputed because stage 8b's frame switch may have
    /// rewritten body integration frames).
    pub integ_origins_post: Duration,
    /// Stage 8c + 8d: post-integration
    /// `propagate_frame_attached_state` + `propagate_kinematic_state`
    /// walks.
    pub kinematic_post: Duration,
    /// Stage 9: `compute_derived_states` (orbital elements, Euler
    /// angles, LVLH, geodetic, solar beta, earth lighting).
    pub derived: Duration,
    /// Post-step ballistic propagation of detached subtrees. Zero on
    /// scenarios without staging.
    pub detached_subtrees: Duration,
    /// Number of `step_internal` calls accumulated. Used by
    /// [`PhaseTimings::summary`] to render per-step averages.
    pub steps: u64,
}

impl PhaseTimings {
    /// Sum of every per-phase `Duration` field. Approximates total
    /// time spent inside `step()`; the small remainder is
    /// instrumentation overhead and unmeasured glue.
    pub fn total(&self) -> Duration {
        self.time_advance
            + self.ephemeris
            + self.mass_recompute
            + self.integ_origins_pre
            + self.kinematic_pre
            + self.environment
            + self.interactions
            + self.integration
            + self.integ_origins_post
            + self.kinematic_post
            + self.derived
            + self.detached_subtrees
    }

    /// Render a multi-line summary suitable for logging at the end of
    /// a profiling run. Each line shows `phase: total (% of total) —
    /// per-step avg`.
    pub fn summary(&self) -> String {
        let total = self.total();
        let total_secs = total.as_secs_f64();
        let steps = self.steps.max(1) as f64;
        let row = |name: &str, d: Duration| {
            let secs = d.as_secs_f64();
            let pct = if total_secs > 0.0 {
                100.0 * secs / total_secs
            } else {
                0.0
            };
            let per_step_us = 1e6 * secs / steps;
            format!("  {name:<22} {secs:>9.3} s  {pct:>6.2}%   {per_step_us:>9.3} µs/step\n")
        };
        let mut out = String::new();
        out.push_str(&format!(
            "PhaseTimings: {} step(s), {:.3} s instrumented total\n",
            self.steps, total_secs
        ));
        out.push_str(&row("time_advance", self.time_advance));
        out.push_str(&row("ephemeris", self.ephemeris));
        out.push_str(&row("mass_recompute", self.mass_recompute));
        out.push_str(&row("integ_origins_pre", self.integ_origins_pre));
        out.push_str(&row("kinematic_pre", self.kinematic_pre));
        out.push_str(&row("environment", self.environment));
        out.push_str(&row("interactions", self.interactions));
        out.push_str(&row("integration", self.integration));
        out.push_str(&row("integ_origins_post", self.integ_origins_post));
        out.push_str(&row("kinematic_post", self.kinematic_post));
        out.push_str(&row("derived", self.derived));
        out.push_str(&row("detached_subtrees", self.detached_subtrees));
        out
    }
}
