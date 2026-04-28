//! Error type returned by stepping methods on [`Simulation`](crate::Simulation).
//!
//! [`Simulation::step`](crate::Simulation::step),
//! [`Simulation::step_n`](crate::Simulation::step_n), and
//! [`Simulation::step_until`](crate::Simulation::step_until) all return
//! `Result<(), StepError>`. Per #172 L2, mission-runtime conditions inside
//! `step_internal` (ephemeris lookup failure, frame-switch target out of
//! range) propagate as a typed `StepError` rather than aborting the process
//! via `panic!`. This lets mission programmes catch the failure (Monte
//! Carlo, batch propagation, resilient long-running services) without
//! resorting to `catch_unwind`.
//!
//! Programmer / configuration errors (bad source index passed to an
//! accessor method, typestate violations) remain `panic!`s — they signal a
//! bug in the calling code and recovering from them is not meaningful.
//!
//! # State semantics on `Err`
//!
//! When a stepping method returns `Err`, the simulation has been
//! **partially advanced** for the failing step. Specifically:
//!
//! - [`StepError::EphemerisLookup`] fires after time has advanced but
//!   before integration runs, so simulation time is ahead by `dt` while
//!   body states are still at the previous step.
//! - [`StepError::FrameSwitchTargetMissing`] fires after both time advance
//!   and integration, but before derived-state computation. Body states
//!   correspond to the new time but derived states (orbital elements,
//!   Euler angles, …) lag by one step.
//!
//! Either way the state is **not safe to step from again**. Mission code
//! that wants to recover should reconstruct the simulation from a saved
//! checkpoint (or from a fresh `SimulationBuilder`) rather than retry the
//! failing step.

use core::fmt;

use jeod_sim::EphemerisBody;

/// Reasons a [`Simulation`](crate::Simulation) step can fail at runtime.
///
/// Construct via the variant constructors (the kernel returns these from
/// `step_internal` on the relevant failure paths). Inspect via pattern
/// match. Implements [`core::error::Error`] so it composes with `?` and
/// the rest of the standard error-handling ecosystem.
///
/// # Reading the carried context
///
/// All variant fields are part of the public surface (an enum variant's
/// fields inherit the enum's visibility, so no per-field `pub` qualifier
/// is needed or accepted by the language). Downstream recovery logic can
/// destructure freely:
///
/// ```ignore
/// use jeod_runner::StepError;
///
/// fn handle(err: StepError) {
///     match err {
///         StepError::EphemerisLookup {
///             source_idx,
///             tdb_jd,
///             ..
///         } => {
///             eprintln!("source {source_idx} missing at TDB JD {tdb_jd}");
///         }
///         StepError::FrameSwitchTargetMissing {
///             body_idx,
///             target_source,
///             num_sources,
///         } => {
///             eprintln!(
///                 "body {body_idx}: target {target_source} >= {num_sources}"
///             );
///         }
///     }
/// }
/// ```
#[derive(Debug, Clone)]
pub enum StepError {
    /// DE4xx ephemeris lookup failed for a source whose position is
    /// driven by an attached ephemeris. Carries enough context to
    /// identify the source and the time scale at which the lookup failed.
    EphemerisLookup {
        /// Index of the source whose position update failed.
        source_idx: usize,
        /// Target body passed to `Ephemeris::get_state_typed`.
        target: EphemerisBody,
        /// Observer body passed to `Ephemeris::get_state_typed`.
        observer: EphemerisBody,
        /// TDB Julian Date at which the lookup was attempted.
        tdb_jd: f64,
        /// Underlying error message from the ephemeris kernel.
        message: String,
    },
    /// A body's [`FrameSwitchConfig`](jeod_sim::FrameSwitchConfig)
    /// referenced a `target_source` index that doesn't correspond to any
    /// registered gravity source. Typically a mid-mission edit hazard
    /// caught at step time when [`Simulation::validate`](crate::Simulation::validate)
    /// wasn't re-run after the change.
    FrameSwitchTargetMissing {
        /// Index of the body whose frame switch contains the bad target.
        body_idx: usize,
        /// Out-of-range target source index.
        target_source: usize,
        /// Number of sources currently registered.
        num_sources: usize,
    },
}

impl fmt::Display for StepError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EphemerisLookup {
                source_idx,
                target,
                observer,
                tdb_jd,
                message,
            } => write!(
                f,
                "Ephemeris lookup failed for source {source_idx} \
                 ({target:?} wrt {observer:?}) at TDB JD {tdb_jd}: {message}"
            ),
            Self::FrameSwitchTargetMissing {
                body_idx,
                target_source,
                num_sources,
            } => write!(
                f,
                "frame switch evaluation: body {body_idx} references target_source \
                 {target_source} but only {num_sources} source(s) are configured. \
                 Re-run Simulation::validate() before stepping; if a frame switch \
                 was added or edited mid-mission, also check the body's \
                 frame_switches list."
            ),
        }
    }
}

impl core::error::Error for StepError {}
