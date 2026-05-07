//! `IntegrableObject` trait — port of JEOD's multi-ODE RK4 scheduling.
//!
//! JEOD's `DynamicsIntegrationGroup` (dynamics/dyn_manager/src/dynamics_integration_group.cc)
//! drives a heterogeneous collection of `IntegrableObject` instances through
//! the RK4 stage loop. Each `IntegrableObject` exposes a per-stage protocol
//! (snapshot at step start → advance to intermediate → combine derivatives at
//! step end) that the group coordinates so multiple ODEs (orbital state,
//! thermal state, and future sub-states) integrate in lockstep against the
//! same stage boundaries.
//!
//! This module defines the Rust equivalent trait. Today the orbital state
//! is integrated directly by the enclosing RK4 kernel (translational and
//! rotational state are not trait implementors), and the only trait
//! implementor is [`crate::FlatPlateState`] for thermal plate temperatures.
//! The trait exists to pin down the per-stage protocol as a documentation
//! and extension point: additional sub-states (e.g. fuel slosh, battery
//! thermal) become drop-in implementors.
//!
//! # Per-step protocol
//!
//! For a single RK4 step with dynamic timestep `dt`, the enclosing kernel:
//!
//! 1. Calls [`IntegrableObject::snapshot`] once at the top of the step,
//!    which saves the current state as the step-start base.
//! 2. Evaluates stage 1 derivatives against the step-start state.
//! 3. Calls [`IntegrableObject::advance_intermediate`] with the stage-1
//!    derivative and `h = dt/2` to set state = snapshot + k1·dt/2.
//! 4. Evaluates stage 2 derivatives.
//! 5. Calls `advance_intermediate` with the stage-2 derivative and `h = dt/2`.
//! 6. Evaluates stage 3 derivatives.
//! 7. Calls `advance_intermediate` with the stage-3 derivative and `h = dt`.
//! 8. Evaluates stage 4 derivatives.
//! 9. Calls [`IntegrableObject::finalize_rk4`] with all four derivatives
//!    and `dt`, which produces the final step result (implementors may
//!    apply clamping or constraint enforcement at this step).
//!
//! State is represented as a flat `&[f64]` slice of length
//! [`IntegrableObject::n_states`]. Derivative slices passed to
//! `advance_intermediate` and `finalize_rk4` share that length.

/// An integrable sub-state that advances in lockstep with the vehicle's
/// orbital state through an RK4 stage loop.
///
/// See module docs for the per-step snapshot → advance → finalize protocol.
pub trait IntegrableObject {
    /// Number of scalar state variables the object owns. Derivative slices
    /// passed to [`Self::advance_intermediate`] and [`Self::finalize_rk4`]
    /// must have this length.
    fn n_states(&self) -> usize;

    /// Save the current state as the step-start snapshot. Called once at
    /// the top of each RK4 step, before stage 1's derivative evaluation.
    fn snapshot(&mut self);

    /// Advance state to `snapshot + deriv * h` in place.
    ///
    /// Called between stages using the previous stage's derivative. `h` is
    /// the intermediate time delta (e.g. `dt/2` before stages 2 and 3,
    /// full `dt` before stage 4).
    fn advance_intermediate(&mut self, deriv: &[f64], h: f64);

    /// Combine four stage derivatives into the final state using the
    /// classical RK4 formula `(k1 + 2·k2 + 2·k3 + k4) · dt / 6`, possibly
    /// with implementor-specific clamping or constraint enforcement.
    ///
    /// Called once after all four stages have been evaluated, with the
    /// full dynamic timestep `dt`.
    fn finalize_rk4(&mut self, k1: &[f64], k2: &[f64], k3: &[f64], k4: &[f64], dt: f64);
}
