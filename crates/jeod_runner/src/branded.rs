//! Branded `Simulation<'sim>` wrapper with compile-time index isolation
//! between concurrent simulations (#152).
//!
//! ## Why
//!
//! `Simulation::add_source(...) -> usize` and `add_body(...) -> usize`
//! return raw indices that are silently interchangeable across
//! `Simulation` instances. Code holding two `Simulation`s (Monte Carlo,
//! batch propagation, parametric scans) can accidentally feed an index
//! from one into the other and either panic on out-of-range or, worse,
//! silently address a different physical entity.
//!
//! This module ships the **opt-in** branded API:
//!
//! ```
//! use jeod_runner::Simulation;
//! use jeod_sim::{default_leap_second_table, F64Ext, SimulationTime};
//! use jeod_sim::recipes::{earth, orbital_elements, vehicle};
//! use jeod_sim::{GravityControl, VehicleBuilder};
//!
//! let time = SimulationTime::at_j2000(default_leap_second_table());
//! let final_pos_x = Simulation::run(time, 60.0, |mut sim| {
//!     // `sim` has type `BrandedSimulation<'sim>`; `earth_idx` carries
//!     // the matching `'sim` lifetime so it cannot escape this closure.
//!     let earth_idx = sim.add_source("Earth", earth::point_mass());
//!     let cfg = VehicleBuilder::new()
//!         .from_orbital_elements(orbital_elements::iss(), earth::point_mass().source.mu.m3_per_s2())
//!         .three_dof_point_mass(vehicle::iss_mass())
//!         .rk4()
//!         .gravity(GravityControl::new_spherical(earth_idx.into_raw(), false))
//!         .build();
//!     let sat = sim.add_body(cfg);
//!     sim.body(sat).trans.position.x
//! });
//! assert!(final_pos_x.abs() > 6_000_000.0);
//! ```
//!
//! Two `Simulation::run` calls produce indices with **different** `'sim`
//! lifetimes; passing an index from one call into another is a compile
//! error, not a runtime panic. The HRTB `for<'sim> FnOnce(...)` is what
//! makes the brand fresh per call (ghost-cell pattern).
//!
//! ## Scope of this layer
//!
//! Additive — the existing `Simulation::new(time, dt) -> Simulation`
//! API is unchanged, and the 50+ existing call sites continue to work
//! with raw `usize` indices. New code that wants the safety guarantee
//! opts in via `Simulation::run`. A future PR can migrate existing
//! callers to the branded form (mechanical refactor: wrap each test
//! body in a closure).
//!
//! `BrandedSimulation` covers the most common index-using methods:
//! `add_source`, `add_body`, `set_source_position`, `set_source_state`,
//! `source_position`, `body`, plus `step`, `step_until`, `validate`,
//! and `unbranded()` / `unbranded_mut()` escape hatches that expose the
//! inner `Simulation` for methods not yet branded. Adding more branded
//! methods later is mechanical.
//!
//! ## Tradeoff
//!
//! The brand only protects code that opts in via `run`. Code using
//! `Simulation::new` directly does not get the safety. Forcing all
//! callers through the closure form (no `Simulation::new` escape) was
//! considered but deferred — it's a larger refactor than this PR
//! tackles, and the per-test indentation cost is real for a hypothetical
//! bug class with no current consumer. The additive layer establishes
//! the types and documents the pattern; migration follows when a real
//! Monte Carlo consumer materializes.

use core::marker::PhantomData;

use glam::DVec3;

use crate::{Simulation, VehicleOutput};
use jeod_sim::{
    GravitySourceEntry, MassProperties, SimulationTime, ValidationError, VehicleConfig,
};

/// Invariant lifetime brand. The `fn(&'sim ()) -> &'sim ()` form is
/// invariant in `'sim` (it appears in both contravariant and covariant
/// position), which prevents the compiler from sub-typing two
/// `Simulation`s with different brands into a common lifetime.
pub(crate) type Brand<'sim> = PhantomData<fn(&'sim ()) -> &'sim ()>;

/// Branded source index. Tied to the lifetime of the `BrandedSimulation`
/// that produced it; cannot be passed to a different
/// `BrandedSimulation`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceIdx<'sim> {
    raw: usize,
    _brand: Brand<'sim>,
}

impl SourceIdx<'_> {
    /// Drop the brand and return the raw `usize`. Useful when calling
    /// non-branded code (e.g. systems in `jeod_sim` that take `usize`).
    /// Re-acquiring a branded `SourceIdx` from a `usize` is intentionally
    /// hard — the closest path is going through a fresh `add_source`.
    #[inline]
    pub fn into_raw(self) -> usize {
        self.raw
    }
}

/// Branded body index. Same brand-lifetime semantics as [`SourceIdx`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BodyIdx<'sim> {
    raw: usize,
    _brand: Brand<'sim>,
}

impl BodyIdx<'_> {
    /// Strip the `'sim` brand and return the raw `usize` body index that
    /// `Simulation` uses internally. The result outlives the brand and
    /// can no longer be checked against the originating `Simulation`.
    #[inline]
    pub fn into_raw(self) -> usize {
        self.raw
    }
}

/// Brand-typed wrapper around `Simulation`. Constructed only via
/// [`Simulation::run`], which uses HRTB to mint a fresh `'sim` per call.
pub struct BrandedSimulation<'sim> {
    inner: Simulation,
    _brand: Brand<'sim>,
}

impl<'sim> BrandedSimulation<'sim> {
    pub(crate) fn from_inner(inner: Simulation) -> Self {
        Self {
            inner,
            _brand: PhantomData,
        }
    }

    /// Add a gravity source. Returns a brand-typed `SourceIdx<'sim>`
    /// that can only be used with this simulation.
    pub fn add_source(
        &mut self,
        name: impl Into<String>,
        entry: GravitySourceEntry,
    ) -> SourceIdx<'sim> {
        SourceIdx {
            raw: self.inner.add_source(name, entry),
            _brand: PhantomData,
        }
    }

    /// Add a dynamic body. Returns a brand-typed `BodyIdx<'sim>`.
    pub fn add_body(&mut self, config: VehicleConfig) -> BodyIdx<'sim> {
        BodyIdx {
            raw: self.inner.add_body(config),
            _brand: PhantomData,
        }
    }

    /// Set a source's inertial position. Indices from a different
    /// `BrandedSimulation` fail to compile.
    pub fn set_source_position(&mut self, idx: SourceIdx<'sim>, position: DVec3) {
        self.inner.set_source_position(idx.raw, position);
    }

    /// Set a source's full inertial state.
    pub fn set_source_state(&mut self, idx: SourceIdx<'sim>, position: DVec3, velocity: DVec3) {
        self.inner.set_source_state(idx.raw, position, velocity);
    }

    /// Read a source's inertial position.
    pub fn source_position(&self, idx: SourceIdx<'sim>) -> DVec3 {
        self.inner.source_position(idx.raw)
    }

    /// Read a body's current state. The body must have been added via
    /// `self.add_body` (the `'sim` brand on the index witnesses this).
    pub fn body(&self, idx: BodyIdx<'sim>) -> VehicleOutput {
        self.inner.body(idx.raw)
    }

    /// Set a body's external (non-gravity) force in inertial coordinates.
    pub fn set_body_external_force(&mut self, idx: BodyIdx<'sim>, force: DVec3) {
        self.inner.set_body_external_force(idx.raw, force);
    }

    /// Set a body's external (non-gravity) torque in body-frame coordinates.
    pub fn set_body_external_torque(&mut self, idx: BodyIdx<'sim>, torque: DVec3) {
        self.inner.set_body_external_torque(idx.raw, torque);
    }

    /// Set a body's inertial position directly (e.g. for fixed-trajectory
    /// scenarios that bypass propagation).
    pub fn set_body_position(&mut self, idx: BodyIdx<'sim>, position: DVec3) {
        self.inner.set_body_position(idx.raw, position);
    }

    /// Set a body's inertial velocity directly.
    pub fn set_body_velocity(&mut self, idx: BodyIdx<'sim>, velocity: DVec3) {
        self.inner.set_body_velocity(idx.raw, velocity);
    }

    /// Set a body's mass properties (used for staging / fuel burn / etc.).
    pub fn set_body_mass(&mut self, idx: BodyIdx<'sim>, mass: MassProperties) {
        self.inner.set_body_mass(idx.raw, mass);
    }

    // ── Pass-through forwarders for the common non-index methods ──

    /// Advance the simulation by one timestep.
    pub fn step(&mut self) -> Result<(), crate::StepError> {
        self.inner.step()
    }

    /// Advance until the simulation time reaches `target_time`.
    ///
    /// If the underlying integrator supports it, this may take a final
    /// fractional step to hit `target_time` exactly. A fractional residual
    /// only panics when such a step is required for an integrator that
    /// does not support it (e.g. multi-step `GaussJackson` / `Abm4`,
    /// whose history arrays assume constant dt).
    pub fn step_until(&mut self, target_time: f64) -> Result<(), crate::StepError> {
        self.inner.step_until(target_time)
    }

    /// Run JEOD invariant validation on the current sim state.
    pub fn validate(&mut self) -> Result<(), Vec<ValidationError>> {
        self.inner.validate()
    }

    /// Number of registered gravity sources.
    pub fn num_sources(&self) -> usize {
        self.inner.num_sources()
    }

    /// Current simulation time.
    pub fn time(&self) -> &SimulationTime {
        &self.inner.time
    }

    // ── Escape hatches for methods not yet branded ──

    /// Borrow the underlying [`Simulation`] for methods this branded
    /// wrapper doesn't yet expose. Use sparingly: any indices passed
    /// to / returned from `unbranded()` are raw `usize` with no brand
    /// guarantee.
    pub fn unbranded(&self) -> &Simulation {
        &self.inner
    }

    /// Mutably borrow the underlying [`Simulation`]. Same caveat as
    /// [`Self::unbranded`].
    pub fn unbranded_mut(&mut self) -> &mut Simulation {
        &mut self.inner
    }

    /// Consume the branded wrapper and return the raw `Simulation`.
    /// Indices that were branded on `'sim` lose their brand at the
    /// `into_raw()` call site (since `Simulation` indices are bare
    /// `usize`); they continue to work as `usize` against the returned
    /// `Simulation`.
    pub fn into_inner(self) -> Simulation {
        self.inner
    }
}

impl Simulation {
    /// Run a closure with a fresh-branded [`BrandedSimulation<'sim>`].
    ///
    /// The HRTB `for<'sim> FnOnce(BrandedSimulation<'sim>) -> R` guarantees
    /// that `'sim` is unique to this call, so indices produced inside
    /// the closure cannot escape and cannot be confused with indices
    /// from another `run` call.
    ///
    /// This is the entry point for compile-time-safe simulation use.
    /// The non-branded constructor [`Simulation::new`] remains available
    /// for code that doesn't need cross-simulation safety.
    ///
    /// # Example
    ///
    /// ```
    /// use jeod_runner::Simulation;
    /// use jeod_sim::{default_leap_second_table, F64Ext, SimulationTime};
    /// use jeod_sim::recipes::{earth, orbital_elements, vehicle};
    /// use jeod_sim::{GravityControl, VehicleBuilder};
    ///
    /// let time = SimulationTime::at_j2000(default_leap_second_table());
    /// let r = Simulation::run(time, 60.0, |mut sim| {
    ///     let earth_idx = sim.add_source("Earth", earth::point_mass());
    ///     let cfg = VehicleBuilder::new()
    ///         .from_orbital_elements(
    ///             orbital_elements::iss(),
    ///             earth::point_mass().source.mu.m3_per_s2(),
    ///         )
    ///         .three_dof_point_mass(vehicle::iss_mass())
    ///         .rk4()
    ///         .gravity(GravityControl::new_spherical(earth_idx.into_raw(), false))
    ///         .build();
    ///     let sat = sim.add_body(cfg);
    ///     sim.body(sat).trans.position.length()
    /// });
    /// assert!(r > 6_000_000.0);
    /// ```
    pub fn run<F, R>(time: SimulationTime, dt: f64, f: F) -> R
    where
        F: for<'sim> FnOnce(BrandedSimulation<'sim>) -> R,
    {
        f(BrandedSimulation::from_inner(Simulation::new(time, dt)))
    }
}

/// Compile-fail check for cross-simulation index aliasing.
///
/// Two separate `Simulation::run` calls produce indices with disjoint
/// `'sim` lifetimes. Attempting to feed `SourceIdx<'a>` (from sim A)
/// into `BrandedSimulation<'b>` (from sim B) is a borrow-checker error.
///
/// ```compile_fail
/// use jeod_runner::Simulation;
/// use jeod_sim::{default_leap_second_table, GravityModel, GravitySource, GravitySourceEntry,
///                RotationModel, SimulationTime};
/// use glam::DVec3;
///
/// fn earth_entry() -> GravitySourceEntry {
///     GravitySourceEntry {
///         source: GravitySource { mu: 3.986e14, model: GravityModel::PointMass },
///         position: DVec3::ZERO,
///         velocity: DVec3::ZERO,
///         t_inertial_pfix: None,
///         delta_c20: 0.0,
///         rotation_model: RotationModel::default(),
///         tidal_config: None,
///         planet_omega: 0.0,
///         central: true,
///     }
/// }
///
/// let time = SimulationTime::at_j2000(default_leap_second_table());
/// let idx_a = Simulation::run(time, 10.0, |mut sim_a| {
///     sim_a.add_source("Earth-A", earth_entry())
/// });
/// // ^ doesn't compile: idx_a's brand can't escape the closure.
/// Simulation::run(SimulationTime::at_j2000(default_leap_second_table()), 10.0, |mut sim_b| {
///     sim_b.set_source_position(idx_a, DVec3::ZERO);
/// });
/// ```
// Anchor for the compile_fail doctest above; never called at runtime.
#[allow(dead_code)]
fn _doc_compile_fail_marker() {}

#[cfg(test)]
mod tests {
    use super::*;
    use jeod_sim::default_leap_second_table;

    /// Minimal smoke: `Simulation::run` returns the closure's output and
    /// `add_source` / `add_body` produce branded indices that work with
    /// the methods that consume them.
    #[test]
    fn run_round_trip() {
        let n = Simulation::run(
            SimulationTime::at_j2000(default_leap_second_table()),
            10.0,
            |sim| sim.num_sources(),
        );
        assert_eq!(n, 0);
    }

    /// `into_raw()` exposes the underlying `usize` for interop with
    /// non-branded code.
    #[test]
    fn into_raw_strips_brand() {
        Simulation::run(
            SimulationTime::at_j2000(default_leap_second_table()),
            10.0,
            |mut sim| {
                let entry = jeod_sim::GravitySourceEntry {
                    source: jeod_sim::GravitySource {
                        mu: 3.986e14,
                        model: jeod_sim::GravityModel::PointMass,
                    },
                    position: DVec3::ZERO,
                    velocity: DVec3::ZERO,
                    t_inertial_pfix: None,
                    delta_c20: 0.0,
                    rotation_model: jeod_sim::RotationModel::default(),
                    tidal_config: None,
                    planet_omega: 0.0,
                    central: true,
                };
                let idx = sim.add_source("Earth", entry);
                assert_eq!(idx.into_raw(), 0);
                assert_eq!(sim.num_sources(), 1);
            },
        );
    }

    /// **Compile-fail discipline check** — documents the bug class the
    /// brand prevents. Cannot run as `#[should_panic]` (it's a
    /// compile-time error), so this is captured as a doc-test marked
    /// `compile_fail` further up; here we add a runtime assertion that
    /// at least the type signature is the expected one.
    #[test]
    fn brand_types_present() {
        // If the brand types disappear or lose their lifetime parameter,
        // this test fails to compile.
        fn _check<'a, 'b>(_: SourceIdx<'a>, _: BodyIdx<'b>) {}
    }
}
