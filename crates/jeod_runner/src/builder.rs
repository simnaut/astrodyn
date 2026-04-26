//! Runner-side terminal methods for the relocated `jeod_sim::SimulationBuilder`.
//!
//! Phase 6 of #101 moved `VehicleConfig`, `SimulationBuilder`, the runtime
//! fluent `VehicleBuilder`, and the per-vehicle option structs (`SrpModel`,
//! `ShadowBody`, …) into `jeod_sim`. The runtime fluent builder consolidated
//! into the Phase-5 typestate `VehicleBuilder` (also in `jeod_sim`). What
//! remains here is the runner-specific terminal step: materializing a
//! [`SimulationBuilder`] into a [`Simulation`].
//!
//! Two paths are provided so call sites read identically to the
//! pre-relocation API:
//!
//! - [`Simulation::from_builder`] / [`Simulation::from_builder_unchecked`] —
//!   inherent constructors on `Simulation`.
//! - [`SimulationBuilderExt::build`] / `.build_unchecked()` — extension trait
//!   that lets `scenarios::iss_leo().build()?` keep its fluent ergonomics.

use jeod_sim::simulation_builder::SimulationBuilder;
use jeod_sim::SimulationTime;

use crate::Simulation;

impl Simulation {
    /// Start building a simulation with the given time and timestep.
    ///
    /// Convenience wrapper around [`SimulationBuilder::new`] that mirrors the
    /// pre-relocation `Simulation::builder(time, dt)` API.
    pub fn builder(time: SimulationTime, dt: f64) -> SimulationBuilder {
        SimulationBuilder::new(time, dt)
    }

    /// Materialize a [`SimulationBuilder`] into a runtime [`Simulation`]
    /// without running validation. Use for tests that intentionally exercise
    /// invalid configurations.
    pub fn from_builder_unchecked(builder: SimulationBuilder) -> Self {
        let SimulationBuilder {
            time,
            dt,
            atmosphere,
            atmosphere_planet_source,
            ephemeris,
            polar_motion,
            sun_source,
            moon_source,
            sources,
            source_ephem_bodies,
            bodies,
            mass_tree_names,
            mass_tree_attachments,
        } = builder;

        let mut sim = Simulation::new(time, dt);
        sim.atmosphere = atmosphere;
        sim.atmosphere_planet_source = atmosphere_planet_source;
        sim.ephemeris = ephemeris;
        sim.polar_motion = polar_motion;
        sim.sun_source = sun_source;
        sim.moon_source = moon_source;

        for (i, (name, source)) in sources.into_iter().enumerate() {
            sim.add_source(name, source);
            if let Some(Some((target, observer))) = source_ephem_bodies.get(i) {
                sim.set_source_ephemeris(i, *target, *observer);
            }
        }

        for body in bodies {
            sim.add_body(body);
        }

        // Wire up mass tree if any bodies were registered.
        let has_tree = mass_tree_names.iter().any(|n| n.is_some());
        if has_tree {
            for (idx, name) in mass_tree_names.into_iter().enumerate() {
                if let Some(name) = name {
                    sim.add_body_to_tree(idx, name);
                }
            }
            for att in mass_tree_attachments {
                sim.attach(
                    att.child_idx,
                    att.parent_idx,
                    att.offset,
                    att.t_parent_child,
                );
            }
        }

        sim
    }

    /// Materialize a [`SimulationBuilder`] into a runtime [`Simulation`] and
    /// run validation. Returns `Err` on validation failure.
    pub fn from_builder(
        builder: SimulationBuilder,
    ) -> Result<Self, Vec<jeod_sim::ValidationError>> {
        let mut sim = Self::from_builder_unchecked(builder);
        sim.validate()?;
        Ok(sim)
    }
}

/// Extension trait providing terminal `.build()` / `.build_unchecked()`
/// methods on the relocated [`jeod_sim::SimulationBuilder`].
///
/// Mission code typically imports this via `jeod_runner::prelude::*` (or
/// directly via `use jeod_runner::SimulationBuilderExt;`) so
/// `scenarios::iss_leo().build()?` reads naturally. Callers that prefer the
/// explicit form can use [`Simulation::from_builder`] directly.
pub trait SimulationBuilderExt: Sized {
    /// Build and validate the simulation. Returns `Err` on validation
    /// failure.
    fn build(self) -> Result<Simulation, Vec<jeod_sim::ValidationError>>;

    /// Build without validation. Use only for tests that intentionally
    /// exercise invalid configurations.
    fn build_unchecked(self) -> Simulation;
}

impl SimulationBuilderExt for SimulationBuilder {
    fn build(self) -> Result<Simulation, Vec<jeod_sim::ValidationError>> {
        Simulation::from_builder(self)
    }

    fn build_unchecked(self) -> Simulation {
        Simulation::from_builder_unchecked(self)
    }
}
