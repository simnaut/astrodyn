//! Runner-side terminal methods for the relocated `jeod_sim::SimulationBuilder`.
//!
//! Phase 6 of #101 moved `VehicleConfig`, `SimulationBuilder`, the runtime
//! fluent `VehicleBuilder`, and the per-vehicle option structs (`SrpModel`,
//! `ShadowBody`, …) into `jeod_sim`. The runtime fluent builder consolidated
//! into the Phase-5 typestate `VehicleBuilder` (also in `jeod_sim`). What
//! remains here is the runner-specific terminal step: materializing a
//! [`SimulationBuilder`] into a [`Simulation`].
//!
//! - [`Simulation::from_builder`] — inherent constructor on `Simulation`.
//! - [`SimulationBuilderExt::build`] — extension trait that lets
//!   `Mission::iss_leo().into_builder().build()?` read naturally.

use jeod_sim::simulation_builder::SimulationBuilder;

use crate::Simulation;

impl Simulation {
    /// Materialize a [`SimulationBuilder`] into a runtime [`Simulation`] and
    /// run validation. Returns `Err` on validation failure.
    pub fn from_builder(
        builder: SimulationBuilder,
    ) -> Result<Self, Vec<jeod_sim::ValidationError>> {
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
        //
        // Builder-time `attach_bodies` declarations route through the
        // runner's `attach_preserving_initial_state` path rather than
        // the public `Simulation::attach`. The latter runs JEOD's
        // `combine_states_at_attach` momentum-conservation kernel and
        // overwrites the integrated tree root's `body.trans` /
        // `body.rot` with the merged composite-body state — the right
        // semantics for a runtime in-flight attach (which is what
        // sub-issue #297 / PR #307 wired up). At build time, however,
        // the caller has already populated `VehicleConfig::trans` /
        // `rot` for each body with the post-attach state they want
        // (e.g. parent on its orbital element initial state, child
        // staged in LVLH relative to the parent, both intended to be
        // a single articulated vehicle from t=0). Running the runtime
        // combine over those would treat the user-supplied initial
        // conditions as a pre-attach pair and merge them with the
        // mass-weighted velocity / CoM-shift formula — silently
        // corrupting the spec'd initial state. The configuration-time
        // path preserves the spec verbatim while still doing the
        // tree-mutation, composite-mass resync, and integrator-history
        // reset that any topology change requires.
        let has_tree = mass_tree_names.iter().any(|n| n.is_some());
        if has_tree {
            for (idx, name) in mass_tree_names.into_iter().enumerate() {
                if let Some(name) = name {
                    sim.add_body_to_tree(idx, name);
                }
            }
            for att in mass_tree_attachments {
                sim.attach_preserving_initial_state(
                    att.child_idx,
                    att.parent_idx,
                    att.offset,
                    att.t_parent_child,
                );
            }
        }

        sim.validate()?;
        Ok(sim)
    }
}

/// Extension trait providing the terminal `.build()` method on the relocated
/// [`jeod_sim::SimulationBuilder`].
///
/// Mission code typically imports this via `jeod_runner::prelude::*` (or
/// directly via `use jeod_runner::SimulationBuilderExt;`) so
/// `Mission::iss_leo().into_builder().build()?` reads naturally. Callers that prefer the
/// explicit form can use [`Simulation::from_builder`] directly.
pub trait SimulationBuilderExt: Sized {
    /// Build and validate the simulation. Returns `Err` on validation
    /// failure.
    fn build(self) -> Result<Simulation, Vec<jeod_sim::ValidationError>>;
}

impl SimulationBuilderExt for SimulationBuilder {
    fn build(self) -> Result<Simulation, Vec<jeod_sim::ValidationError>> {
        Simulation::from_builder(self)
    }
}
