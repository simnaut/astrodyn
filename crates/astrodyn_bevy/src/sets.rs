//! [`AstrodynSet`] — the Bevy `SystemSet` partition that mirrors JEOD's
//! per-step pipeline.
//!
//! # Schedule as integration group
//!
//! JEOD's `JeodIntegrationGroup` collects a set of bodies that advance
//! together under a single shared time step, with one integrator stage
//! that consumes the group's force/torque accumulators and produces the
//! next state for every member. In `astrodyn_bevy`, the Bevy `FixedUpdate`
//! schedule **is** that integration group:
//!
//! - One `FixedUpdate` tick = one group advance at the schedule's fixed
//!   `dt`. Every entity matched by the integrating system's query
//!   shares that `dt` by construction — no per-body bookkeeping.
//! - The seven [`AstrodynSet`] stages below run once per tick, in the order
//!   declared, mirroring JEOD's init/update pipeline (force collection →
//!   integration → derived state).
//! - **Multi-stage integrators (RK4, etc.) are an inner loop inside
//!   [`AstrodynSet::Integration`], not multiple schedule passes.** A four-
//!   stage RK4 evaluates four sub-steps within the integration system
//!   for the same outer `FixedUpdate` tick.
//! - Multi-body sims (Apollo, Earth–Moon) need no extra wiring: every
//!   body registered in the schedule is automatically a member of the
//!   "group" in JEOD's sense.
//!
//! # Multiple integration groups
//!
//! When a scenario genuinely needs *separate* groups — different bodies
//! integrating at different cadences — register a second Bevy schedule
//! and run it on its own tick rate. This is plain Bevy mechanics, not
//! a JEOD-specific construct, and no current scenario in this workspace
//! uses it; the single-`FixedUpdate` model covers every shipped sim.

use std::any::TypeId;

use bevy::prelude::*;

/// One per-planet `SystemSet` per registered [`astrodyn::Planet`] type,
/// keyed by [`TypeId`].
///
/// Every `<P>`-typed system in `AstrodynPlugin::build` (Earth) and in
/// [`crate::register_planet_systems::<P>`] (every other planet) is added
/// to its planet's `PerPlanetSet`. Cross-planet sets are declared
/// `.ambiguous_with` each other inside `register_planet_systems`,
/// which walks [`crate::body_action::RegisteredPlanetsR`] to find every
/// prior planet and marks the cross-set pair safe.
///
/// `TypeId` keying (rather than a generic
/// `PerPlanetSet<P>(PhantomData<P>)`) is what makes the cross-planet
/// walk expressible. Reading a `TypeId` out of `RegisteredPlanetsR`
/// does not give us the concrete planet type back, so a generic
/// `PerPlanetSet<P>` would not be constructible from runtime data.
/// `TypeId` is `Send + Sync + 'static + Debug + Clone + Eq + Hash`, so
/// the `SystemSet` derive accepts the newtype.
///
/// Intra-planet ordering inside one `PerPlanetSet` continues to be
/// expressed with the existing `.before` / `.after` edges — set
/// membership is solely a cross-planet structural-disjointness
/// declaration, not an ordering construct.
///
/// See issue #562 / PR #565 for the motivation: the multi-planet unit
/// tests in `astrodyn_bevy` register both Earth and Mars and the static
/// schedule audit (under the `schedule_audit` Cargo feature) flagged
/// cross-planet pairs of per-planet systems racing on shared
/// non-generic component types (`RotationalStateC`, `FrameTransC`,
/// `MassPropertiesC`, …). The cross-planet pairs are runtime-disjoint
/// by entity, but the *mechanism* of disjointness varies by system,
/// and `PerPlanetSet` is how we communicate either route to Bevy:
///
/// 1. **Required-query-item filtering.** Systems like
///    [`crate::systems::integration_system::<P>`] and
///    [`crate::systems::flat_plate_srp_system::<P>`] take
///    `&mut TranslationalStateC<P>` as a required query item. Their
///    query only matches entities that carry the `<P>`-tagged
///    storage, which by construction is a single planet's. Bodies
///    carry `TranslationalStateC<Earth>` *or*
///    `TranslationalStateC<Mars>`, never both, so the `<Earth>` and
///    `<Mars>` instantiations of these systems iterate disjoint
///    entity subsets.
///
/// 2. **Queue partitioning.** Systems like
///    [`crate::body_action::body_action_system::<P>`] take
///    `Option<&mut TranslationalStateC<P>>` and filter only by
///    `With<DynamicsConfigC>`, so the query itself *could* match a
///    body of either planet. The runtime disjointness comes from the
///    per-planet `BodyActionsR<P>` queue: actions are routed to
///    `<P>`'s queue at submit time via
///    `BodyActionCommandsExt::add_body_action_for::<P>` and the
///    `body_action_unregistered_planet_fence_system` panics on a
///    `<P>` whose `TypeId` isn't in `RegisteredPlanetsR`. The
///    `<Mars>` apply pass walks `BodyActionsR<Mars>` and reaches
///    only Mars-tagged bodies — Earth-tagged bodies are reached
///    only via `BodyActionsR<Earth>` from the `<Earth>` apply pass.
///
/// Both routes produce the same runtime invariant (no two
/// per-planet `<P>` systems write the same entity in the same tick)
/// but Bevy's static analysis sees only the shared mutable component
/// access. `.ambiguous_with` is the right mechanism for both
/// because the structural justification is symmetric: we know
/// runtime-disjoint, Bevy doesn't.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct PerPlanetSet(pub TypeId);

impl PerPlanetSet {
    /// Construct the set for planet `P`.
    pub fn of<P: 'static>() -> Self {
        Self(TypeId::of::<P>())
    }
}

/// Bevy system-set partition mirroring JEOD's per-step pipeline. Stages
/// run in declaration order; `AstrodynPlugin::build` configures the ordering
/// when the plugin is added to the app.
///
/// All seven stages execute together inside one `FixedUpdate` tick at
/// the schedule's shared `dt`, defining a single JEOD-style integration
/// group; see the module-level docs for the full mapping.
// JEOD_INV: DM.04 — system set ordering mirrors JEOD init/update pipeline
// JEOD_INV: DM.13 — EphemerisUpdate before Environment ensures ephemeris is current for gravity
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum AstrodynSet {
    /// Time scale update (TAI, UTC, TDB, GMST, etc.).
    TimeUpdate,
    /// Ephemeris update (planet positions from DE4xx data).
    EphemerisUpdate,
    /// Environment computation (gravity, atmosphere).
    Environment,
    /// Interaction computation (aero drag, SRP, gravity torque).
    Interaction,
    /// Force and torque collection.
    ForceCollection,
    /// State integration (RK4, etc.).
    Integration,
    /// Derived state computation (orbital elements, Euler angles, etc.).
    DerivedState,
}
