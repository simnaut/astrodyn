// JEOD_INV: TS.01 — `<SelfRef>` wildcards on this file's `BodyReader` /
// `BodyMutator` SystemParam returns and inputs are the storage-boundary
// surface for per-entity body identity (mirrors the `<SelfRef>` use in
// `crate::components::state` for the underlying body components). The
// wildcard is resolved by the entity passed to the accessor at call
// time; system code paths that consume the returned typed values pin
// `<V: Vehicle>` themselves.
//! Body-state read/write API for Bevy missions.
//!
//! `astrodyn_runner::Simulation` exposes `body()` / `body_state()` for
//! reads and `set_body_external_force` / `set_body_external_torque` /
//! `set_body_external_force_struct` / `set_body_external_torque_struct`
//! (each with a `_typed` sibling) for time-scheduled external loads.
//!
//! The Bevy adapter mirrors that surface as two `SystemParam`s:
//!
//! - [`BodyReader`] — read-only per-field accessors over a body entity's
//!   translational / rotational / mass / per-step acceleration / external
//!   load components. All accessors return *typed* values (`Force<…>`,
//!   `Acceleration<RootInertial>`, etc.); the raw-`DVec3` reader is not
//!   exposed because the underlying components are already typed end-to-end.
//! - [`BodyMutator`] — read+write API for the four external-load setters,
//!   typed-first with raw `// ESCAPE_HATCH:` siblings retained for parity
//!   with the runner's raw `Simulation::set_body_external_force` entry
//!   points.
//!
//! Both filter on [`DynamicsConfigC`] (the universally-present body
//! component — every body spawned via [`crate::VehicleConfigBevyExt::spawn_bevy`]
//! carries one). Passing a non-body entity (e.g. a gravity source) panics
//! with a diagnostic naming the misuse — the same fail-loud pattern
//! [`crate::source_mutator::SourceMutator`] applies for source-targeted
//! queries.
//!
//! The struct-frame variants ([`ExternalForceStructC`] /
//! [`ExternalTorqueStructC`]) default to zero on bodies that don't carry
//! them; the mutator's typed setters auto-insert the component on the
//! zero-to-non-zero transition — mirroring the runner's
//! `SimBody::from_config` zero-init pattern and the
//! `SourceMutator::set_source_state` auto-insert precedent.

use astrodyn::{
    Acceleration, AngularAcceleration, BodyFrame, Force, Planet, PlanetInertial, RootInertial,
    RotationalStateTyped, SelfRef, StructuralFrame, Torque, TranslationalStateTyped,
};
use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use glam::DVec3;

use crate::components::{
    DynamicsConfigC, ExternalForceC, ExternalForceStructC, ExternalTorqueC, ExternalTorqueStructC,
    FrameDerivativesC, MassPropertiesC, RotationalStateC, TranslationalStateC,
};

/// Bevy `SystemParam` exposing the **read-only** half of the
/// `astrodyn_runner::Simulation::body*` API on entities carrying
/// [`DynamicsConfigC`] (which `spawn_bevy` always inserts).
///
/// All queries are immutable, so this composes inside one system with
/// other read-only SystemParams (e.g. [`crate::frame_param::FrameOrigin`]).
/// Mission code that only inspects body state should pick
/// [`BodyReader`]; reach for [`BodyMutator`] only when also writing
/// external loads.
///
/// Per-field accessors rather than a monolithic `body() -> VehicleOutput`
/// — the runner's `VehicleOutput` mixes a dozen+ optional fields most
/// readers don't need, and the per-field shape keeps Bevy queries narrow
/// (avoids holding `&'static` views of components the consumer never
/// reads).
#[derive(SystemParam)]
pub struct BodyReader<'w, 's, P: Planet> {
    body_filter: Query<'w, 's, (), With<DynamicsConfigC>>,
    trans: Query<'w, 's, &'static TranslationalStateC<P>>,
    rot: Query<'w, 's, &'static RotationalStateC>,
    mass: Query<'w, 's, &'static MassPropertiesC>,
    frame_derivs: Query<'w, 's, &'static FrameDerivativesC>,
    ext_force: Query<'w, 's, &'static ExternalForceC>,
    ext_torque: Query<'w, 's, &'static ExternalTorqueC>,
    ext_force_struct: Query<'w, 's, &'static ExternalForceStructC>,
    ext_torque_struct: Query<'w, 's, &'static ExternalTorqueStructC>,
    names: Query<'w, 's, &'static Name>,
}

impl<P: Planet> BodyReader<'_, '_, P> {
    /// Typed translational state of `body` in `P`'s planet-inertial
    /// frame. Mirrors `VehicleOutput.trans` after the
    /// `TranslationalStateC<P>` storage convention (the component
    /// already carries `<PlanetInertial<P>>`, so this is a direct
    /// projection — no relabel, no arithmetic).
    ///
    /// # Panics
    ///
    /// Panics if `body` is not a registered dynamic body (missing
    /// [`DynamicsConfigC`]) or lacks [`TranslationalStateC<P>`].
    pub fn body_translational_state(
        &self,
        body: Entity,
    ) -> TranslationalStateTyped<PlanetInertial<P>> {
        _assert_is_body(
            &self.body_filter,
            &self.names,
            body,
            "body_translational_state",
        );
        self.trans
            .get(body)
            .unwrap_or_else(|err| {
                panic!(
                    "BodyReader::body_translational_state: {label} is a registered \
                     dynamic body but lacks TranslationalStateC<P> ({err:?}). \
                     Spawn the body via VehicleConfig::spawn_bevy (which inserts \
                     TranslationalStateC), or attach the component directly.",
                    label = _entity_label(&self.names, body),
                )
            })
            .0
    }

    /// Typed rotational state of `body`, or `None` for 3-DOF bodies
    /// (those spawned without a `RotationalStateC`). Mirrors
    /// `VehicleOutput.rot`.
    ///
    /// # Panics
    ///
    /// Panics if `body` is not a registered dynamic body.
    pub fn body_rotational_state(&self, body: Entity) -> Option<RotationalStateTyped<SelfRef>> {
        _assert_is_body(
            &self.body_filter,
            &self.names,
            body,
            "body_rotational_state",
        );
        self.rot.get(body).ok().map(|c| c.0)
    }

    /// Typed mass properties of `body`, or `None` for 3-DOF bodies
    /// spawned without a `MassPropertiesC`.
    ///
    /// # Panics
    ///
    /// Panics if `body` is not a registered dynamic body.
    pub fn body_mass(&self, body: Entity) -> Option<astrodyn::MassPropertiesTyped<SelfRef>> {
        _assert_is_body(&self.body_filter, &self.names, body, "body_mass");
        self.mass.get(body).ok().map(|c| c.0)
    }

    /// Typed translational acceleration of `body` (`Acceleration<RootInertial>`)
    /// from the last `Update` cycle. Mirrors `VehicleOutput.trans_accel`;
    /// reads directly from [`FrameDerivativesC`] (which has carried the
    /// typed value since #397 — the runner's `VehicleOutput.trans_accel`
    /// is the laggard tightened in this PR).
    ///
    /// Returns the typed zero if the body has no `FrameDerivativesC`
    /// (which the `#[require(FrameDerivativesC)]` attribute on
    /// `DynamicsConfigC` prevents in practice — but the fallback keeps
    /// the accessor `Option`-free for ergonomics).
    ///
    /// # Panics
    ///
    /// Panics if `body` is not a registered dynamic body.
    pub fn body_trans_accel(&self, body: Entity) -> Acceleration<RootInertial> {
        _assert_is_body(&self.body_filter, &self.names, body, "body_trans_accel");
        self.frame_derivs
            .get(body)
            .map(|c| c.0.trans_accel)
            .unwrap_or_default()
    }

    /// Typed rotational acceleration of `body`
    /// (`AngularAcceleration<BodyFrame<SelfRef>>`). Mirrors
    /// `VehicleOutput.rot_accel`; returns `None` for 3-DOF bodies
    /// (those without a `RotationalStateC`).
    ///
    /// # Panics
    ///
    /// Panics if `body` is not a registered dynamic body.
    pub fn body_rot_accel(&self, body: Entity) -> Option<AngularAcceleration<BodyFrame<SelfRef>>> {
        _assert_is_body(&self.body_filter, &self.names, body, "body_rot_accel");
        // Match the runner's `VehicleOutput::rot_accel` gating: `None` for
        // 3-DOF bodies (no rotational state), `Some(typed)` for 6-DOF.
        self.rot.get(body).ok().map(|_| {
            self.frame_derivs
                .get(body)
                .map(|c| c.0.rot_accel)
                .unwrap_or_default()
        })
    }

    /// Typed external inertial-frame force scheduled for collection on
    /// `body`. Returns the typed zero if the body has no
    /// [`ExternalForceC`] — matching the runner's "uninstalled = 0"
    /// convention.
    ///
    /// # Panics
    ///
    /// Panics if `body` is not a registered dynamic body.
    pub fn body_external_force(&self, body: Entity) -> Force<RootInertial> {
        _assert_is_body(&self.body_filter, &self.names, body, "body_external_force");
        self.ext_force.get(body).map(|c| c.0).unwrap_or_default()
    }

    /// Typed external body-frame torque scheduled for collection on
    /// `body`. Returns the typed zero if the body has no
    /// [`ExternalTorqueC`].
    ///
    /// # Panics
    ///
    /// Panics if `body` is not a registered dynamic body.
    pub fn body_external_torque(&self, body: Entity) -> Torque<BodyFrame<SelfRef>> {
        _assert_is_body(&self.body_filter, &self.names, body, "body_external_torque");
        self.ext_torque.get(body).map(|c| c.0).unwrap_or_default()
    }

    /// Typed external structural-frame force scheduled for collection
    /// on `body`. Returns the typed zero if the body has no
    /// [`ExternalForceStructC`].
    ///
    /// # Panics
    ///
    /// Panics if `body` is not a registered dynamic body.
    pub fn body_external_force_struct(&self, body: Entity) -> Force<StructuralFrame<SelfRef>> {
        _assert_is_body(
            &self.body_filter,
            &self.names,
            body,
            "body_external_force_struct",
        );
        self.ext_force_struct
            .get(body)
            .map(|c| c.0)
            .unwrap_or_default()
    }

    /// Typed external structural-frame torque scheduled for collection
    /// on `body`. Returns the typed zero if the body has no
    /// [`ExternalTorqueStructC`].
    ///
    /// # Panics
    ///
    /// Panics if `body` is not a registered dynamic body.
    pub fn body_external_torque_struct(&self, body: Entity) -> Torque<StructuralFrame<SelfRef>> {
        _assert_is_body(
            &self.body_filter,
            &self.names,
            body,
            "body_external_torque_struct",
        );
        self.ext_torque_struct
            .get(body)
            .map(|c| c.0)
            .unwrap_or_default()
    }
}

/// Bevy `SystemParam` exposing the **read+write** body-load API on
/// entities carrying [`DynamicsConfigC`]. Mirrors
/// `astrodyn_runner::Simulation`'s four `set_body_external_*` setters
/// (with `_typed` siblings) plus the read accessors mission code needs
/// to inspect state while writing back forces.
///
/// The typed setters auto-insert the load component when transitioning
/// from absent (or zero) — the same pattern
/// [`crate::source_mutator::SourceMutator::set_source_state`] uses for
/// `SourceInertialVelocityC`. The raw setters are `// ESCAPE_HATCH:`
/// annotated parity shims for the runner's raw `DVec3` entry points;
/// callers should prefer the typed siblings.
///
/// Bodies have no analogue of [`crate::components::CentralSourceMarker`]
/// (the central-body convention applies to gravity sources only), so no
/// "central body protection" panic exists here — the
/// `With<DynamicsConfigC>` filter is the only fail-loud check.
#[derive(SystemParam)]
pub struct BodyMutator<'w, 's, P: Planet> {
    commands: Commands<'w, 's>,
    body_filter: Query<'w, 's, (), With<DynamicsConfigC>>,
    ext_force: Query<'w, 's, &'static mut ExternalForceC, With<DynamicsConfigC>>,
    ext_torque: Query<'w, 's, &'static mut ExternalTorqueC, With<DynamicsConfigC>>,
    ext_force_struct: Query<'w, 's, &'static mut ExternalForceStructC, With<DynamicsConfigC>>,
    ext_torque_struct: Query<'w, 's, &'static mut ExternalTorqueStructC, With<DynamicsConfigC>>,
    // Read-only siblings — let one mutator read+write within one system.
    trans_read: Query<'w, 's, &'static TranslationalStateC<P>>,
    rot_read: Query<'w, 's, &'static RotationalStateC>,
    frame_derivs_read: Query<'w, 's, &'static FrameDerivativesC>,
    names: Query<'w, 's, &'static Name>,
}

impl<P: Planet> BodyMutator<'_, '_, P> {
    // ── Read-only accessors (composition with mutating systems) ──

    /// See [`BodyReader::body_translational_state`].
    pub fn body_translational_state(
        &self,
        body: Entity,
    ) -> TranslationalStateTyped<PlanetInertial<P>> {
        _assert_is_body(
            &self.body_filter,
            &self.names,
            body,
            "body_translational_state",
        );
        self.trans_read
            .get(body)
            .unwrap_or_else(|err| {
                panic!(
                    "BodyMutator::body_translational_state: {label} is a registered \
                     dynamic body but lacks TranslationalStateC<P> ({err:?}).",
                    label = _entity_label(&self.names, body),
                )
            })
            .0
    }

    /// See [`BodyReader::body_rotational_state`].
    pub fn body_rotational_state(&self, body: Entity) -> Option<RotationalStateTyped<SelfRef>> {
        _assert_is_body(
            &self.body_filter,
            &self.names,
            body,
            "body_rotational_state",
        );
        self.rot_read.get(body).ok().map(|c| c.0)
    }

    /// See [`BodyReader::body_trans_accel`].
    pub fn body_trans_accel(&self, body: Entity) -> Acceleration<RootInertial> {
        _assert_is_body(&self.body_filter, &self.names, body, "body_trans_accel");
        self.frame_derivs_read
            .get(body)
            .map(|c| c.0.trans_accel)
            .unwrap_or_default()
    }

    // ── Typed setters (primary surface) ──

    /// Typed sibling of [`set_body_external_force`](Self::set_body_external_force).
    /// The `Force<RootInertial>` argument's phantom enforces at compile
    /// time that the value is in the root-inertial frame, matching the
    /// runner's `Simulation::set_body_external_force_typed`.
    ///
    /// Auto-inserts [`ExternalForceC`] if the body doesn't carry one yet
    /// (i.e. was spawned with a zero `VehicleConfig.external_force`, the
    /// runner's `SimBody::from_config:373-374` zero-init equivalent).
    pub fn set_body_external_force_typed(&mut self, body: Entity, force: Force<RootInertial>) {
        _assert_is_body(
            &self.body_filter,
            &self.names,
            body,
            "set_body_external_force_typed",
        );
        if let Ok(mut c) = self.ext_force.get_mut(body) {
            c.0 = force;
        } else {
            self.commands.entity(body).insert(ExternalForceC(force));
        }
    }

    /// Typed sibling of [`set_body_external_torque`](Self::set_body_external_torque).
    /// The `Torque<BodyFrame<SelfRef>>` argument's phantom enforces at
    /// compile time that the value is expressed in the body frame.
    pub fn set_body_external_torque_typed(
        &mut self,
        body: Entity,
        torque: Torque<BodyFrame<SelfRef>>,
    ) {
        _assert_is_body(
            &self.body_filter,
            &self.names,
            body,
            "set_body_external_torque_typed",
        );
        if let Ok(mut c) = self.ext_torque.get_mut(body) {
            c.0 = torque;
        } else {
            self.commands.entity(body).insert(ExternalTorqueC(torque));
        }
    }

    /// Typed sibling of [`set_body_external_force_struct`](Self::set_body_external_force_struct).
    /// The `Force<StructuralFrame<SelfRef>>` argument's phantom enforces
    /// at compile time that the value is in the body's structural frame.
    ///
    /// Auto-inserts [`ExternalForceStructC`] when the body doesn't carry
    /// one — `VehicleConfig` has no `external_force_struct` field today
    /// (matching the runner's `SimBody::from_config` zero-init), so the
    /// component is absent until first set.
    pub fn set_body_external_force_struct_typed(
        &mut self,
        body: Entity,
        force: Force<StructuralFrame<SelfRef>>,
    ) {
        _assert_is_body(
            &self.body_filter,
            &self.names,
            body,
            "set_body_external_force_struct_typed",
        );
        if let Ok(mut c) = self.ext_force_struct.get_mut(body) {
            c.0 = force;
        } else {
            self.commands
                .entity(body)
                .insert(ExternalForceStructC(force));
        }
    }

    /// Typed sibling of [`set_body_external_torque_struct`](Self::set_body_external_torque_struct).
    /// The `Torque<StructuralFrame<SelfRef>>` argument's phantom enforces
    /// at compile time that the value is in the body's structural frame.
    pub fn set_body_external_torque_struct_typed(
        &mut self,
        body: Entity,
        torque: Torque<StructuralFrame<SelfRef>>,
    ) {
        _assert_is_body(
            &self.body_filter,
            &self.names,
            body,
            "set_body_external_torque_struct_typed",
        );
        if let Ok(mut c) = self.ext_torque_struct.get_mut(body) {
            c.0 = torque;
        } else {
            self.commands
                .entity(body)
                .insert(ExternalTorqueStructC(torque));
        }
    }

    // ── Raw setters (parity shims for the runner's raw entry points) ──

    /// Raw-`DVec3` parity shim for the runner's
    /// `Simulation::set_body_external_force`. Prefer
    /// [`set_body_external_force_typed`](Self::set_body_external_force_typed).
    // ESCAPE_HATCH: parity with raw runner setter `Simulation::set_body_external_force`.
    // Closes once the runner's raw entry point is itself retired.
    pub fn set_body_external_force(&mut self, body: Entity, force: DVec3) {
        // allowed: raw-DVec3 escape-hatch boundary mirroring the runner's
        // raw `Simulation::set_body_external_force` input.
        self.set_body_external_force_typed(body, Force::<RootInertial>::from_raw_si(force));
    }

    /// Raw-`DVec3` parity shim for the runner's
    /// `Simulation::set_body_external_torque`. Prefer
    /// [`set_body_external_torque_typed`](Self::set_body_external_torque_typed).
    // ESCAPE_HATCH: parity with raw runner setter `Simulation::set_body_external_torque`.
    pub fn set_body_external_torque(&mut self, body: Entity, torque: DVec3) {
        // allowed: raw-DVec3 escape-hatch boundary mirroring the runner's raw input.
        let typed = Torque::<BodyFrame<SelfRef>>::from_raw_si(torque);
        self.set_body_external_torque_typed(body, typed);
    }

    /// Raw-`DVec3` parity shim for the runner's
    /// `Simulation::set_body_external_force_struct`. Prefer
    /// [`set_body_external_force_struct_typed`](Self::set_body_external_force_struct_typed).
    // ESCAPE_HATCH: parity with raw runner setter `Simulation::set_body_external_force_struct`.
    pub fn set_body_external_force_struct(&mut self, body: Entity, force: DVec3) {
        // allowed: raw-DVec3 escape-hatch boundary mirroring the runner's raw input.
        let typed = Force::<StructuralFrame<SelfRef>>::from_raw_si(force);
        self.set_body_external_force_struct_typed(body, typed);
    }

    /// Raw-`DVec3` parity shim for the runner's
    /// `Simulation::set_body_external_torque_struct`. Prefer
    /// [`set_body_external_torque_struct_typed`](Self::set_body_external_torque_struct_typed).
    // ESCAPE_HATCH: parity with raw runner setter `Simulation::set_body_external_torque_struct`.
    pub fn set_body_external_torque_struct(&mut self, body: Entity, torque: DVec3) {
        // allowed: raw-DVec3 escape-hatch boundary mirroring the runner's raw input.
        let typed = Torque::<StructuralFrame<SelfRef>>::from_raw_si(torque);
        self.set_body_external_torque_struct_typed(body, typed);
    }
}

// ---------------------------------------------------------------------
// Shared helpers (mirror `source_mutator::_entity_label` / `_fetch_*`).

/// Format a user-facing label for `entity`. Mirrors
/// `source_mutator::_entity_label`.
fn _entity_label(names: &Query<&'static Name>, entity: Entity) -> String {
    match names.get(entity) {
        Ok(name) => format!("{name} ({entity:?})"),
        Err(_) => format!("{entity:?}"),
    }
}

/// Assert that `body` is a registered dynamic body (carries
/// [`DynamicsConfigC`]). The same misuse → fail-loud pattern
/// [`source_mutator::_fetch_frame_entity`] applies for source-targeted
/// queries — passing a non-body entity (e.g. a gravity source) panics
/// here with a diagnostic naming the misuse.
fn _assert_is_body(
    body_filter: &Query<(), With<DynamicsConfigC>>,
    names: &Query<&'static Name>,
    body: Entity,
    method: &str,
) {
    if body_filter.get(body).is_err() {
        panic!(
            "BodyReader/BodyMutator::{method}: {label} is not a registered dynamic body \
             — it is missing DynamicsConfigC. Spawn the body via \
             `VehicleConfig::spawn_bevy(...)` (which inserts DynamicsConfigC + the \
             per-stage state / load components) before reading or writing its loads. \
             If the entity is a gravity source (planet), pass it to SourceReader / \
             SourceMutator instead — bodies and sources occupy disjoint entity \
             slots and the SystemParam filter is the demarcation.",
            label = _entity_label(names, body),
        );
    }
}
