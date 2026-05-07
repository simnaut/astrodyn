//! Bevy `Component` newtypes wrapping `astrodyn` typed siblings (state,
//! mass, gravity controls, interactions, derived states).
//!
//! `Reflect` derives are deferred until inspector / scene-tooling adoption
//! demands them; absent that bound, components can carry `<P: Planet>`
//! generics directly without the `TypePath` constraint that previously
//! forced a `<SelfPlanet>` wildcard at the storage layer.
//!
//! JEOD_INV: TS.01 — this module is a per-entity storage boundary. The
//! `<SelfRef>` and `<SelfPlanet>` wildcard tags on Bevy `Component`,
//! `Message`, and runner-state field types in this module (and its
//! per-stage submodules) are the canonical sites where runtime-resolved
//! entity identity meets the compile-time phantom-frame discipline. All
//! system code paths and `astrodyn_*` / `astrodyn` APIs use `<V: Vehicle>` /
//! `<P: Planet>` parameters; the wildcards are confined to this storage
//! layer. See `tests/self_ref_self_planet_discipline.rs` for the lint
//! that enforces the rule across the workspace.
//!
//! # Submodule layout
//!
//! Components are grouped along the same axis as
//! [`crate::systems`]: each pipeline-stage module
//! co-locates the components its systems read and write. Glue
//! components (frame-tree, joint, mass-tree, frame-attach) live in
//! their own modules.
//!
//! - `state` — translational / rotational / mass / accumulators /
//!   integrator state / external loads.
//! - `gravity` — gravity sources, planet shape / rotation,
//!   ephemeris config, gravity-gradient torque, tidal config / output.
//! - `interaction` — drag, SRP, atmosphere, shadow.
//! - `integration` — integration-frame source and frame-switch config.
//! - `derived` — orbital elements, Euler angles, LVLH, geodetic,
//!   solar beta, Earth lighting (config + outputs).
//! - `frame_tree` — frames-as-entities translation / rotation /
//!   angular-velocity components and frame-kind markers / handles.
//! - `joint` — kinematic-joint spec components.
//! - `mass_tree` — mass-tree edges, kinematic gating, attach /
//!   detach messages.
//! - `frame_attach` — non-body frame attachment + detached-subtree
//!   propagation.
//!
//! Submodules are private to keep the per-stage names from colliding
//! with the same names in [`crate::systems`] under
//! `pub use components::*` / `pub use systems::*` glob exports. Every
//! item from each submodule is re-exported here so existing
//! `astrodyn_bevy::components::<Name>` and `crate::components::*` paths
//! continue to resolve.

mod derived;
mod frame_attach;
mod frame_tree;
mod gravity;
mod integration;
mod interaction;
mod joint;
mod mass_tree;
mod state;

pub use derived::*;
pub use frame_attach::*;
pub use frame_tree::*;
pub use gravity::*;
pub use integration::*;
pub use interaction::*;
pub use joint::*;
pub use mass_tree::*;
pub use state::*;
