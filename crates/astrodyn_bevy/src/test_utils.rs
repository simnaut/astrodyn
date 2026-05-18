//! Shared `App`-construction helpers for in-crate unit tests.
//!
//! Test modules across `astrodyn_bevy` repeatedly open with
//! `App::new(); add_plugins(MinimalPlugins)` (sometimes paired with
//! `AstrodynPlugin`). Funnelling those through a single helper keeps
//! the plugin set audited in one place: any future change to the
//! `MinimalPlugins` / `AstrodynPlugin` composition (e.g. adding a new
//! foundation plugin every test must opt into) propagates without
//! editing eleven call sites by hand.
//!
//! Scope is deliberately narrow: only the two shapes that appear
//! verbatim across the in-crate test modules are exposed. Test
//! modules that prepend custom resource inserts (e.g. a synthetic
//! root-frame entity, a hand-built `Time::<Fixed>` for a specific
//! `dt`) keep open-coding `App::new()` — wrapping those one-offs
//! would force every variant to widen the helper's signature.

use bevy::prelude::*;

use crate::AstrodynPlugin;

/// Bare Bevy `App` with `MinimalPlugins` installed.
///
/// Use this when the test exercises a single system or two in
/// isolation and does not need the full `AstrodynPlugin` schedule
/// graph (resources, system sets, ordering edges).
pub(crate) fn create_minimal_test_app() -> App {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app
}

/// Bevy `App` with `MinimalPlugins` and `AstrodynPlugin` installed.
///
/// Use this when the test needs `AstrodynPlugin`'s schedule graph
/// — resource initialization (`RootFrameEntityR`, message
/// registrations) and the full system-ordering edges over
/// `FixedUpdate` — but does not need any of the broader Bevy
/// `DefaultPlugins` (windowing, rendering, audio).
pub(crate) fn create_astrodyn_test_app() -> App {
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, AstrodynPlugin));
    app
}
