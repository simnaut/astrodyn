//! [`AstrodynAppExt`] — ergonomic [`App`] setup and fixed-step advancement.
//!
//! Mission code, examples, and tests routinely re-type the same five-line
//! boilerplate to bring up an [`App`] with [`AstrodynPlugin`]:
//!
//! ```ignore
//! let mut app = App::new();
//! app.add_plugins(MinimalPlugins);
//! app.insert_resource(Time::<Fixed>::from_seconds(dt));
//! app.add_plugins(AstrodynPlugin);
//! ```
//!
//! followed by a four-line block to advance `Time<Fixed>` and run the
//! `FixedUpdate` schedule. The advance pattern is a footgun: forgetting
//! `run_schedule(FixedUpdate)` after `advance_by(...)` silently does nothing,
//! so a test reads stale state and asserts against it.
//!
//! [`AstrodynAppExt`] collapses both shapes onto a single canonical surface
//! that mirrors the proven crate-private helpers in
//! `astrodyn_verif_parity::tests::common`. The order in which `add_astrodyn`
//! installs `Time<Fixed>` and the plugin is the same order
//! [`crate::scenario::SimulationBuilderBevyExt::populate_app`] uses — the
//! plugin must observe `Time<Fixed>` already in the world.

use std::time::Duration;

use bevy::prelude::*;

use crate::{AstrodynPlugin, IntegrationDtR};

/// Ergonomic helpers for setting up and advancing an [`App`] running
/// [`AstrodynPlugin`] in `FixedUpdate`.
///
/// Implemented for [`App`]. The methods chain (each returns `&mut Self`) so
/// callers can write
///
/// ```ignore
/// App::new()
///     .add_plugins(MinimalPlugins)
///     .add_astrodyn(10.0)
///     .add_systems(Startup, setup);
/// ```
///
/// instead of the four-statement bring-up block.
pub trait AstrodynAppExt {
    /// Insert `Time::<Fixed>::from_seconds(dt_seconds)` then add
    /// [`AstrodynPlugin`].
    ///
    /// Insertion order is load-bearing: [`AstrodynPlugin::build`] expects
    /// `Time<Fixed>` to already be in the world (the schedule's shared `dt`
    /// drives every JEOD pipeline stage). Mirrors the ordering used by
    /// `populate_app` in `crate::scenario`. A naive impl that adds the
    /// plugin first would re-introduce the same precondition footgun the
    /// trait exists to abolish.
    ///
    /// Idempotent on the plugin: a second call is a no-op for the plugin
    /// half (Bevy panics on duplicate plugin adds, so this method does not
    /// re-add). The `Time<Fixed>` resource is overwritten on each call —
    /// the typical usage is a single bring-up, but a test that wants to
    /// reconfigure `dt` mid-test can re-call this method without standing
    /// up a new `App`.
    fn add_astrodyn(&mut self, dt_seconds: f64) -> &mut Self;

    /// Advance `Time<Fixed>` by `dt_seconds` and run the `FixedUpdate`
    /// schedule, repeated `n` times.
    ///
    /// Each iteration corresponds to one JEOD-pipeline tick (one pass
    /// through the seven `AstrodynSet` stages). The advance and schedule
    /// run are bonded as a single unit so callers cannot accidentally
    /// advance the clock without running the pipeline (or vice versa).
    fn step_fixed_dt(&mut self, n: usize, dt_seconds: f64) -> &mut Self;

    /// Convenience: read `dt` from the existing [`IntegrationDtR`] resource
    /// and invoke [`Self::step_fixed_dt`] with it.
    ///
    /// `IntegrationDtR` is the bit-exact `f64` source the pipeline reads.
    /// Reading from `Time<Fixed>::timestep()` instead would round through
    /// `Duration` (integer nanoseconds), reintroducing the precision loss
    /// the resource exists to avoid for non-nanosecond-representable
    /// timesteps.
    ///
    /// # Panics
    ///
    /// Panics if [`IntegrationDtR`] is absent from the world. Callers
    /// must install it first via [`Self::add_astrodyn`] (or a manual
    /// `insert_resource(IntegrationDtR(...))` before running the
    /// pipeline). A missing `IntegrationDtR` means the pipeline never
    /// advanced under a known `dt`, so silently picking a default would
    /// propagate physics under the wrong integrator step — exactly the
    /// silent-wrong-physics failure mode the fail-loud rule forbids.
    fn step_fixed(&mut self, n: usize) -> &mut Self;
}

impl AstrodynAppExt for App {
    fn add_astrodyn(&mut self, dt_seconds: f64) -> &mut Self {
        // Insert `Time<Fixed>` *before* the plugin so `AstrodynPlugin::build`
        // sees a populated resource. `populate_app` in `crate::scenario`
        // documents the same ordering for the same reason.
        // allowed: `Time::<Fixed>::from_seconds` is Bevy's own constructor
        // for the `Time<Fixed>` resource, not the banned typed-quantity
        // `SecondsSince::from_seconds` bypass. The grep pattern catches
        // `from_seconds` indiscriminately; the argument is a plain `f64`
        // integrator timestep, not a typed-duration phantom.
        self.insert_resource(Time::<Fixed>::from_seconds(dt_seconds));
        // `IntegrationDtR` is the bit-exact f64 source of `dt` for the
        // pipeline; `Time<Fixed>` is kept in sync for `FixedUpdate`
        // gating but its `Duration`-rounded delta is no longer the
        // physics source. See `IntegrationDtR` doc.
        self.insert_resource(IntegrationDtR(dt_seconds));
        if !self.is_plugin_added::<AstrodynPlugin>() {
            self.add_plugins(AstrodynPlugin);
        }
        self
    }

    fn step_fixed_dt(&mut self, n: usize, dt_seconds: f64) -> &mut Self {
        let dur = Duration::from_secs_f64(dt_seconds);
        // Keep `IntegrationDtR` in sync each call so callers that
        // bypass `add_astrodyn` (or that vary `dt` between calls) still
        // drive the pipeline with the exact f64 they passed in.
        self.insert_resource(IntegrationDtR(dt_seconds));
        for _ in 0..n {
            self.world_mut()
                .resource_mut::<Time<Fixed>>()
                .advance_by(dur);
            self.world_mut().run_schedule(FixedUpdate);
        }
        self
    }

    fn step_fixed(&mut self, n: usize) -> &mut Self {
        // Read from `IntegrationDtR`, not `Time<Fixed>::timestep()` —
        // the latter rounds through `Duration` (integer ns) and would
        // re-introduce the precision loss for non-nanosecond-representable
        // dts that `IntegrationDtR` exists to avoid.
        let dt_seconds = self
            .world()
            .get_resource::<IntegrationDtR>()
            .unwrap_or_else(|| {
                panic!(
                    "AstrodynAppExt::step_fixed: `IntegrationDtR` resource is missing from \
                     the world. The convenience overload reads `dt` from `IntegrationDtR` \
                     to drive the pipeline with the bit-exact f64 it was configured for. \
                     Call `app.add_astrodyn(dt_seconds)` first (which installs both \
                     `Time<Fixed>` and `IntegrationDtR` then adds `AstrodynPlugin`), or \
                     use `step_fixed_dt(n, dt_seconds)` to pass the step explicitly."
                )
            })
            .0;
        self.step_fixed_dt(n, dt_seconds)
    }
}
