//! `Simulation` runner: standalone driver of the JEOD physics pipeline.
//!
//! The implementation is split across this module's children, each
//! reopening `impl Simulation { ... }`:
//!
//! - [`types`] — supporting structs (`SimBody`, `VehicleOutput`,
//!   `ContactPairConfig`, `GravityData`).
//!   `SourceFrameIds` lives in `astrodyn::source_frames` (issue #71).
//! - [`sources`] — gravity-source registration and accessors.
//! - [`bodies`] — body lifecycle, accessors, setters, contact-pair
//!   registration.
//! - [`mass_tree`] — attach/detach topology, detached-subtree machinery.
//! - [`frame_attach`] — `attach_to_frame` / `detach_from_frame` for
//!   non-body reference-frame attachment (port of JEOD's
//!   `DynBody::attach_to_frame`; the kinematic-only side of frame-tree
//!   attachment).
//! - [`validate`] — pre-step setup-time validation.
//! - [`step`] — the per-step integration pipeline.
//!
//! Children are private modules; the parent `lib.rs` re-exports the
//! public surface (`Simulation`, `VehicleOutput`, etc.) via `pub use`.

mod bodies;
mod frame_attach;
mod mass_tree;
mod sources;
mod step;
pub(crate) mod types;
mod validate;

pub use astrodyn::{DetachedSubtreeState, GroundFacet, SphericalTerrain, Terrain};
pub use types::{ContactPairConfig, FrameAttachState, GroundContactPairConfig, VehicleOutput};

use std::collections::HashMap;

use glam::DMat3;

use astrodyn::atmosphere::AtmosphereConfig;
use astrodyn::{FrameId, FrameTree, MassBodyId, RefFrameKind, SimulationTime, SourceFrameIds};

use crate::simulation::types::{GravityData, SimBody};

/// Cached slowly-varying components of the Earth RNP composition when
/// [`Simulation::earth_rnp_refresh_cadence_s`] is non-zero. JEOD's
/// `Earth_GGM05C_SimObject` (`Base/earth_GGM05C_baseline.sm:114, 119`)
/// schedules two distinct RNP jobs:
///
/// - `(LOW_RATE_ENV, "environment") rnp.update_rnp(...)` — refreshes
///   precession, nutation, polar motion at the configured cadence
///   (60 s for SIM_dyncomp's `LOW_RATE_ENV`).
/// - `P_ENV("derivative") rnp.update_axial_rotation(gmst)` — refreshes
///   the GAST rotation at every derivative call, using the *cached*
///   nutation `equa_of_equi` term and the current `gmst_seconds`.
///
/// `propagate_rnp` then composes the cached precession/nutation/polar
/// motion with the freshly-computed GAST rotation into the final
/// `T_parent_this`. The runner mirrors that split: the cache stores
/// the polar-motion-by-NP product (`pm^T × NP` or `NP` when polar
/// motion is disabled) plus the cached `equa_of_equi`, and the
/// per-step path recomputes GAST + final composition from the current
/// `gmst_seconds`. Holding the full inertial → pfix matrix constant
/// across 60 s would freeze the planet's diurnal rotation between
/// refreshes (≈ 30 m position drift per 60 s at LEO altitude); caching
/// only the non-diurnal terms preserves the per-step diurnal sample
/// while still matching JEOD's lower-cadence precession/nutation/polar
/// motion sampling.
///
/// Production runs default to per-step refresh
/// (`earth_rnp_refresh_cadence_s == 0.0`); cross-validation runs that
/// need to match a JEOD verif sim's lower-rate cadence call
/// [`Simulation::set_earth_rnp_refresh_cadence`] before stepping.
#[derive(Debug, Clone, Copy)]
pub(crate) struct EarthRnpCache {
    /// `simtime` (seconds since the start of the simulation) at which
    /// the slowly-varying components were last refreshed. Used to gate
    /// the next refresh against `earth_rnp_refresh_cadence_s`.
    pub(crate) simtime_at_refresh: f64,
    /// Cached `NP = nutation^T × precession^T`. Composed with the
    /// per-step GAST rotation (and optional polar motion) to produce
    /// the final `T_parent_this`. The composition order is
    /// `T = polar^T × gast^T × NP` (or `T = gast^T × NP` when polar
    /// motion is disabled), matching `compute_t_parent_this_with_polar`.
    pub(crate) np: DMat3,
    /// Cached `polar^T` matrix, present iff `polar_motion = Some(...)`
    /// at the time of the refresh. The per-step recomposition reuses
    /// this matrix without recomputing the polar-motion trig — a
    /// stationary value over the cadence window since `polar_motion`
    /// itself only changes at simulation-config boundaries.
    pub(crate) polar_t: Option<DMat3>,
    /// Cached nutation `equa_of_equi` (equation of the equinoxes), in
    /// the same units `gast_rotation_matrix` expects. Refreshed
    /// alongside `np`; consumed by every per-step GAST recomputation.
    pub(crate) equa_of_equi: f64,
}

// ══════════════════════════════════════════════════════════════════════════════
// Simulation
// ══════════════════════════════════════════════════════════════════════════════

/// ECS-agnostic simulation runner.
///
/// Owns all simulation state and runs the JEOD pipeline in `step()`.
/// This is the **non-ECS** path — ECS adapters should call the per-body
/// functions (`accumulate_gravity`, `evaluate_atmosphere`, etc.) directly
/// from their system functions.
///
/// # Public API conventions
///
/// The methods on `Simulation` group into four families:
///
/// - **Source registry** (`add_source`, `set_source_*`, `source_*`) — gravity
///   sources, ephemeris, planet rotation, tides.
/// - **Body registry** (`add_body`, `body`, `set_body_*`) — dynamic vehicles.
/// - **Mass tree** (`add_body_to_tree`, `attach`, `detach`,
///   `sync_body_mass_from_tree`) — multi-body composites.
/// - **Lifecycle** (`validate`, `step`, `step_n`, `step_until`, `set_dt`,
///   `elapsed`).
///
/// # Error handling
///
/// Validation can happen either at construction time or after mutation.
/// Builder-based constructors that validate configuration
/// (`Simulation::from_builder` and `SimulationBuilderExt::build`) return
/// `Result`. After construction, `validate()` is the only `Result`-returning
/// instance method on `Simulation`; it batches configuration errors before
/// stepping. Other runtime methods panic on misuse with a method-name-prefixed
/// message (e.g. `"set_source_position: source index 7 out of range"`).
/// Out-of-range indices, configuration conflicts, and numerical
/// preconditions are programmer errors, not runtime conditions.
///
/// # Example
/// ```
/// use astrodyn_runner::SimulationBuilderExt;
/// use astrodyn::recipes::Mission;
///
/// let mut sim = Mission::iss_leo().into_builder().build().unwrap();
/// sim.step_n(10);
/// let output = sim.body(0);
/// assert!(output.trans.position.length() > 6_000_000.0);
/// ```
pub struct Simulation {
    /// Simulation time (TAI, UTC, TDB, GMST, etc.).
    pub time: SimulationTime,
    /// Dynamic bodies (internal, private).
    // JEOD_INV: DS.01 — private to prevent runtime mutation of derived-state config
    bodies: Vec<SimBody>,
    /// Reference frame tree — single source of truth for celestial body positions,
    /// velocities, and rotations. Updated each step from ephemeris data.
    /// Private to protect invariants; use [`frame_tree()`](Self::frame_tree) for
    /// read-only access.
    frame_tree: FrameTree,
    /// Root inertial frame ID for this simulation. This is the integration-origin
    /// frame to which all positions are relative, and it is not necessarily
    /// `Earth.inertial` (for example, it may be renamed to match the configured
    /// central body, such as `Mars.inertial`).
    pub root_frame_id: FrameId,
    /// Per-source frame tree node IDs (parallel to `gravity_data`).
    source_frame_ids: Vec<SourceFrameIds>,
    /// Per-source gravity model data (parallel to `source_frame_ids`).
    gravity_data: Vec<GravityData>,
    /// Per-source ephemeris body mapping (parallel to `source_frame_ids`).
    source_ephem_bodies: Vec<Option<(astrodyn::EphemerisBody, astrodyn::EphemerisBody)>>,
    /// Atmosphere configuration. `None` disables atmosphere for all bodies.
    pub atmosphere: Option<AtmosphereConfig>,
    /// Source index for the planet whose rotation is used for atmosphere.
    pub atmosphere_planet_source: Option<usize>,
    /// Source index for the Sun (used by SRP and earth lighting).
    pub sun_source: Option<usize>,
    /// Source index for the Moon (used by earth lighting).
    pub moon_source: Option<usize>,
    /// Polar motion parameters (xp, yp) in radians. When `Some`, the RNP
    /// composition includes polar motion: W(xp,yp) × R(GAST) × N × P.
    /// When `None`, polar motion is omitted (matches JEOD `enable_polar=false`).
    pub polar_motion: Option<(f64, f64)>,
    /// Integration timestep (seconds).
    pub dt: f64,
    /// Optional ephemeris for per-step source position updates.
    pub ephemeris: Option<astrodyn::Ephemeris>,
    /// Optional mass tree for multi-body vehicles (attach/detach/staging).
    /// Bodies participating in the tree have `SimBody::mass_body_id` set.
    ///
    /// # IG.37 — Topology-change safety
    ///
    /// **Prefer the high-level methods** ([`attach`](Self::attach),
    /// [`detach`](Self::detach), [`detach_subtree`](Self::detach_subtree),
    /// [`attach_subtree_aligned`](Self::attach_subtree_aligned)) for
    /// mid-flight topology changes — they call `mark_topology_dirty()` +
    /// [`astrodyn::reset_integrators`] on every affected body, matching
    /// JEOD's `dyn_body_attach.cc::reset_integrators()` semantics.
    ///
    /// If you reach in here with `as_mut()` and call `tree.attach` /
    /// `tree.detach` directly, you **must** call
    /// [`sync_body_mass_from_tree`](Self::sync_body_mass_from_tree) for
    /// each affected Simulation body afterward — that is, the directly
    /// mutated child *plus every ancestor of the (former) parent that
    /// is registered as a Simulation body*, since `MassTree::attach`
    /// and `MassTree::detach` propagate composite mass updates up the
    /// full ancestor chain. Syncing only the child or only the
    /// immediate parent leaves higher ancestors with stale composite
    /// mass and stale multi-step integrator history. That method also
    /// resets the multi-step integrator (Gauss-Jackson, ABM4)
    /// predictor/corrector history; skipping it leaves stale history
    /// that produces wrong physics with no panic. Single-step
    /// integrators (RK4, RKF4(5)) carry no per-step history, so the
    /// reset is a no-op for them — but `sync_body_mass_from_tree` is
    /// still the correct sync site regardless of integrator choice.
    pub mass_tree: Option<astrodyn::MassTree>,
    /// Composite-body inertial state of free-flying mass-tree subtrees
    /// that have been detached from the integrated body's tree but not
    /// yet re-attached. Populated by [`detach_subtree`](Self::detach_subtree),
    /// propagated each step by [`step_detached_subtrees`](Self::step_detached_subtrees)
    /// (called from `step`), consumed by
    /// [`attach_subtree_aligned`](Self::attach_subtree_aligned). Tree-only
    /// subtrees never become Simulation bodies — only their composite-CoM
    /// state is tracked, sufficient for JEOD's `attach_child`
    /// momentum-conservation algorithm to combine them back in.
    pub detached_subtrees: HashMap<MassBodyId, DetachedSubtreeState>,
    /// Registered contact pairs (inter-body spring-damper contact).
    ///
    /// When non-empty, `step_internal` uses a multi-body coupled RK4 path in
    /// which contact forces are recomputed at each of the 4 RK4 stages with
    /// each pair's intermediate states. This matches JEOD's `check_contact()`
    /// derivative-class job. Only RK4 + 6-DOF is supported; adding a pair
    /// while a body uses non-RK4 or 3-DOF is a validation error.
    contact_pairs: Vec<ContactPairConfig>,
    /// Registered ground-contact pairs (vehicle-vs-planet-surface).
    ///
    /// Same coupled-RK4 path as `contact_pairs`: when non-empty,
    /// `step_internal` evaluates ground contact at every RK4 stage. Per
    /// JEOD `SIM_ground_contact/S_modules/contact.sm`, `check_contact_ground()`
    /// is a derivative-class job alongside `check_contact()`.
    ground_contact_pairs: Vec<GroundContactPairConfig>,
    /// Source index for the planet whose pfix rotation is used to query
    /// ground-contact terrain. `None` when no ground-contact pairs are
    /// registered (or all use [`SphericalTerrain`], for which pfix
    /// rotation cancels and identity may be passed).
    ground_contact_planet_source: Option<usize>,
    /// Preallocated scratch buffers for the coupled RK4 integrator. Retained
    /// across steps so the inner loop is allocation-free once the body count
    /// stabilizes.
    coupled_integ_scratch: astrodyn::integration::CoupledIntegScratch,
    /// `true` once `step_internal` has run at least once. Used by
    /// `register_contact_pair` / `register_ground_contact_pair` to reject
    /// late registration: the runner stashes a `Phase::Initialization`
    /// impulse in `pending_initial_impulse` against `t=0` body state at
    /// registration time, so a mid-run call would inject a spurious
    /// impulse independent of vehicle altitude. JEOD itself does not
    /// enforce this — `Contact::register_contact` is a public method
    /// with no init-state guard — but JEOD's `SIM_ground_contact` S_define
    /// wires registration only at `P_BODY/P_DYN("initialization")`
    /// (`sv_dyn.sm:130-133`, `contact.sm:70-72`), which is consistent
    /// with the constraint. This guard is a port-specific safety check,
    /// not a JEOD invariant.
    pub(crate) has_stepped: bool,
    /// Refresh cadence for the Earth RNP rotation matrix, in simulated
    /// seconds. `0.0` (default) refreshes every step — the production
    /// configuration. A positive value reuses the cached matrix until
    /// `cadence` simulated seconds have elapsed since the last refresh,
    /// matching Trick's `(LOW_RATE_ENV, "environment")` job class on
    /// `Earth_GGM05C_SimObject::rnp.update_rnp`
    /// (`Base/earth_GGM05C_baseline.sm:114`).
    ///
    /// Set via [`Self::set_earth_rnp_refresh_cadence`]. The cadence is
    /// per-`Simulation`, not global; multiple simulations in the same
    /// process can have different cadences.
    pub(crate) earth_rnp_refresh_cadence_s: f64,
    /// Cached Earth RNP matrix when
    /// [`Self::earth_rnp_refresh_cadence_s`] is non-zero. `None` before
    /// the first step and whenever cadence is zero (the per-step path
    /// recomputes unconditionally). Refreshed inside `update_ephemeris`
    /// when the elapsed time since `simtime_at_refresh` reaches the
    /// configured cadence.
    pub(crate) earth_rnp_cache: Option<EarthRnpCache>,
}

impl Simulation {
    /// Create a new simulation with the given initial time and timestep.
    ///
    /// Creates a frame tree whose root is initially named "Earth.inertial"
    /// and may be renamed when a central source is added via
    /// [`Self::add_source`].
    /// All positions are relative to this root frame regardless of its name.
    pub fn new(time: SimulationTime, dt: f64) -> Self {
        let mut frame_tree = FrameTree::new();
        let root_frame_id = frame_tree.add_root("Earth.inertial".into(), RefFrameKind::Inertial);
        Self {
            time,
            bodies: Vec::new(),
            frame_tree,
            root_frame_id,
            source_frame_ids: Vec::new(),
            gravity_data: Vec::new(),
            source_ephem_bodies: Vec::new(),
            atmosphere: None,
            atmosphere_planet_source: None,
            sun_source: None,
            moon_source: None,
            polar_motion: None,
            dt,
            ephemeris: None,
            mass_tree: None,
            detached_subtrees: HashMap::new(),
            contact_pairs: Vec::new(),
            ground_contact_pairs: Vec::new(),
            ground_contact_planet_source: None,
            coupled_integ_scratch: astrodyn::integration::CoupledIntegScratch::new(),
            has_stepped: false,
            earth_rnp_refresh_cadence_s: 0.0,
            earth_rnp_cache: None,
        }
    }

    /// Set the Earth RNP rotation-matrix refresh cadence, in simulated
    /// seconds.
    ///
    /// The default (`0.0`) refreshes the inertial → pfix rotation every
    /// dynamics step — the production configuration: it preserves
    /// continuous accuracy as the body propagates and matches what
    /// Trick's `P_ENV("derivative") rnp.update_axial_rotation` would
    /// give if it shared the rotation slot directly.
    ///
    /// A positive cadence makes `update_ephemeris` reuse the cached
    /// matrix until the configured number of simulated seconds have
    /// elapsed since the last refresh, mirroring Trick's
    /// `(LOW_RATE_ENV, "environment")` job class on
    /// `Earth_GGM05C_SimObject::rnp.update_rnp`
    /// (`Base/earth_GGM05C_baseline.sm:114`). Use this only for
    /// cross-validation runs that need bit-for-bit alignment with a
    /// JEOD verif sim's lower-cadence RNP refresh.
    ///
    /// The cadence is per-`Simulation`. Calling this method clears the
    /// existing cache so the next step refreshes from scratch.
    ///
    /// # Panics
    /// Panics if `cadence_s` is not finite or is negative.
    // JEOD_INV: PF.06 — RNP refresh cadence configurable per simulation
    pub fn set_earth_rnp_refresh_cadence(&mut self, cadence_s: f64) -> &mut Self {
        assert!(
            cadence_s.is_finite() && cadence_s >= 0.0,
            "set_earth_rnp_refresh_cadence: cadence must be finite and >= 0, got {cadence_s}. \
             Use 0.0 for per-step refresh (default) or a positive value to reuse the cached \
             matrix across that many simulated seconds."
        );
        self.earth_rnp_refresh_cadence_s = cadence_s;
        // Drop the prior cache so the next step computes a fresh matrix
        // at the new cadence — switching between cadences mid-run would
        // otherwise serve a stale matrix that was computed against the
        // previous cadence schedule.
        self.earth_rnp_cache = None;
        self
    }

    /// Read back the Earth RNP refresh cadence in simulated seconds.
    ///
    /// `0.0` is the default (per-step refresh). See
    /// [`Self::set_earth_rnp_refresh_cadence`] for the semantics.
    pub fn earth_rnp_refresh_cadence(&self) -> f64 {
        self.earth_rnp_refresh_cadence_s
    }

    /// Number of bodies in the simulation.
    pub fn num_bodies(&self) -> usize {
        self.bodies.len()
    }

    /// Set the integration timestep (must be positive).
    ///
    /// For JEOD-style time reversal, use `sim.time.time_scale_factor = -1.0`
    /// instead of negative dt. This keeps `simtime` monotonically increasing
    /// while reversing dynamic time (TAI, TDB, etc.) and integration direction.
    ///
    /// # Panics
    /// Panics if `dt` is not finite or not positive.
    pub fn set_dt(&mut self, dt: f64) {
        assert!(
            dt.is_finite() && dt > 0.0,
            "dt must be finite and > 0, got {dt}"
        );
        self.dt = dt;
    }

    /// Current simulation elapsed time in seconds.
    pub fn elapsed(&self) -> f64 {
        self.time.simtime
    }
}
