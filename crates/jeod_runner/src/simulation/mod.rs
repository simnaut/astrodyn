//! `Simulation` runner: standalone driver of the JEOD physics pipeline.
//!
//! The implementation is split across this module's children, each
//! reopening `impl Simulation { ... }`:
//!
//! - [`types`] — supporting structs (`SimBody`, `VehicleOutput`,
//!   `ContactPairConfig`, `SourceFrameIds`, `GravityData`).
//! - [`sources`] — gravity-source registration and accessors.
//! - [`bodies`] — body lifecycle, accessors, setters, contact-pair
//!   registration.
//! - [`mass_tree`] — attach/detach topology, detached-subtree machinery.
//! - [`validate`] — pre-step setup-time validation.
//! - [`step`] — the per-step integration pipeline.
//!
//! Children are private modules; the parent `lib.rs` re-exports the
//! public surface (`Simulation`, `VehicleOutput`, etc.) via `pub use`.

mod bodies;
mod mass_tree;
mod sources;
mod step;
pub(crate) mod types;
mod validate;

pub use jeod_dynamics::DetachedSubtreeState;
pub use jeod_sim::{GroundFacet, SphericalTerrain, Terrain};
pub use types::{ContactPairConfig, GroundContactPairConfig, VehicleOutput};

use std::collections::HashMap;

use jeod_dynamics::MassBodyId;
use jeod_frames::{FrameId, FrameTree, RefFrameKind};
use jeod_sim::atmosphere::AtmosphereConfig;
use jeod_sim::SimulationTime;

use crate::simulation::types::{GravityData, SimBody, SourceFrameIds};

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
/// use jeod_runner::SimulationBuilderExt;
/// use jeod_sim::recipes::Mission;
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
    source_ephem_bodies: Vec<Option<(jeod_sim::EphemerisBody, jeod_sim::EphemerisBody)>>,
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
    pub ephemeris: Option<jeod_sim::Ephemeris>,
    /// Optional mass tree for multi-body vehicles (attach/detach/staging).
    /// Bodies participating in the tree have `SimBody::mass_body_id` set.
    pub mass_tree: Option<jeod_dynamics::MassTree>,
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
    coupled_integ_scratch: jeod_sim::integration::CoupledIntegScratch,
    /// `true` once `step_internal` has run at least once. Used to enforce
    /// JEOD's S_define-level invariant that contact-pair registration is
    /// `P_BODY("initialization")` / `P_DYN("initialization")`-only — i.e.,
    /// runs before integration starts. JEOD enforces this structurally via
    /// Trick job phasing; we mirror it with a runtime guard since our API
    /// surface lets callers invoke `register_*_contact_pair` at any time
    /// (JEOD_INV: IN.38).
    pub(crate) has_stepped: bool,
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
            coupled_integ_scratch: jeod_sim::integration::CoupledIntegScratch::new(),
            has_stepped: false,
        }
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
