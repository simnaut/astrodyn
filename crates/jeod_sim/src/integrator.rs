//! `jeod_sim`-owned vocabulary for integrator selection and state.
//!
//! These types are the contract between mission crates / the `bevy_jeod`
//! adapter and the integrator family. The wrapping insulates downstream
//! code from internal field / variant renames inside the
//! `jeod_dynamics::integration`, `jeod_dynamics::gauss_jackson`, and
//! `jeod_dynamics::abm4` modules: a rename there only ripples to the
//! delegating `From` / method bodies in this module, never to mission
//! code or to the Bevy adapter's `IntegratorTypeC` /
//! `GaussJacksonStateC` / `Abm4StateC` newtypes.
//!
//! The kernel in [`crate::integration`] still operates on the raw
//! `jeod_dynamics::{GaussJacksonState, Abm4State}` storage internally —
//! integrator runtime state is private scratch, not part of the
//! mission-facing API. The wrappers expose only the methods consumers
//! actually call, plus `inner_mut()` so the kernel can borrow into the
//! raw state across the boundary.

use jeod_dynamics::{
    Abm4State as RawAbm4State, GaussJacksonConfig as RawGaussJacksonConfig,
    GaussJacksonState as RawGaussJacksonState, IntegratorType as RawIntegratorType,
};

/// Integration method selection.
///
/// Mirrors [`jeod_dynamics::IntegratorType`] one-to-one. `jeod_sim`
/// owns this name so a downstream rename inside `jeod_dynamics` does
/// not ripple to mission code.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum IntegratorType {
    /// Classical 4th-order Runge-Kutta (fixed step).
    #[default]
    Rk4,
    /// Runge-Kutta-Fehlberg 4(5) (fixed step, 5th-order result).
    Rkf45,
    /// Gauss-Jackson (Störmer-Cowell) multi-step predictor-corrector.
    ///
    /// Carries a [`GaussJacksonConfig`]; persistent
    /// [`GaussJacksonState`] must be retained externally.
    /// Forward-time only — see [`jeod_dynamics::IntegratorType::GaussJackson`].
    GaussJackson(GaussJacksonConfig),
    /// Adams-Bashforth-Moulton 4th-order (PECE scheme, fixed step).
    ///
    /// Persistent [`Abm4State`] must be retained externally. Translational-
    /// only; 6-DOF is not yet supported.
    Abm4,
}

impl From<IntegratorType> for RawIntegratorType {
    fn from(value: IntegratorType) -> Self {
        match value {
            IntegratorType::Rk4 => RawIntegratorType::Rk4,
            IntegratorType::Rkf45 => RawIntegratorType::Rkf45,
            IntegratorType::GaussJackson(cfg) => RawIntegratorType::GaussJackson(cfg.into()),
            IntegratorType::Abm4 => RawIntegratorType::Abm4,
        }
    }
}

impl From<RawIntegratorType> for IntegratorType {
    fn from(value: RawIntegratorType) -> Self {
        match value {
            RawIntegratorType::Rk4 => IntegratorType::Rk4,
            RawIntegratorType::Rkf45 => IntegratorType::Rkf45,
            RawIntegratorType::GaussJackson(cfg) => IntegratorType::GaussJackson(cfg.into()),
            RawIntegratorType::Abm4 => IntegratorType::Abm4,
        }
    }
}

/// Configuration for the Gauss-Jackson integrator.
///
/// Opaque newtype over [`jeod_dynamics::GaussJacksonConfig`]. Construct
/// via [`Self::default`], [`Self::with_order`], or [`Self::standard`]
/// — the underlying field layout is intentionally not exposed, so a
/// future field rename inside `jeod_dynamics` does not break the
/// mission-facing surface.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct GaussJacksonConfig(RawGaussJacksonConfig);

impl GaussJacksonConfig {
    /// Create a config with fixed order, no step-doubling. Bootstrap
    /// editing still runs; see
    /// [`jeod_dynamics::GaussJacksonConfig::with_order`].
    pub fn with_order(order: usize) -> Self {
        Self(RawGaussJacksonConfig::with_order(order))
    }

    /// JEOD standard configuration.
    /// See [`jeod_dynamics::GaussJacksonConfig::standard`].
    pub fn standard() -> Self {
        Self(RawGaussJacksonConfig::standard())
    }

    /// Non-panicking validation. See
    /// [`jeod_dynamics::GaussJacksonConfig::check`].
    pub fn check(&self) -> Vec<String> {
        self.0.check()
    }

    /// Validate the configuration, panicking on invalid values. See
    /// [`jeod_dynamics::GaussJacksonConfig::validate`].
    pub fn validate(&self) {
        self.0.validate()
    }
}

impl From<GaussJacksonConfig> for RawGaussJacksonConfig {
    #[inline]
    fn from(value: GaussJacksonConfig) -> Self {
        value.0
    }
}

impl From<RawGaussJacksonConfig> for GaussJacksonConfig {
    #[inline]
    fn from(value: RawGaussJacksonConfig) -> Self {
        Self(value)
    }
}

/// Persistent Gauss-Jackson integrator state.
///
/// Opaque newtype over [`jeod_dynamics::GaussJacksonState`]. Only the
/// methods consumers actually call across the boundary are exposed;
/// the remaining surface (history arrays, FSM scratch, primer state)
/// stays inside `jeod_dynamics` where it belongs.
#[derive(Debug, Clone)]
pub struct GaussJacksonState(RawGaussJacksonState);

impl GaussJacksonState {
    /// Create a new Gauss-Jackson integrator with the given configuration.
    /// Delegates to [`jeod_dynamics::GaussJacksonState::new`].
    pub fn new(config: GaussJacksonConfig) -> Self {
        Self(RawGaussJacksonState::new(config.into()))
    }

    /// Reset the integrator to its initial state. Delegates to
    /// [`jeod_dynamics::GaussJacksonState::reset`].
    pub fn reset(&mut self) {
        self.0.reset()
    }

    /// Reset the integrator and clear the topology-dirty flag. Delegates
    /// to [`jeod_dynamics::GaussJacksonState::reset_for_topology_change`].
    pub fn reset_for_topology_change(&mut self) {
        self.0.reset_for_topology_change()
    }

    /// Mark the integrator as carrying stale predictor / corrector
    /// history. Delegates to
    /// [`jeod_dynamics::GaussJacksonState::mark_topology_dirty`].
    pub fn mark_topology_dirty(&mut self) {
        self.0.mark_topology_dirty()
    }

    /// Returns true if the integrator is carrying stale history.
    /// Delegates to
    /// [`jeod_dynamics::GaussJacksonState::is_topology_dirty`].
    pub fn is_topology_dirty(&self) -> bool {
        self.0.is_topology_dirty()
    }

    /// Returns the configuration this integrator was created with.
    /// Returns the wrapped [`GaussJacksonConfig`] (not a reference) —
    /// the type is `Copy`, so this incurs no allocation.
    pub fn config(&self) -> GaussJacksonConfig {
        GaussJacksonConfig(*self.0.config())
    }

    /// Returns true if the integrator is still in the priming phase.
    /// Delegates to [`jeod_dynamics::GaussJacksonState::is_priming`].
    pub fn is_priming(&self) -> bool {
        self.0.is_priming()
    }

    /// Cumulative count of unconverged bootstrap-edit iterations.
    /// Delegates to
    /// [`jeod_dynamics::GaussJacksonState::bootstrap_unconverged_iterations`].
    pub fn bootstrap_unconverged_iterations(&self) -> u32 {
        self.0.bootstrap_unconverged_iterations()
    }

    /// Mutable reference to the wrapped raw state. Used by the
    /// `jeod_sim` integration kernel to pass through to
    /// `jeod_dynamics::abm4_translational_step` / GJ's `integrate`
    /// without copying. Mission code should not need this.
    #[inline]
    pub fn inner_mut(&mut self) -> &mut RawGaussJacksonState {
        &mut self.0
    }

    /// Shared reference to the wrapped raw state. Symmetry partner of
    /// [`Self::inner_mut`].
    #[inline]
    pub fn inner(&self) -> &RawGaussJacksonState {
        &self.0
    }
}

impl From<RawGaussJacksonState> for GaussJacksonState {
    #[inline]
    fn from(value: RawGaussJacksonState) -> Self {
        Self(value)
    }
}

impl From<GaussJacksonState> for RawGaussJacksonState {
    #[inline]
    fn from(value: GaussJacksonState) -> Self {
        value.0
    }
}

/// Persistent Adams-Bashforth-Moulton 4 integrator state.
///
/// Opaque newtype over [`jeod_dynamics::Abm4State`]. Only the methods
/// consumers actually call across the boundary are exposed; the
/// internal sliding-window history stays in `jeod_dynamics`.
#[derive(Debug, Clone, Default)]
pub struct Abm4State(RawAbm4State);

impl Abm4State {
    /// Create a fresh, unprimed integrator state.
    /// Delegates to [`jeod_dynamics::Abm4State::new`].
    pub fn new() -> Self {
        Self(RawAbm4State::new())
    }

    /// Reset the integrator back to its unprimed state.
    /// Delegates to [`jeod_dynamics::Abm4State::reset`].
    pub fn reset(&mut self) {
        self.0.reset()
    }

    /// Reset the integrator and clear the topology-dirty flag.
    /// Delegates to [`jeod_dynamics::Abm4State::reset_for_topology_change`].
    pub fn reset_for_topology_change(&mut self) {
        self.0.reset_for_topology_change()
    }

    /// Mark the integrator as carrying stale predictor history.
    /// Delegates to [`jeod_dynamics::Abm4State::mark_topology_dirty`].
    pub fn mark_topology_dirty(&mut self) {
        self.0.mark_topology_dirty()
    }

    /// Returns true if the integrator is carrying stale history.
    /// Delegates to [`jeod_dynamics::Abm4State::is_topology_dirty`].
    pub fn is_topology_dirty(&self) -> bool {
        self.0.is_topology_dirty()
    }

    /// Returns true while the integrator is still priming with RK4.
    /// Delegates to [`jeod_dynamics::Abm4State::is_priming`].
    pub fn is_priming(&self) -> bool {
        self.0.is_priming()
    }

    /// Mutable reference to the wrapped raw state. Used by the
    /// `jeod_sim` integration kernel to pass through to
    /// `jeod_dynamics::abm4_translational_step` without copying.
    /// Mission code should not need this.
    #[inline]
    pub fn inner_mut(&mut self) -> &mut RawAbm4State {
        &mut self.0
    }

    /// Shared reference to the wrapped raw state. Symmetry partner of
    /// [`Self::inner_mut`].
    #[inline]
    pub fn inner(&self) -> &RawAbm4State {
        &self.0
    }
}

impl From<RawAbm4State> for Abm4State {
    #[inline]
    fn from(value: RawAbm4State) -> Self {
        Self(value)
    }
}

impl From<Abm4State> for RawAbm4State {
    #[inline]
    fn from(value: Abm4State) -> Self {
        value.0
    }
}
