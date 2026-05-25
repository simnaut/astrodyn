//! `astrodyn`-owned vocabulary for integrator selection and state.
//!
//! These types are the contract between mission crates / the `astrodyn_bevy`
//! adapter and the integrator family. The wrapping insulates downstream
//! code from internal field / variant renames inside the
//! `astrodyn_dynamics::integration`, `astrodyn_dynamics::gauss_jackson`, and
//! `astrodyn_dynamics::abm4` modules: a rename there only ripples to the
//! delegating `From` / method bodies in this module, never to mission
//! code or to the Bevy adapter's `IntegratorTypeC` /
//! `GaussJacksonStateC` / `Abm4StateC` newtypes.
//!
//! The kernel in [`crate::integration`] still operates on the raw
//! `astrodyn_dynamics::{GaussJacksonState, Abm4State}` storage internally —
//! integrator runtime state is private scratch, not part of the
//! mission-facing API. The wrappers expose only the methods consumers
//! actually call, plus `inner_mut()` so the kernel can borrow into the
//! raw state across the boundary.

use astrodyn_dynamics::{
    Abm4State as RawAbm4State, GaussJacksonConfig as RawGaussJacksonConfig,
    GaussJacksonState as RawGaussJacksonState, IntegratorType as RawIntegratorType,
    LsodeConfig as RawLsodeConfig, LsodeState as RawLsodeState,
};

/// Integration method selection.
///
/// Mirrors [`astrodyn_dynamics::IntegratorType`] one-to-one. `astrodyn`
/// owns this name so a downstream rename inside `astrodyn_dynamics` does
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
    /// Forward-time only — see [`astrodyn_dynamics::IntegratorType::GaussJackson`].
    GaussJackson(GaussJacksonConfig),
    /// Adams-Bashforth-Moulton 4th-order (PECE scheme, fixed step).
    ///
    /// Persistent [`Abm4State`] must be retained externally. Translational-
    /// only; 6-DOF is not yet supported.
    Abm4,
    /// LSODE (Livermore Solver) — variable-order, variable-step Nordsieck
    /// multistep. Carries an [`LsodeConfig`]; persistent [`LsodeState`] must
    /// be retained externally. Forward-time only; translational-only.
    Lsode(LsodeConfig),
}

impl From<IntegratorType> for RawIntegratorType {
    fn from(value: IntegratorType) -> Self {
        match value {
            IntegratorType::Rk4 => RawIntegratorType::Rk4,
            IntegratorType::Rkf45 => RawIntegratorType::Rkf45,
            IntegratorType::GaussJackson(cfg) => RawIntegratorType::GaussJackson(cfg.into()),
            IntegratorType::Abm4 => RawIntegratorType::Abm4,
            IntegratorType::Lsode(cfg) => RawIntegratorType::Lsode(cfg.into()),
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
            RawIntegratorType::Lsode(cfg) => IntegratorType::Lsode(cfg.into()),
        }
    }
}

/// Configuration for the Gauss-Jackson integrator.
///
/// Opaque newtype over [`astrodyn_dynamics::GaussJacksonConfig`]. Construct
/// via [`Self::default`], [`Self::with_order`], or [`Self::standard`]
/// — the underlying field layout is intentionally not exposed, so a
/// future field rename inside `astrodyn_dynamics` does not break the
/// mission-facing surface.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct GaussJacksonConfig(RawGaussJacksonConfig);

impl GaussJacksonConfig {
    /// Create a config with fixed order, no step-doubling. Bootstrap
    /// editing still runs; see
    /// [`astrodyn_dynamics::GaussJacksonConfig::with_order`].
    pub fn with_order(order: usize) -> Self {
        Self(RawGaussJacksonConfig::with_order(order))
    }

    /// JEOD standard configuration.
    /// See [`astrodyn_dynamics::GaussJacksonConfig::standard`].
    pub fn standard() -> Self {
        Self(RawGaussJacksonConfig::standard())
    }

    /// Non-panicking validation. See
    /// [`astrodyn_dynamics::GaussJacksonConfig::check`].
    pub fn check(&self) -> Vec<String> {
        self.0.check()
    }

    /// Validate the configuration, panicking on invalid values. See
    /// [`astrodyn_dynamics::GaussJacksonConfig::validate`].
    pub fn validate(&self) {
        self.0.validate()
    }

    /// Opt in to JEOD-faithful warn-and-continue on corrector or
    /// bootstrap non-convergence. See
    /// [`astrodyn_dynamics::GaussJacksonConfig::allow_non_convergence`]
    /// for the full rationale — the short version is that the default
    /// (`false`) panics on a non-converged step, and setting this to
    /// `true` restores JEOD's behavior of logging a warning and
    /// continuing with a degraded position. Use only for matching JEOD
    /// reference runs exactly.
    pub fn with_allow_non_convergence(mut self, allow: bool) -> Self {
        self.0.allow_non_convergence = allow;
        self
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
/// Opaque newtype over [`astrodyn_dynamics::GaussJacksonState`]. Only the
/// methods consumers actually call across the boundary are exposed;
/// the remaining surface (history arrays, FSM scratch, primer state)
/// stays inside `astrodyn_dynamics` where it belongs.
#[derive(Debug, Clone)]
pub struct GaussJacksonState(RawGaussJacksonState);

impl GaussJacksonState {
    /// Create a new Gauss-Jackson integrator with the given configuration.
    /// Delegates to [`astrodyn_dynamics::GaussJacksonState::new`].
    pub fn new(config: GaussJacksonConfig) -> Self {
        Self(RawGaussJacksonState::new(config.into()))
    }

    /// Reset the integrator to its initial state. Delegates to
    /// [`astrodyn_dynamics::GaussJacksonState::reset`].
    pub fn reset(&mut self) {
        self.0.reset()
    }

    /// Reset the integrator and clear the topology-dirty flag. Delegates
    /// to [`astrodyn_dynamics::GaussJacksonState::reset_for_topology_change`].
    pub fn reset_for_topology_change(&mut self) {
        self.0.reset_for_topology_change()
    }

    /// Mark the integrator as carrying stale predictor / corrector
    /// history. Delegates to
    /// [`astrodyn_dynamics::GaussJacksonState::mark_topology_dirty`].
    pub fn mark_topology_dirty(&mut self) {
        self.0.mark_topology_dirty()
    }

    /// Returns true if the integrator is carrying stale history.
    /// Delegates to
    /// [`astrodyn_dynamics::GaussJacksonState::is_topology_dirty`].
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
    /// Delegates to [`astrodyn_dynamics::GaussJacksonState::is_priming`].
    pub fn is_priming(&self) -> bool {
        self.0.is_priming()
    }

    /// Cumulative count of unconverged bootstrap-edit iterations.
    /// Delegates to
    /// [`astrodyn_dynamics::GaussJacksonState::bootstrap_unconverged_iterations`].
    pub fn bootstrap_unconverged_iterations(&self) -> u32 {
        self.0.bootstrap_unconverged_iterations()
    }

    /// Mutable reference to the wrapped raw state. Used by the
    /// `astrodyn` integration kernel to pass through to
    /// `astrodyn_dynamics::abm4_translational_step` / GJ's `integrate`
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

/// Configuration for the LSODE integrator.
///
/// Opaque newtype over [`astrodyn_dynamics::LsodeConfig`]. Defaults to the
/// non-stiff implicit-Adams family with functional iteration (JEOD's
/// `RUN_lsode` configuration). The stiff BDF family is not yet selectable.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LsodeConfig(RawLsodeConfig);

impl LsodeConfig {
    /// Non-stiff implicit-Adams configuration (the default).
    pub fn non_stiff_adams() -> Self {
        Self(RawLsodeConfig::default())
    }

    /// Stiff backward-differentiation (BDF) configuration: the BDF family
    /// (orders 1–5) with a modified-Newton chord corrector driven by an
    /// internally-generated finite-difference Jacobian (ODEPACK MITER=2).
    /// Use for stiff systems where the non-stiff Adams family would be
    /// forced to take tiny steps.
    pub fn bdf_stiff() -> Self {
        Self(RawLsodeConfig {
            method: astrodyn_dynamics::IntegrationMethod::ImplicitBackDiffStiff,
            corrector: astrodyn_dynamics::CorrectorMethod::NewtonIterInternalJac,
            max_order: 5,
            ..RawLsodeConfig::default()
        })
    }

    /// Set the relative and absolute error tolerances (RTOL, ATOL).
    pub fn with_tolerances(mut self, rel_tolerance: f64, abs_tolerance: f64) -> Self {
        self.0.rel_tolerance = rel_tolerance;
        self.0.abs_tolerance = abs_tolerance;
        self
    }

    /// Set the maximum integration order (clamped to the family cap).
    pub fn with_max_order(mut self, max_order: usize) -> Self {
        self.0.max_order = max_order;
        self
    }

    /// Set the maximum number of internal steps per integrate-to-target
    /// call (MXSTEP).
    pub fn with_max_num_steps(mut self, max_num_steps: usize) -> Self {
        self.0.max_num_steps = max_num_steps;
        self
    }

    /// Validate the configuration, panicking on invalid values. See
    /// [`astrodyn_dynamics::LsodeConfig::check`].
    pub fn validate(&self) {
        self.0.check()
    }
}

impl From<LsodeConfig> for RawLsodeConfig {
    fn from(value: LsodeConfig) -> Self {
        value.0
    }
}

impl From<RawLsodeConfig> for LsodeConfig {
    fn from(value: RawLsodeConfig) -> Self {
        Self(value)
    }
}

/// Persistent LSODE integrator state.
///
/// Opaque newtype over [`astrodyn_dynamics::LsodeState`] (the Nordsieck
/// history + adaptive-control bookkeeping). Only the methods the kernel and
/// runner call across the boundary are exposed.
#[derive(Debug, Clone)]
pub struct LsodeState(RawLsodeState);

impl LsodeState {
    /// Create a new LSODE integrator with the given configuration.
    pub fn new(config: LsodeConfig) -> Self {
        Self(RawLsodeState::new(config.into()))
    }

    /// Returns the configuration this integrator was created with.
    pub fn config(&self) -> LsodeConfig {
        LsodeConfig(*self.0.config())
    }

    /// Mark the multistep history stale after a topology change.
    pub fn mark_topology_dirty(&mut self) {
        self.0.mark_topology_dirty()
    }

    /// Returns true if the history is awaiting a reset.
    pub fn is_topology_dirty(&self) -> bool {
        self.0.is_topology_dirty()
    }

    /// Reset to a cold start (history re-primed on the next step).
    pub fn reset_for_topology_change(&mut self) {
        self.0.reset_for_topology_change()
    }

    /// Mutable reference to the wrapped raw state, for the integration
    /// kernel to pass through to `lsode_translational_step`.
    #[inline]
    pub fn inner_mut(&mut self) -> &mut RawLsodeState {
        &mut self.0
    }

    /// Shared reference to the wrapped raw state.
    #[inline]
    pub fn inner(&self) -> &RawLsodeState {
        &self.0
    }
}

impl From<RawLsodeState> for LsodeState {
    #[inline]
    fn from(value: RawLsodeState) -> Self {
        Self(value)
    }
}

impl From<LsodeState> for RawLsodeState {
    #[inline]
    fn from(value: LsodeState) -> Self {
        value.0
    }
}

/// Persistent Adams-Bashforth-Moulton 4 integrator state.
///
/// Opaque newtype over [`astrodyn_dynamics::Abm4State`]. Only the methods
/// consumers actually call across the boundary are exposed; the
/// internal sliding-window history stays in `astrodyn_dynamics`.
#[derive(Debug, Clone, Default)]
pub struct Abm4State(RawAbm4State);

impl Abm4State {
    /// Create a fresh, unprimed integrator state.
    /// Delegates to [`astrodyn_dynamics::Abm4State::new`].
    pub fn new() -> Self {
        Self(RawAbm4State::new())
    }

    /// Reset the integrator back to its unprimed state.
    /// Delegates to [`astrodyn_dynamics::Abm4State::reset`].
    pub fn reset(&mut self) {
        self.0.reset()
    }

    /// Reset the integrator and clear the topology-dirty flag.
    /// Delegates to [`astrodyn_dynamics::Abm4State::reset_for_topology_change`].
    pub fn reset_for_topology_change(&mut self) {
        self.0.reset_for_topology_change()
    }

    /// Mark the integrator as carrying stale predictor history.
    /// Delegates to [`astrodyn_dynamics::Abm4State::mark_topology_dirty`].
    pub fn mark_topology_dirty(&mut self) {
        self.0.mark_topology_dirty()
    }

    /// Returns true if the integrator is carrying stale history.
    /// Delegates to [`astrodyn_dynamics::Abm4State::is_topology_dirty`].
    pub fn is_topology_dirty(&self) -> bool {
        self.0.is_topology_dirty()
    }

    /// Returns true while the integrator is still priming with RK4.
    /// Delegates to [`astrodyn_dynamics::Abm4State::is_priming`].
    pub fn is_priming(&self) -> bool {
        self.0.is_priming()
    }

    /// Mutable reference to the wrapped raw state. Used by the
    /// `astrodyn` integration kernel to pass through to
    /// `astrodyn_dynamics::abm4_translational_step` without copying.
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
