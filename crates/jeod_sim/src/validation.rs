//! Per-body and whole-scenario validation: cross-checks vehicle
//! configuration, gravity / atmosphere / SRP wiring, integrator
//! compatibility, and contact-pair consistency. Adapters call
//! [`validate_body`] before stepping; misconfigurations surface as
//! [`ValidationError`] variants instead of silently producing bad
//! physics.

use jeod_dynamics::{DynamicsConfig, MassProperties, TranslationalState};
use jeod_gravity::{GravityControls, GravitySource};

/// Validation error for a body's configuration.
///
/// Returned by [`validate_body`] instead of panicking, so callers can decide
/// how to handle errors. The Bevy adapter wraps these and panics with entity
/// context; standalone users can log or handle gracefully.
///
/// Each variant carries diagnostic context (offending indices, lengths,
/// detail strings) intended for the [`Display`](std::fmt::Display) impl. The
/// per-field `usize` payloads are self-explanatory in that context, so the
/// `#[allow(missing_docs)]` is applied at the enum level.
#[allow(missing_docs)]
#[derive(Debug, Clone)]
pub enum ValidationError {
    /// `GravityControls` present but no gravity acceleration storage.
    MissingGravityAcceleration,
    /// `three_dof=true` with `rotational_dynamics=true`.
    ThreeDofWithRotational,
    /// `rotational_dynamics=true` but no mass properties.
    RotationalWithoutMass,
    /// `rotational_dynamics=true` but no rotational state.
    RotationalWithoutRotState,
    /// Inertia tensor inconsistency (`I * I^-1 != identity`).
    InertiaInconsistent,
    /// Gravity control references a nonexistent source.
    GravitySourceMissing { source_id: String },
    /// Translational state appears uninitialized (all zeros).
    UninitializedState,
    /// `plate_temperatures` or `plate_t_pow4_cached` length doesn't match `flat_plates`.
    PlateTemperatureLengthMismatch {
        num_plates: usize,
        num_temperatures: usize,
        num_t_pow4: usize,
    },
    /// `sun_source` index is out of range for the sources table.
    SunSourceOutOfRange { index: usize, num_sources: usize },
    /// `shadow_body` index is out of range for the sources table.
    ShadowBodyOutOfRange { index: usize, num_sources: usize },
    /// `geodetic_planet` source index is out of range for the sources table.
    GeodeticPlanetOutOfRange { index: usize, num_sources: usize },
    /// `orbital_elements_source` index is out of range for the sources table.
    OrbitalElementsSourceOutOfRange { index: usize, num_sources: usize },
    /// `atmosphere_planet_source` index is out of range for the sources table.
    AtmospherePlanetOutOfRange { index: usize, num_sources: usize },
    /// `moon_source` index is out of range for the sources table.
    MoonSourceOutOfRange { index: usize, num_sources: usize },
    /// Drag or SRP configured but no mass properties (force → acceleration requires mass).
    ForceProducerWithoutMass { body_idx: usize },
    /// GaussJackson integrator with rotational_dynamics=true (6-DOF not supported).
    GaussJacksonWith6Dof { body_idx: usize },
    /// GaussJackson config invalid.
    GaussJacksonConfigInvalid { body_idx: usize, detail: String },
    /// ABM4 integrator with rotational_dynamics=true (6-DOF not supported).
    Abm4With6Dof { body_idx: usize },
    /// Body has `atmospheric_state` but simulation has no atmosphere config.
    AtmosphericStateWithoutAtmosphere { body_idx: usize },
    /// Body has `compute_solar_beta=true` but simulation has no `sun_source`.
    SolarBetaWithoutSunSource { body_idx: usize },
    /// Body has `compute_gravity_torque=true` but lacks mass or rotational state.
    GravityTorqueWithoutMassOrRot { body_idx: usize },
    /// Body has `earth_lighting_config` but simulation has no `sun_source`.
    EarthLightingWithoutSunSource { body_idx: usize },
    /// Body has `earth_lighting_config` but simulation has no `moon_source`.
    EarthLightingWithoutMoonSource { body_idx: usize },
    /// Frame switch `target_source` index exceeds number of gravity sources.
    FrameSwitchTargetSourceOutOfRange {
        body_idx: usize,
        target_source: usize,
        num_sources: usize,
    },
    /// Frame switch `target_source` is not in the body's gravity controls.
    /// The post-switch gravity reclassification requires a control entry for
    /// the target source (it becomes the non-differential central body).
    FrameSwitchTargetSourceNotInControls {
        body_idx: usize,
        target_source: usize,
    },
    /// Body uses a non-root integration frame (or has an active frame switch to
    /// a non-root frame) but has features that assume root-inertial coordinates.
    NonRootFrameWithRootDependentFeatures { body_idx: usize },
    /// Source has an ephemeris mapping but its inertial frame is the root frame.
    /// The root frame must remain at the origin; ephemeris position updates
    /// would be silently ignored.
    EphemerisOnRootSource { source_idx: usize },
    /// Contact pairs are registered but a body uses a non-RK4 integrator.
    /// The contact-coupled integration path only supports RK4 (all bodies
    /// share one multi-body RK4 kernel when contact pairs are active).
    ContactPairsRequireRk4 { body_idx: usize },
    /// Contact pairs are registered but a body lacks rotational state or mass
    /// properties. The contact-coupled path requires full 6-DOF on every body.
    ContactPairsRequire6Dof { body_idx: usize },
    /// A contact pair references two bodies integrated in different frames.
    /// The coupled RK4 contact evaluator consumes stage states directly,
    /// so both bodies' states must be expressed in a common frame.
    ContactPairFrameMismatch {
        pair_idx: usize,
        body_a: usize,
        body_b: usize,
        frame_a: usize,
        frame_b: usize,
    },
    /// A contact pair's bodies are integrated in a non-root frame. The
    /// coupled contact evaluator assumes coordinates in the root inertial
    /// frame (contact forces and torques are computed without any per-step
    /// frame transform).
    ContactPairNonRootFrame {
        pair_idx: usize,
        body_idx: usize,
        frame: usize,
        root: usize,
    },
    /// A ground-contact pair's body is integrated in a non-root frame.
    /// The coupled-RK4 closure feeds the body's stage state directly to
    /// the ground-contact evaluator without any frame transform, so the
    /// body must integrate in the root inertial frame.
    GroundContactPairNonRootFrame {
        pair_idx: usize,
        body_idx: usize,
        frame: usize,
        root: usize,
    },
    /// A ground-contact pair references a non-central planet source.
    /// `compute_ground_contact_geometry` projects the vehicle's
    /// inertial position into pfix coords assuming the planet center
    /// is at the inertial origin (the central source convention).
    GroundContactNonCentralPlanet {
        pair_idx: usize,
        planet_source: usize,
    },
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingGravityAcceleration => {
                write!(
                    f,
                    "GravityControls present but no GravityAcceleration storage. \
                     In JEOD, grav_interaction is a value member of DynBody."
                )
            }
            Self::ThreeDofWithRotational => {
                write!(
                    f,
                    "three_dof=true AND rotational_dynamics=true is invalid. \
                     In JEOD, three_dof=true prevents creation of the rotational integrator."
                )
            }
            Self::RotationalWithoutMass => {
                write!(
                    f,
                    "rotational_dynamics=true but no MassProperties. \
                     In JEOD, DynBody always has MassBody (inertia tensor required for \
                     Euler's equation)."
                )
            }
            Self::RotationalWithoutRotState => {
                write!(
                    f,
                    "rotational_dynamics=true but no RotationalState. \
                     Provide initial quaternion and angular velocity."
                )
            }
            Self::InertiaInconsistent => {
                write!(f, "Inertia tensor inconsistency (I * I^-1 != identity)")
            }
            Self::GravitySourceMissing { source_id } => {
                write!(
                    f,
                    "Gravity source {source_id} does not exist. \
                     JEOD logs a non-fatal error and skips; we report to prevent \
                     silently wrong physics."
                )
            }
            Self::UninitializedState => {
                write!(
                    f,
                    "Translational state appears uninitialized (position and velocity \
                     both zero). In JEOD, uninitialized state is a fatal error."
                )
            }
            Self::PlateTemperatureLengthMismatch {
                num_plates,
                num_temperatures,
                num_t_pow4,
            } => {
                write!(
                    f,
                    "plate_temperatures (len={num_temperatures}) or plate_t_pow4_cached \
                     (len={num_t_pow4}) does not match flat_plates (len={num_plates}). \
                     All three must have the same length."
                )
            }
            Self::SunSourceOutOfRange { index, num_sources } => {
                write!(
                    f,
                    "sun_source index {index} is out of range (only {num_sources} sources). \
                     Ensure sun_source refers to a valid source index."
                )
            }
            Self::MoonSourceOutOfRange { index, num_sources } => {
                write!(
                    f,
                    "moon_source index {index} is out of range (only {num_sources} sources). \
                     Ensure moon_source refers to a valid source index."
                )
            }
            Self::ShadowBodyOutOfRange { index, num_sources } => {
                write!(
                    f,
                    "shadow_body index {index} is out of range (only {num_sources} sources). \
                     Ensure shadow_body refers to a valid source index."
                )
            }
            Self::GeodeticPlanetOutOfRange { index, num_sources } => {
                write!(
                    f,
                    "geodetic_planet index {index} is out of range (only {num_sources} sources). \
                     Ensure geodetic_planet refers to a valid source index."
                )
            }
            Self::OrbitalElementsSourceOutOfRange { index, num_sources } => {
                write!(
                    f,
                    "orbital_elements_source index {index} is out of range (only {num_sources} sources). \
                     Ensure orbital_elements_source refers to a valid source index."
                )
            }
            Self::AtmospherePlanetOutOfRange { index, num_sources } => {
                write!(
                    f,
                    "atmosphere_planet_source index {index} is out of range (only {num_sources} sources). \
                     Ensure atmosphere_planet_source refers to a valid source index."
                )
            }
            Self::ForceProducerWithoutMass { body_idx } => {
                write!(
                    f,
                    "Body {body_idx}: drag or SRP configured but no MassProperties. In JEOD, \
                     DynBody always has mass. Provide MassProperties for any body with \
                     interaction forces."
                )
            }
            Self::GaussJacksonWith6Dof { body_idx } => {
                write!(
                    f,
                    "Body {body_idx}: GaussJackson integrator with rotational_dynamics=true. \
                     GJ is currently translational-only. Set rotational_dynamics=false \
                     for GJ bodies."
                )
            }
            Self::GaussJacksonConfigInvalid { body_idx, detail } => {
                write!(f, "Body {body_idx}: GaussJackson config invalid: {detail}")
            }
            Self::Abm4With6Dof { body_idx } => {
                write!(
                    f,
                    "Body {body_idx}: ABM4 integrator with rotational_dynamics=true. \
                     ABM4 is currently translational-only. Set rotational_dynamics=false \
                     for ABM4 bodies."
                )
            }
            Self::AtmosphericStateWithoutAtmosphere { body_idx } => {
                write!(
                    f,
                    "Body {body_idx}: atmospheric_state is set but simulation has no \
                     atmosphere config. Set Simulation::atmosphere to enable atmosphere \
                     evaluation."
                )
            }
            Self::SolarBetaWithoutSunSource { body_idx } => {
                write!(
                    f,
                    "Body {body_idx}: compute_solar_beta=true but simulation has no sun_source. \
                     Set Simulation::sun_source to enable solar beta computation."
                )
            }
            Self::GravityTorqueWithoutMassOrRot { body_idx } => {
                write!(
                    f,
                    "Body {body_idx}: compute_gravity_torque=true but body lacks mass \
                     properties or rotational state. Gravity gradient torque requires both."
                )
            }
            Self::EarthLightingWithoutSunSource { body_idx } => {
                write!(
                    f,
                    "Body {body_idx}: earth_lighting_config is set but simulation has no \
                     sun_source. Set Simulation::sun_source for earth lighting computation."
                )
            }
            Self::EarthLightingWithoutMoonSource { body_idx } => {
                write!(
                    f,
                    "Body {body_idx}: earth_lighting_config is set but simulation has no \
                     moon_source. Set Simulation::moon_source for earth lighting computation."
                )
            }
            Self::FrameSwitchTargetSourceOutOfRange {
                body_idx,
                target_source,
                num_sources,
            } => {
                write!(
                    f,
                    "Body {body_idx}: frame switch target_source index {target_source} \
                     is out of range (simulation has {num_sources} gravity sources)."
                )
            }
            Self::FrameSwitchTargetSourceNotInControls {
                body_idx,
                target_source,
            } => {
                write!(
                    f,
                    "Body {body_idx}: frame switch target_source index {target_source} \
                     is not in the body's gravity_controls. The post-switch gravity \
                     reclassification requires the target source to have a \
                     GravityControl entry (it becomes the non-differential central body)."
                )
            }
            Self::NonRootFrameWithRootDependentFeatures { body_idx } => {
                write!(
                    f,
                    "Body {body_idx}: non-root integration frame with features that \
                     assume root-inertial coordinates (drag, SRP, orbital elements, \
                     euler angles, LVLH, geodetic, solar beta, or earth lighting). \
                     These derived states assume the simulation's central-body \
                     inertial frame and will produce incorrect results in other frames."
                )
            }
            Self::EphemerisOnRootSource { source_idx } => {
                write!(
                    f,
                    "Source {source_idx}: has an ephemeris mapping but its inertial frame \
                     is the root frame. The root frame must remain at the origin; \
                     ephemeris position updates would be silently ignored. Remove the \
                     ephemeris mapping for the central body source."
                )
            }
            Self::ContactPairsRequireRk4 { body_idx } => {
                write!(
                    f,
                    "Body {body_idx}: contact pairs are registered but this body uses a \
                     non-RK4 integrator. The contact-coupled path drives all bodies through \
                     a shared multi-body RK4 kernel; switch the integrator to RK4, avoid \
                     registering contact pairs for this simulation, or recreate the \
                     simulation without contact pairs."
                )
            }
            Self::ContactPairsRequire6Dof { body_idx } => {
                write!(
                    f,
                    "Body {body_idx}: contact pairs are registered but this body lacks \
                     rotational state or mass properties. The contact-coupled path \
                     requires full 6-DOF (rot + mass) on every body."
                )
            }
            Self::ContactPairFrameMismatch {
                pair_idx,
                body_a,
                body_b,
                frame_a,
                frame_b,
            } => {
                write!(
                    f,
                    "Contact pair {pair_idx}: bodies {body_a} and {body_b} are \
                     integrated in different frames ({frame_a} != {frame_b}). \
                     The coupled RK4 contact evaluator reads stage states \
                     directly from each body, so both bodies must share the \
                     same integration frame."
                )
            }
            Self::ContactPairNonRootFrame {
                pair_idx,
                body_idx,
                frame,
                root,
            } => {
                write!(
                    f,
                    "Contact pair {pair_idx}: body {body_idx} is integrated in \
                     frame {frame} but the coupled contact evaluator assumes \
                     coordinates in the root inertial frame ({root}). Integrate \
                     contact-participating bodies in the root inertial frame, \
                     or transform stage states before contact evaluation."
                )
            }
            Self::GroundContactPairNonRootFrame {
                pair_idx,
                body_idx,
                frame,
                root,
            } => {
                write!(
                    f,
                    "Ground-contact pair {pair_idx}: body {body_idx} is \
                     integrated in frame {frame} but the ground-contact \
                     evaluator consumes the body's stage state without any \
                     frame transform; must be the root inertial frame ({root})."
                )
            }
            Self::GroundContactNonCentralPlanet {
                pair_idx,
                planet_source,
            } => {
                write!(
                    f,
                    "Ground-contact pair {pair_idx}: planet_source {planet_source} \
                     is not the central source. \
                     `compute_ground_contact_geometry` projects vehicle inertial \
                     position into pfix assuming the planet center is at the \
                     inertial origin; that holds only for the central source. \
                     Use a central planet for ground contact, or extend the \
                     algorithm to accept a non-central planet position."
                )
            }
        }
    }
}

impl std::error::Error for ValidationError {}

impl ValidationError {
    /// Whether this is a warning rather than a fatal error.
    ///
    /// Warnings indicate suspicious-but-valid state (e.g., a body at the origin
    /// might be intentional). Both the Bevy adapter and `Simulation::validate()`
    /// should use this to decide severity.
    pub fn is_warning(&self) -> bool {
        matches!(
            self,
            Self::UninitializedState | Self::NonRootFrameWithRootDependentFeatures { .. }
        )
    }
}

/// Validate a body's configuration against JEOD invariants.
///
/// Returns a list of errors (empty = valid). The caller decides how to handle
/// errors: `Simulation::validate()` returns them; the Bevy adapter panics with
/// entity context.
///
/// # Arguments
/// - `config`: dynamics configuration flags
/// - `gravity_controls`: the body's gravity controls
/// - `has_gravity_accel`: whether gravity acceleration storage exists
/// - `mass`: optional mass properties (for inertia consistency check)
/// - `has_rot_state`: whether rotational state exists
/// - `trans_state`: optional translational state (for uninitialized check)
/// - `source_lookup`: resolves source IDs to `GravitySource` (returns `None` if missing)
/// - `plate_counts`: if flat plates are present, `Some((num_plates, num_temperatures, num_t_pow4))`
// JEOD_INV: DM.03 — validation runs once before first integration step
#[allow(clippy::too_many_arguments)]
pub fn validate_body<'a, S: Copy + std::fmt::Debug>(
    config: &DynamicsConfig,
    gravity_controls: &GravityControls<S>,
    has_gravity_accel: bool,
    mass: Option<&MassProperties>,
    has_rot_state: bool,
    trans_state: Option<&TranslationalState>,
    source_lookup: impl Fn(S) -> Option<&'a GravitySource>,
    plate_counts: Option<(usize, usize, usize)>,
) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    // GravityAcceleration required for integration
    if !has_gravity_accel {
        errors.push(ValidationError::MissingGravityAcceleration);
    }

    // three_dof consistency (JEOD_INV: DB.05, DB.06)
    if config.three_dof && config.rotational_dynamics {
        errors.push(ValidationError::ThreeDofWithRotational);
    }

    // Rotational dynamics requires mass and rotational state (JEOD_INV: MA.01)
    if config.rotational_dynamics {
        if mass.is_none() {
            errors.push(ValidationError::RotationalWithoutMass);
        }
        if !has_rot_state {
            errors.push(ValidationError::RotationalWithoutRotState);
        }
    }

    // Inertia consistency (JEOD_INV: DB.19, MA.04)
    if let Some(m) = mass {
        let product = m.inertia * m.inverse_inertia;
        let identity = glam::DMat3::IDENTITY;
        let diff = (product.x_axis - identity.x_axis).length()
            + (product.y_axis - identity.y_axis).length()
            + (product.z_axis - identity.z_axis).length();
        if diff > jeod_dynamics::INERTIA_CONSISTENCY_TOL {
            errors.push(ValidationError::InertiaInconsistent);
        }
    }

    // JEOD_INV: GV.12 — gravity source must exist for control
    // JEOD_INV: DM.08 — gravitation requires gravity source (init-time check)
    for ctrl in &gravity_controls.controls {
        if source_lookup(ctrl.source_name).is_none() {
            errors.push(ValidationError::GravitySourceMissing {
                source_id: format!("{:?}", ctrl.source_name),
            });
        }
    }

    // Uninitialized state detection (JEOD_INV: DM.05, DB.11)
    if config.translational_dynamics {
        if let Some(trans) = trans_state {
            if trans.is_likely_uninitialized() {
                errors.push(ValidationError::UninitializedState);
            }
        }
    }

    // Plate temperature / t_pow4_cached length must match flat_plates
    if let Some((num_plates, num_temperatures, num_t_pow4)) = plate_counts {
        if num_temperatures != num_plates || num_t_pow4 != num_plates {
            errors.push(ValidationError::PlateTemperatureLengthMismatch {
                num_plates,
                num_temperatures,
                num_t_pow4,
            });
        }
    }

    errors
}
