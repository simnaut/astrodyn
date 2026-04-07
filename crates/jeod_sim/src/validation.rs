use jeod_dynamics::{DynamicsConfig, MassProperties, TranslationalState};
use jeod_gravity::{GravityControls, GravitySource};

/// Validation error for a body's configuration.
///
/// Returned by [`validate_body`] instead of panicking, so callers can decide
/// how to handle errors. The Bevy adapter wraps these and panics with entity
/// context; standalone users can log or handle gracefully.
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
    /// Drag or SRP configured but no mass properties (force → acceleration requires mass).
    ForceProducerWithoutMass { body_idx: usize },
    /// GaussJackson integrator with rotational_dynamics=true (6-DOF not supported).
    GaussJacksonWith6Dof { body_idx: usize },
    /// GaussJackson order out of supported range (1..=8).
    GaussJacksonOrderOutOfRange { body_idx: usize, order: usize },
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
            Self::GaussJacksonOrderOutOfRange { body_idx, order } => {
                write!(
                    f,
                    "Body {body_idx}: GaussJackson order {order} is out of supported range \
                     (1..=8). AB/AM coefficient tables only exist up to order 8."
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
        matches!(self, Self::UninitializedState)
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
