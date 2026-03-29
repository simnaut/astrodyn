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
        }
    }
}

impl std::error::Error for ValidationError {}

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
// JEOD_INV: DM.03 — validation runs once before first integration step
pub fn validate_body<'a, S: Copy + std::fmt::Debug>(
    config: &DynamicsConfig,
    gravity_controls: &GravityControls<S>,
    has_gravity_accel: bool,
    mass: Option<&MassProperties>,
    has_rot_state: bool,
    trans_state: Option<&TranslationalState>,
    source_lookup: impl Fn(S) -> Option<&'a GravitySource>,
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

    errors
}
