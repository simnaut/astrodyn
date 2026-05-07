//! Convenience aliases for common physical 3-vectors.
//!
//! All aliases default to the inertial frame; override with an explicit
//! frame tag, e.g. `Velocity<Ecef>`.

use uom::si::{
    acceleration, angular_acceleration, angular_momentum, angular_velocity, force, jerk, length,
    torque, velocity,
};

use crate::frame::RootInertial;
use crate::qty3::Qty3;

/// Position in frame `F` (default: `RootInertial`). Base SI unit: meters.
pub type Position<F = RootInertial> = Qty3<length::Dimension, F>;

/// Velocity in frame `F`. Base SI unit: m/s.
pub type Velocity<F = RootInertial> = Qty3<velocity::Dimension, F>;

/// Acceleration in frame `F`. Base SI unit: m/s².
pub type Acceleration<F = RootInertial> = Qty3<acceleration::Dimension, F>;

/// Jerk (time derivative of acceleration) in frame `F`. Base SI unit: m/s³.
pub type Jerk<F = RootInertial> = Qty3<jerk::Dimension, F>;

/// Force in frame `F`. Base SI unit: newtons.
pub type Force<F = RootInertial> = Qty3<force::Dimension, F>;

/// Torque in frame `F`. Base SI unit: N·m.
pub type Torque<F = RootInertial> = Qty3<torque::Dimension, F>;

/// Angular velocity in frame `F`. Base SI unit: rad/s.
pub type AngularVelocity<F = RootInertial> = Qty3<angular_velocity::Dimension, F>;

/// Angular acceleration in frame `F`. Base SI unit: rad/s².
pub type AngularAcceleration<F = RootInertial> = Qty3<angular_acceleration::Dimension, F>;

/// Angular momentum in frame `F`. Base SI unit: kg·m²/s.
pub type AngularMomentum<F = RootInertial> = Qty3<angular_momentum::Dimension, F>;

// `InertiaTensor<F>` is a `DMat3` newtype rather than a `Qty3`, but it
// belongs in the same conceptual namespace as the typed quantities — re-
// export it here for `use astrodyn_quantities::aliases::*;` callers.
pub use crate::inertia::InertiaTensor;

// Spherical-harmonic ordinal index (a `usize` newtype that the type
// checker keeps distinct from angular `Degree` and dimensionless
// `Ratio`).
pub use crate::harmonic::HarmonicDegree;
