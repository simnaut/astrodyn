//! `jeod_math` — JEOD-faithful math kernels.
//!
//! Phase 2 of the type-system refactor (#104) unifies the quaternion type
//! with [`jeod_quantities`]. `JeodQuat` is now a re-export of
//! `jeod_quantities::JeodQuat` (the canonical `Quat<ScalarFirst,
//! LeftTransform>` type alias), and all algebraic/conversion methods live
//! on that unified type so there is only one quaternion in the workspace.

#![forbid(unsafe_code)]

pub use jeod_quantities::prelude::*;

pub mod error;
pub mod euler_angles;
pub mod geodetic;
pub mod lvlh;
pub mod orbital_elements;
pub mod quaternion;
pub mod solar_beta;
#[cfg(any(test, feature = "test-utils"))]
pub mod test_utils;
pub mod types;

pub use error::*;
pub use euler_angles::{
    compute_euler_angles_from_matrix_typed, compute_matrix_from_euler_angles_typed,
    compute_quaternion_from_euler_angles_typed, quaternion_to_matrix_normalized, EulerSequence,
    ALL_SEQUENCES,
};
pub use geodetic::{
    cartesian_to_geodetic_typed, geodetic_to_cartesian_typed, GeodeticState, GeodeticStateTyped,
};
pub use jeod_quantities::{
    JeodQuat, Layout, LeftTransform, NormalizedQuat, Quat, RightTransform, ScalarFirst, ScalarLast,
    Transform,
};
pub use lvlh::{compute_lvlh_frame_typed, LvlhFrame};
pub use orbital_elements::OrbitalElements;
pub use solar_beta::solar_beta_angle_typed;
pub use types::*;
