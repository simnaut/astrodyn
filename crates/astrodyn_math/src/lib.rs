//! `astrodyn_math` — JEOD-faithful math kernels.
//!
//! Pure-Rust port of the standalone math utilities under JEOD
//! `models/utils/`: quaternions, Euler angles, orbital elements, geodetic
//! coordinates, the LVLH frame, the solar beta angle, and the small but
//! load-bearing collection of vector/matrix helpers that mirror JEOD's
//! `Vector3`/`Matrix3x3` inline functions.
//!
//! Phase 2 of the type-system refactor (#104) unifies the quaternion type
//! with `astrodyn_quantities`. [`JeodQuat`] is now a re-export of
//! `astrodyn_quantities::JeodQuat` (the canonical `Quat<ScalarFirst,
//! LeftTransform>` type alias), and all algebraic/conversion methods live
//! on that unified type so there is only one quaternion in the workspace.
//! [`Quat`], [`NormalizedQuat`], the [`Layout`] tags ([`ScalarFirst`],
//! [`ScalarLast`]) and the [`Transform`] tags ([`LeftTransform`],
//! [`RightTransform`]) are re-exported so kernel code can name its inputs
//! at the JEOD convention without reaching into upstream crates directly.
//!
//! ## Public surface
//!
//! - [`orbital_elements::OrbitalElements`] — Cartesian↔Keplerian conversion
//!   ported from `models/utils/orbital_elements/src/orbital_elements.cc`.
//!   Holds semi-major axis, eccentricity, inclination, RAAN, argument of
//!   periapsis, and the three anomalies (true, mean, orbital), plus
//!   energy/angular-momentum diagnostics.
//! - [`euler_angles`] — port of JEOD `models/utils/orientation/`:
//!   [`EulerSequence`], [`ALL_SEQUENCES`], and the
//!   `compute_*_typed`/`quaternion_to_matrix_normalized` helpers.
//! - [`geodetic`] — port of `models/utils/planet_fixed/`:
//!   [`cartesian_to_geodetic_typed`], [`geodetic_to_cartesian_typed`],
//!   [`GeodeticState`], [`GeodeticStateTyped`].
//! - [`lvlh`] — port of `models/utils/lvlh_frame/`:
//!   [`compute_lvlh_frame_typed`], [`LvlhFrame`].
//! - [`solar_beta::solar_beta_angle_typed`] — solar beta angle for an
//!   orbit, used in eclipse-fraction and thermal calculations.
//! - [`quaternion`] — JEOD-convention quaternion algebra on the unified
//!   [`JeodQuat`] type.
//!
//! ## Vector / matrix helpers
//!
//! [`types`] re-exports `glam`'s [`DMat3`] / [`DQuat`] / [`DVec3`] and adds
//! the [`mat3_from_rows`] constructor (JEOD source is row-major while
//! `glam` is column-major, so this lets ports read top-to-bottom against
//! the C++ source) plus the inline `Vector3::*` / `Matrix3x3::*` operators
//! ported from `models/utils/math/include/vector3_inline.hh` and
//! `matrix3x3_inline.hh` (e.g., `vector3_transform`,
//! `vector3_transform_transpose`, `matrix3x3_transform_matrix`,
//! `matrix3x3_transpose_transform_matrix`,
//! `matrix3x3_product_transpose_transpose`).
//!
//! Pure Rust, zero Bevy dependency.
//!
//! ## Example
//!
//! Build a JEOD-convention left-transformation quaternion from an
//! eigen-rotation (angle + axis) and verify the round-trip
//! `q.conjugate(q.transform(v)) == v`:
//!
//! ```
//! use astrodyn_math::JeodQuat;
//! use glam::DVec3;
//!
//! // Arbitrary rotation: 0.7 rad about the (1, 1, 1)/√3 axis.
//! let axis = DVec3::new(1.0, 1.0, 1.0).normalize();
//! let q = JeodQuat::left_quat_from_eigen_rotation(0.7, axis);
//!
//! // Transform-then-inverse-transform recovers the original vector
//! // (i.e. q.conjugate() reverses q).
//! let v = DVec3::new(3.0, -2.0, 5.0);
//! let v_rotated = q.left_quat_transform(v);
//! let v_back = q.conjugate().left_quat_transform(v_rotated);
//! assert!((v_back - v).length() < 1e-12);
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub use astrodyn_quantities::prelude::*;

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

pub use astrodyn_quantities::{
    JeodQuat, Layout, LeftTransform, NormalizedQuat, Quat, RightTransform, ScalarFirst, ScalarLast,
    Transform,
};
pub use error::*;
pub use euler_angles::{
    compute_euler_angles_from_matrix_typed, compute_matrix_from_euler_angles_typed,
    compute_quaternion_from_euler_angles_typed, quaternion_to_matrix_normalized, EulerSequence,
    ALL_SEQUENCES,
};
pub use geodetic::{
    cartesian_to_geodetic_typed, compute_body_geodetic, compute_body_geodetic_typed,
    geodetic_to_cartesian_typed, GeodeticState, GeodeticStateTyped,
};
pub use lvlh::{compute_lvlh_frame_typed, LvlhFrame};
pub use orbital_elements::OrbitalElements;
pub use solar_beta::{
    compute_body_solar_beta, compute_body_solar_beta_typed, solar_beta_angle_typed,
};
pub use types::*;
