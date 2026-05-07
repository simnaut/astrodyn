//! Per-body force / torque accumulators consumed by the integrator.
//!
//! Mirrors the force-collection step of JEOD's
//! [`models/dynamics/dyn_body/`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/dynamics/dyn_body/)
//! pipeline (v5.4.0). Interactions push contributions into these
//! accumulators; the integrator reads them in the `ForceCollectionSet`
//! schedule slot. Holds [`GravityAcceleration`], [`TotalForce`], the
//! per-frame typed siblings, and helpers like
//! `accumulate_gravity` and [`compute_translational_acceleration`].

use core::marker::PhantomData;

use astrodyn_quantities::aliases::{Acceleration, Force, Torque};
use astrodyn_quantities::frame::{BodyFrame, Frame, RootInertial, Vehicle};
use glam::{DMat3, DVec3};

/// Gravitational acceleration, gradient tensor, and potential for a body.
///
/// Computed by the gravity subsystem and consumed by the dynamics integrator.
/// All quantities are expressed in the integration frame (typically J2000 ECI).
///
/// # Sign conventions
/// - `grav_accel`: gravitational acceleration in m/s^2. Points toward the
///   attracting body (negative radial direction for a single point mass).
/// - `grav_grad`: gravity gradient tensor in 1/s^2. Symmetric 3x3 matrix;
///   trace is zero outside the attracting body (Laplace's equation).
/// - `grav_pot`: gravitational potential in m^2/s^2. Convention: **+mu/r**
///   for point mass (positive, matching JEOD `gravity_controls.cc`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GravityAcceleration {
    /// Gravitational acceleration in m/s^2, in integration frame.
    pub grav_accel: DVec3,
    /// Gravity gradient tensor in 1/s^2. Symmetric; trace = 0 outside body.
    pub grav_grad: DMat3,
    /// Gravitational potential in m^2/s^2. Convention: +mu/r for point mass (JEOD).
    pub grav_pot: f64,
}

impl Default for GravityAcceleration {
    fn default() -> Self {
        Self {
            grav_accel: DVec3::ZERO,
            grav_grad: DMat3::ZERO,
            grav_pot: 0.0,
        }
    }
}

/// Typed sibling of [`GravityAcceleration`].
///
/// Only `grav_accel` carries a frame phantom; `grav_grad` (1/s², a
/// gradient tensor) and `grav_pot` (m²/s², a scalar) remain DMat3 / f64.
/// Their dimensions are not exposed as typed aliases in
/// `astrodyn_quantities` because they are evaluated within a single frame
/// and never composed across frames in the dynamics layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GravityAccelerationTyped<F: Frame = RootInertial> {
    /// Gravitational acceleration in frame `F`.
    pub grav_accel: Acceleration<F>,
    /// Gravity-gradient tensor (`1/s²`).
    pub grav_grad: DMat3,
    /// Gravitational potential (`m²/s²`, JEOD sign convention is
    /// positive for bound orbits).
    pub grav_pot: f64,
}

impl<F: Frame> Default for GravityAccelerationTyped<F> {
    #[inline]
    fn default() -> Self {
        Self {
            grav_accel: Acceleration::<F>::zero(),
            grav_grad: DMat3::ZERO,
            grav_pot: 0.0,
        }
    }
}

impl<F: Frame> GravityAccelerationTyped<F> {
    /// Drop the phantom and emit the untyped storage form.
    #[inline]
    pub fn to_untyped(&self) -> GravityAcceleration {
        GravityAcceleration {
            grav_accel: self.grav_accel.raw_si(),
            grav_grad: self.grav_grad,
            grav_pot: self.grav_pot,
        }
    }

    /// Wrap an untyped [`GravityAcceleration`] as typed. **The caller
    /// asserts** the gravity acceleration is expressed in frame `F`.
    #[inline]
    pub fn from_untyped_unchecked(g: &GravityAcceleration) -> Self {
        Self {
            grav_accel: Acceleration::<F>::from_raw_si(g.grav_accel),
            grav_grad: g.grav_grad,
            grav_pot: g.grav_pot,
        }
    }
}

/// Net body force / torque, the integrator's input each step.
///
/// Force is expressed in the integration frame (typically J2000 ECI);
/// torque is in the body frame.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TotalForce {
    /// Net force in newtons, in the integration frame.
    pub force: DVec3,
    /// Net torque in N·m, in the body frame.
    pub torque: DVec3,
}

/// Typed sibling of [`TotalForce`].
///
/// `force` carries the integration frame `F` (defaults to `RootInertial`
/// to match the existing untyped struct's "force in integration
/// frame" convention); `torque` is in `BodyFrame<V>` (JEOD
/// convention). The two type parameters are independent so a body
/// integrating translation in one frame may carry torque about a
/// separate body axis. `V` has no default because no production
/// vehicle marker exists yet (Phase 5 territory).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TotalForceTyped<V: Vehicle, F: Frame = RootInertial> {
    /// Net force in the integration frame `F`.
    pub force: Force<F>,
    /// Net torque about the body-frame origin of vehicle `V`.
    pub torque: Torque<BodyFrame<V>>,
    _v: PhantomData<V>,
}

impl<V: Vehicle, F: Frame> Default for TotalForceTyped<V, F> {
    #[inline]
    fn default() -> Self {
        Self {
            force: Force::<F>::zero(),
            torque: Torque::<BodyFrame<V>>::zero(),
            _v: PhantomData,
        }
    }
}

impl<V: Vehicle, F: Frame> TotalForceTyped<V, F> {
    /// Drop the phantoms and emit the untyped storage form.
    #[inline]
    pub fn to_untyped(&self) -> TotalForce {
        TotalForce {
            force: self.force.raw_si(),
            torque: self.torque.raw_si(),
        }
    }

    /// Wrap an untyped [`TotalForce`] as typed. **The caller asserts**
    /// the force is in `F` and the torque in `BodyFrame<V>`.
    #[inline]
    pub fn from_untyped_unchecked(t: &TotalForce) -> Self {
        Self {
            force: Force::<F>::from_raw_si(t.force),
            torque: Torque::<BodyFrame<V>>::from_raw_si(t.torque),
            _v: PhantomData,
        }
    }
}

/// Frame-state derivatives: translational and rotational
/// acceleration. The integrator's per-step output.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FrameDerivatives {
    /// Translational acceleration in m/s², in the integration frame.
    pub trans_accel: DVec3,
    /// Rotational acceleration (angular acceleration) in rad/s²,
    /// expressed in the body frame.
    pub rot_accel: DVec3,
}

/// Typed sibling of [`FrameDerivatives`].
///
/// `trans_accel` carries the integration frame `F`; `rot_accel`
/// carries the body frame for vehicle `V`. The two type parameters
/// are independent — a body integrating translational state in one
/// frame may carry rotational state about a body axis whose
/// orientation is completely unrelated.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameDerivativesTyped<F: Frame, V: Vehicle> {
    /// Translational acceleration in the integration frame `F`.
    pub trans_accel: Acceleration<F>,
    /// Angular acceleration about the body-frame origin of `V`.
    pub rot_accel: astrodyn_quantities::aliases::AngularAcceleration<BodyFrame<V>>,
    _v: PhantomData<V>,
}

impl<F: Frame, V: Vehicle> Default for FrameDerivativesTyped<F, V> {
    #[inline]
    fn default() -> Self {
        Self {
            trans_accel: Acceleration::<F>::zero(),
            rot_accel: astrodyn_quantities::aliases::AngularAcceleration::<BodyFrame<V>>::zero(),
            _v: PhantomData,
        }
    }
}

impl<F: Frame, V: Vehicle> FrameDerivativesTyped<F, V> {
    /// Drop the frame phantoms and emit the untyped storage form.
    #[inline]
    pub fn to_untyped(&self) -> FrameDerivatives {
        FrameDerivatives {
            trans_accel: self.trans_accel.raw_si(),
            rot_accel: self.rot_accel.raw_si(),
        }
    }

    /// Wrap an untyped [`FrameDerivatives`] as typed. **The caller
    /// asserts** the translational accel is in `F` and the rotational
    /// accel is in `BodyFrame<V>`.
    #[inline]
    pub fn from_untyped_unchecked(d: &FrameDerivatives) -> Self {
        Self {
            trans_accel: Acceleration::<F>::from_raw_si(d.trans_accel),
            rot_accel:
                astrodyn_quantities::aliases::AngularAcceleration::<BodyFrame<V>>::from_raw_si(
                    d.rot_accel,
                ),
            _v: PhantomData,
        }
    }
}

/// Per-body dynamics-mode toggles consumed by the integrator.
///
/// Mirrors JEOD's `DynBody.translational_dynamics` /
/// `rotational_dynamics` / `three_dof` flags. `three_dof = true`
/// suppresses creation of the rotational integrator; the validator
/// panics on the inconsistent `three_dof = true && rotational = true`
/// combination.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DynamicsConfig {
    /// Integrate translational state.
    pub translational_dynamics: bool,
    /// Integrate rotational state.
    pub rotational_dynamics: bool,
    /// Run as a 3-DoF point mass (no rotational integrator created).
    pub three_dof: bool,
}

impl Default for DynamicsConfig {
    fn default() -> Self {
        Self {
            translational_dynamics: true,
            rotational_dynamics: false,
            three_dof: true,
        }
    }
}

impl DynamicsConfig {
    /// Validate consistency of dynamics configuration flags.
    ///
    /// In JEOD, `three_dof=true` prevents creation of the rotational integrator.
    /// Having `rotational_dynamics=true` with `three_dof=true` would attempt to
    /// integrate rotation without an integrator (undefined behavior in JEOD).
    ///
    /// # Panics
    /// Panics if `three_dof` and `rotational_dynamics` are both true.
    // JEOD_INV: DB.05 — three_dof=true prevents rotational integrator creation
    // JEOD_INV: DB.06 — three_dof=true AND rotational_dynamics=true is invalid
    pub fn validate(&self) {
        assert!(
            !(self.three_dof && self.rotational_dynamics),
            "DynamicsConfig has three_dof=true AND rotational_dynamics=true. \
             In JEOD, three_dof=true prevents creation of the rotational integrator. \
             Set rotational_dynamics=false when three_dof=true."
        );
    }
}

/// Compute translational acceleration from force and pre-computed
/// inverse mass: `a = F · (1/m)`.
///
/// JEOD pre-computes `1/m` once per step on `MassPointState` so the
/// inner loop is a multiply rather than a divide.
// JEOD_INV: DB.18 — F=ma via precomputed inverse_mass (matches JEOD MassPointState.inverse_mass)
pub fn compute_translational_acceleration(force: DVec3, inverse_mass: f64) -> DVec3 {
    force * inverse_mass
}

/// Compute the inertial-to-structural rotation matrix from component transforms.
///
/// `T_inertial_struct = T_struct_body^T * T_inertial_body`
///
/// This composite transform is needed by aerodynamic drag (to express
/// velocity in the structural frame) and force collection (to rotate
/// structural-frame forces to inertial). Matches JEOD
/// `dyn_body_collect.cc` lines 219-221.
///
/// # Arguments
/// - `t_struct_body`: structural-to-body rotation (from mass tree; identity when structure = body)
/// - `t_inertial_body`: inertial-to-body rotation (from attitude quaternion)
pub fn compute_t_inertial_struct(t_struct_body: &DMat3, t_inertial_body: &DMat3) -> DMat3 {
    t_struct_body.transpose() * *t_inertial_body
}

/// Force and torque contributions from interaction models, each in its native frame.
///
/// Used by [`collect_forces`] to accumulate forces/torques with the correct
/// frame rotations. Fields default to zero — set only the contributions that
/// are active for a given entity.
///
/// Frame conventions (matching JEOD `dyn_body_collect.cc`):
/// - Aerodynamic force/torque: **structural** frame
/// - SRP force: **inertial** frame (spherical model); torque: **structural** frame
/// - Gravity gradient torque: **body** frame
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ForceContributions {
    /// Aerodynamic force in structural frame (N).
    pub aero_force_struct: DVec3,
    /// Aerodynamic torque in structural frame (N*m).
    pub aero_torque_struct: DVec3,
    /// SRP force in inertial frame (N) — spherical model.
    pub srp_force_inertial: DVec3,
    /// SRP torque in structural frame (N*m).
    pub srp_torque_struct: DVec3,
    /// Gravity gradient torque in body frame (N*m).
    pub gravity_torque_body: DVec3,
}

/// Collect forces and torques from multiple sources into a single [`TotalForce`].
///
/// Faithfully ports the frame pipeline from JEOD `dyn_body_collect.cc`:
///
/// **Forces** (collected in structural frame, rotated to inertial):
///   `T_inertial_struct^T * aero_force_struct` + `srp_force_inertial`
///
/// **Torques** (collected in structural frame, rotated to body):
///   `T_struct_body * aero_torque_struct` + `T_struct_body * srp_torque_struct` +
///   `gravity_torque_body`
///
/// Where `T_inertial_struct = T_struct_body^T * T_inertial_body`.
///
/// # Arguments
/// - `contributions`: force/torque values from each interaction model
/// - `t_struct_body`: rotation from structural to body frame (from mass tree)
/// - `t_inertial_body`: rotation from inertial to body frame (from attitude quaternion)
// JEOD_INV: DB.28 — forces collected in structural frame, rotated to inertial at root
// JEOD_INV: DB.29 — torques collected in structural frame, rotated to body at root
pub fn collect_forces(
    contributions: &ForceContributions,
    t_struct_body: &DMat3,
    t_inertial_body: &DMat3,
) -> TotalForce {
    let t_inertial_struct = compute_t_inertial_struct(t_struct_body, t_inertial_body);

    // Forces: structural → inertial via T_inertial_struct^T, plus inertial-frame SRP
    let force = t_inertial_struct.transpose() * contributions.aero_force_struct
        + contributions.srp_force_inertial;

    // Torques: structural → body via T_struct_body, plus body-frame gravity torque
    // JEOD dyn_body_collect.cc line 250: T_struct_body * torq_struct → torq_body
    let torque = *t_struct_body * contributions.aero_torque_struct
        + *t_struct_body * contributions.srp_torque_struct
        + contributions.gravity_torque_body;

    TotalForce { force, torque }
}

/// Compute translational-only frame derivatives (no rotational dynamics).
///
/// For vehicles without rotational state or mass properties, only the
/// translational acceleration is meaningful. Rotational acceleration is zero.
///
/// # Arguments
/// - `non_grav_force`: accumulated non-gravity force in inertial frame (N)
/// - `inverse_mass`: precomputed 1/mass (1/kg)
/// - `grav_accel`: gravitational acceleration in m/s^2 (inertial frame)
// JEOD_INV: FD.01 — trans_accel = non_grav_accel + grav_accel
pub fn compute_translational_derivatives(
    non_grav_force: DVec3,
    inverse_mass: f64,
    grav_accel: DVec3,
) -> FrameDerivatives {
    let non_grav_accel = if non_grav_force == DVec3::ZERO {
        DVec3::ZERO
    } else {
        compute_translational_acceleration(non_grav_force, inverse_mass)
    };
    FrameDerivatives {
        trans_accel: non_grav_accel + grav_accel,
        rot_accel: DVec3::ZERO,
    }
}

/// Compute translational and rotational frame derivatives from collected forces.
///
/// Faithfully ports JEOD `dyn_body_collect.cc` lines 224-264:
/// ```text
/// non_grav_accel = F_non_grav / m
/// trans_accel = non_grav_accel + grav_accel
/// rot_accel = I^-1 * (tau - omega x I*omega)
/// ```
///
/// # Arguments
/// - `total_force`: accumulated non-gravity force (inertial) and torque (body)
/// - `inverse_mass`: precomputed 1/mass (1/kg, from MassProperties)
/// - `grav_accel`: gravitational acceleration in m/s^2 (inertial frame)
/// - `inertia`: inertia tensor in body frame (kg*m^2)
/// - `inverse_inertia`: precomputed inverse of inertia tensor
/// - `ang_vel_body`: angular velocity in body frame (rad/s)
// JEOD_INV: FD.01 — trans_accel = non_grav_accel + grav_accel
// JEOD_INV: FD.02 — rot_accel = I^-1 * (tau - omega x I*omega)
pub fn compute_frame_derivatives(
    total_force: &TotalForce,
    inverse_mass: f64,
    grav_accel: DVec3,
    inertia: &DMat3,
    inverse_inertia: &DMat3,
    ang_vel_body: DVec3,
) -> FrameDerivatives {
    // JEOD_INV: DB.18 — F=ma via precomputed inverse_mass (matches JEOD Vector3::scale with inverse_mass)
    let non_grav_accel = if total_force.force == DVec3::ZERO {
        DVec3::ZERO
    } else {
        total_force.force * inverse_mass
    };

    FrameDerivatives {
        trans_accel: non_grav_accel + grav_accel,
        rot_accel: crate::rotational::compute_rotational_acceleration(
            inertia,
            inverse_inertia,
            ang_vel_body,
            total_force.torque,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_gravity_acceleration() {
        let ga = GravityAcceleration::default();
        assert_eq!(ga.grav_accel, DVec3::ZERO);
        assert_eq!(ga.grav_grad, DMat3::ZERO);
        assert_eq!(ga.grav_pot, 0.0);
    }

    #[test]
    fn default_total_force() {
        let tf = TotalForce::default();
        assert_eq!(tf.force, DVec3::ZERO);
        assert_eq!(tf.torque, DVec3::ZERO);
    }

    #[test]
    fn default_frame_derivatives() {
        let fd = FrameDerivatives::default();
        assert_eq!(fd.trans_accel, DVec3::ZERO);
        assert_eq!(fd.rot_accel, DVec3::ZERO);
    }

    #[test]
    fn default_dynamics_config() {
        let dc = DynamicsConfig::default();
        assert!(dc.translational_dynamics);
        assert!(!dc.rotational_dynamics);
        assert!(dc.three_dof);
    }

    #[test]
    fn translational_acceleration_basic() {
        let force = DVec3::new(10.0, 20.0, 30.0);
        let inverse_mass = 1.0 / 5.0;
        let accel = compute_translational_acceleration(force, inverse_mass);
        assert!((accel - DVec3::new(2.0, 4.0, 6.0)).length() < 1e-15);
    }

    #[test]
    fn translational_acceleration_unit_mass() {
        let force = DVec3::new(3.0, -1.0, 7.0);
        let accel = compute_translational_acceleration(force, 1.0);
        assert_eq!(accel, force);
    }

    // ── DynamicsConfig::validate() ──

    #[test]
    fn validate_default_config() {
        DynamicsConfig::default().validate(); // three_dof=true, rotational=false → ok
    }

    #[test]
    fn validate_six_dof_config() {
        let dc = DynamicsConfig {
            translational_dynamics: true,
            rotational_dynamics: true,
            three_dof: false,
        };
        dc.validate(); // should not panic
    }

    #[test]
    #[should_panic(expected = "three_dof=true AND rotational_dynamics=true")]
    fn validate_rejects_three_dof_with_rotational() {
        let dc = DynamicsConfig {
            translational_dynamics: true,
            rotational_dynamics: true,
            three_dof: true,
        };
        dc.validate();
    }

    // ── collect_forces() ──

    #[test]
    fn collect_forces_identity_frames() {
        // With identity frames, structural = body = inertial
        let c = ForceContributions {
            aero_force_struct: DVec3::new(1.0, 0.0, 0.0),
            aero_torque_struct: DVec3::new(0.0, 0.0, 5.0),
            srp_force_inertial: DVec3::new(0.0, 2.0, 0.0),
            srp_torque_struct: DVec3::new(0.0, 3.0, 0.0),
            gravity_torque_body: DVec3::new(0.0, 0.0, 1.0),
        };
        let result = collect_forces(&c, &DMat3::IDENTITY, &DMat3::IDENTITY);
        assert_eq!(result.force, DVec3::new(1.0, 2.0, 0.0));
        assert_eq!(result.torque, DVec3::new(0.0, 3.0, 6.0));
    }

    #[test]
    fn collect_forces_zero_contributions() {
        let c = ForceContributions::default();
        let result = collect_forces(&c, &DMat3::IDENTITY, &DMat3::IDENTITY);
        assert_eq!(result.force, DVec3::ZERO);
        assert_eq!(result.torque, DVec3::ZERO);
    }

    #[test]
    fn collect_forces_rotated_frame() {
        // 90° rotation about z: structural x-axis maps to inertial y-axis
        let rot_z90 = DMat3::from_cols(
            DVec3::new(0.0, 1.0, 0.0),
            DVec3::new(-1.0, 0.0, 0.0),
            DVec3::new(0.0, 0.0, 1.0),
        );
        let c = ForceContributions {
            aero_force_struct: DVec3::new(10.0, 0.0, 0.0),
            ..Default::default()
        };
        // t_struct_body = identity, t_inertial_body = rot_z90
        // T_inertial_struct = I^T * rot_z90 = rot_z90
        // force_inertial = rot_z90^T * [10,0,0]
        let result = collect_forces(&c, &DMat3::IDENTITY, &rot_z90);
        let expected_force = rot_z90.transpose() * DVec3::new(10.0, 0.0, 0.0);
        assert!((result.force - expected_force).length() < 1e-12);
    }

    // ── compute_frame_derivatives() ──

    #[test]
    fn frame_derivatives_gravity_only() {
        let total_force = TotalForce::default(); // zero non-grav force
        let grav = DVec3::new(0.0, 0.0, -9.81);
        let inertia = DMat3::from_diagonal(DVec3::new(10.0, 20.0, 30.0));
        let inv_inertia = DMat3::from_diagonal(DVec3::new(0.1, 0.05, 1.0 / 30.0));

        let fd = compute_frame_derivatives(
            &total_force,
            1.0 / 100.0,
            grav,
            &inertia,
            &inv_inertia,
            DVec3::ZERO,
        );
        assert_eq!(fd.trans_accel, grav);
        assert!(fd.rot_accel.length() < 1e-20);
    }

    #[test]
    fn frame_derivatives_with_force() {
        let total_force = TotalForce {
            force: DVec3::new(100.0, 0.0, 0.0),
            torque: DVec3::ZERO,
        };
        let grav = DVec3::new(0.0, 0.0, -9.81);
        let inertia = DMat3::from_diagonal(DVec3::splat(10.0));
        let inv_inertia = DMat3::from_diagonal(DVec3::splat(0.1));

        let fd = compute_frame_derivatives(
            &total_force,
            1.0 / 50.0,
            grav,
            &inertia,
            &inv_inertia,
            DVec3::ZERO,
        );
        // non_grav_accel = 100 * (1/50) = 2.0 in x
        assert!((fd.trans_accel - DVec3::new(2.0, 0.0, -9.81)).length() < 1e-12);
    }

    #[test]
    fn typed_frame_derivatives_round_trip() {
        use astrodyn_quantities::frame::{RootInertial, TestVehicle};

        let untyped = FrameDerivatives {
            trans_accel: DVec3::new(1.0, 2.0, 3.0),
            rot_accel: DVec3::new(0.1, 0.2, 0.3),
        };
        let typed =
            FrameDerivativesTyped::<RootInertial, TestVehicle>::from_untyped_unchecked(&untyped);
        let back = typed.to_untyped();
        assert_eq!(back, untyped);
    }

    #[test]
    fn typed_total_force_round_trip() {
        use astrodyn_quantities::frame::TestVehicle;

        let untyped = TotalForce {
            force: DVec3::new(100.0, 0.0, 0.0),
            torque: DVec3::new(0.0, 0.5, 0.0),
        };
        // Use the F = RootInertial default for the integration frame.
        let typed = TotalForceTyped::<TestVehicle>::from_untyped_unchecked(&untyped);
        let back = typed.to_untyped();
        assert_eq!(back, untyped);
    }

    #[test]
    fn typed_total_force_default_is_zero() {
        use astrodyn_quantities::frame::TestVehicle;

        let typed = TotalForceTyped::<TestVehicle>::default();
        let back = typed.to_untyped();
        assert_eq!(back, TotalForce::default());
    }

    #[test]
    fn typed_gravity_acceleration_round_trip() {
        use astrodyn_quantities::frame::RootInertial;

        let untyped = GravityAcceleration {
            grav_accel: DVec3::new(0.0, 0.0, -9.81),
            grav_grad: DMat3::IDENTITY * 1e-6,
            grav_pot: 6.25e7,
        };
        let typed = GravityAccelerationTyped::<RootInertial>::from_untyped_unchecked(&untyped);
        let back = typed.to_untyped();
        assert_eq!(back, untyped);
    }
}
