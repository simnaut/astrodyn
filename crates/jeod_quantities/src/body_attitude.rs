//! `BodyAttitude<V>` — a vehicle-tagged inertial-to-body attitude that owns
//! its mutation API.
//!
//! This type exists to make the JEOD `q̇ = -½(ω ⊗ q)` left-multiply
//! convention impossible to violate. The phantom-tag pattern catches
//! quaternion **layout** (`ScalarFirst` vs `ScalarLast`) and
//! **transform handedness** (`LeftTransform` vs `RightTransform`)
//! mismatches, but it cannot discriminate `a ⊗ b` from `b ⊗ a` when both
//! operands have the same type. PR #251 / issue #248 caught a case where
//! an inline `q.multiply(&dq)` (right-multiply) produced an exact
//! 1.708 mrad/s drift in Apollo's detached S3 subtree.
//!
//! `BodyAttitude<V>` removes the operand-swap surface entirely:
//!
//! * the only mutator that uses a quaternion product is
//!   [`BodyAttitude::advance_under_body_rate`], which builds the delta
//!   quaternion internally and never exposes it to the caller;
//! * the only path between attitudes for different vehicles is
//!   [`BodyAttitude::compose_with`] over a typed
//!   [`FrameTransform`]`<BodyFrame<V>, BodyFrame<W>>`;
//! * the inner [`NormalizedQuat`] is reachable read-only via
//!   [`BodyAttitude::as_witness`] for matrix derivation, vector
//!   transform, and glam interop.
//!
//! There is no public `multiply`, `mul`, or `Mul` impl. The compile-fail
//! doctests on this type's public methods exercise that.

use core::marker::PhantomData;

use glam::DVec3;

use crate::aliases::AngularVelocity;
use crate::frame::{BodyFrame, Vehicle};
use crate::frame_transform::FrameTransform;
use crate::quat::{JeodQuat, LeftTransform, NormalizedQuat, ScalarFirst, NORM_LIMIT};

/// Inertial → body attitude of vehicle `V`, witnessed unit-norm and
/// scalar-first / left-transformation (JEOD canonical).
///
/// The phantom `V` makes "this is the attitude of vehicle V"
/// load-bearing: `BodyAttitude<Apollo>` and `BodyAttitude<Iss>` are
/// distinct types and cannot be interchanged.
///
/// ## Public mutation API
///
/// | Method | What it does |
/// |---|---|
/// | [`advance_under_body_rate`](Self::advance_under_body_rate) | step under body-frame ω, dt — owns the left-multiply |
/// | [`compose_with`](Self::compose_with) | typed body→body change-of-frame |
///
/// ## What is *not* available
///
/// There is no `multiply`, `mul`, `Mul` impl, or any other way to feed
/// two `BodyAttitude` values (or a `BodyAttitude` and a raw `JeodQuat`)
/// into a quaternion product. The compiler rejects it.
///
/// **Bare `multiply` between two attitudes does not compile** — this is
/// the type-level guarantee that supersedes PR #251's structural fix:
///
/// ```compile_fail
/// use jeod_quantities::prelude::*;
/// let a: BodyAttitude<SelfRef> = BodyAttitude::identity();
/// let b: BodyAttitude<SelfRef> = BodyAttitude::identity();
/// // No `multiply` method on BodyAttitude:
/// let _ = a.multiply(&b);
/// ```
///
/// **Multiplying an attitude with a raw `JeodQuat` does not compile**
/// either — preventing the exact `q.multiply(&dq)` shape that produced
/// the SIM_Apollo S3 attitude drift:
///
/// ```compile_fail
/// use jeod_quantities::prelude::*;
/// let a: BodyAttitude<SelfRef> = BodyAttitude::identity();
/// let dq: JeodQuat = JeodQuat::identity();
/// let _ = a.multiply(&dq);
/// ```
///
/// **Assigning an attitude across vehicle phantoms does not compile** —
/// `BodyAttitude<Iss>` is not `BodyAttitude<Soyuz>`:
///
/// ```compile_fail
/// use jeod_quantities::prelude::*;
/// use jeod_quantities::define_vehicle;
/// define_vehicle!(Iss);
/// define_vehicle!(Soyuz);
/// let iss: BodyAttitude<Iss> = BodyAttitude::identity();
/// let _soyuz: BodyAttitude<Soyuz> = iss;   // phantom mismatch
/// ```
///
/// **Advancing under the wrong body's angular velocity does not
/// compile** — `AngularVelocity<BodyFrame<Iss>>` is not
/// `AngularVelocity<BodyFrame<Soyuz>>`:
///
/// ```compile_fail
/// use jeod_quantities::prelude::*;
/// use jeod_quantities::define_vehicle;
/// define_vehicle!(Iss);
/// define_vehicle!(Soyuz);
/// let iss: BodyAttitude<Iss> = BodyAttitude::identity();
/// let omega_soyuz: AngularVelocity<BodyFrame<Soyuz>> = Qty3::zero();
/// let _ = iss.advance_under_body_rate(omega_soyuz, 1.0);
/// ```
///
/// **The positive control compiles** — same vehicle, typed angular
/// velocity, integration goes through:
///
/// ```
/// use jeod_quantities::prelude::*;
/// let a: BodyAttitude<SelfRef> = BodyAttitude::identity();
/// let omega: AngularVelocity<BodyFrame<SelfRef>> = Qty3::zero();
/// let _ = a.advance_under_body_rate(omega, 1.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BodyAttitude<V: Vehicle> {
    q: NormalizedQuat<ScalarFirst, LeftTransform>,
    _v: PhantomData<V>,
}

impl<V: Vehicle> BodyAttitude<V> {
    /// The identity attitude (zero rotation from inertial to body).
    #[inline]
    pub fn identity() -> Self {
        Self {
            q: NormalizedQuat::new(JeodQuat::identity()).expect("identity quaternion is unit-norm"),
            _v: PhantomData,
        }
    }

    /// Wrap a pre-witnessed normalized quaternion as the inertial-to-body
    /// attitude of vehicle `V`. The caller asserts the quaternion has
    /// inertial-to-body semantics (this cannot be checked by the type
    /// system without further phantom tags on the quaternion itself).
    #[inline]
    pub fn from_witness(q: NormalizedQuat<ScalarFirst, LeftTransform>) -> Self {
        Self { q, _v: PhantomData }
    }

    /// Read-only view of the underlying normalized quaternion.
    ///
    /// Use for matrix derivation
    /// (`as_witness().left_quat_to_transformation()`), vector transform
    /// (`as_witness().left_quat_transform(v)`), and glam interop
    /// (`as_witness().inner().to_glam()`). The witness cannot be mutated
    /// without producing a new `BodyAttitude` — that is the point.
    #[inline]
    pub const fn as_witness(&self) -> &NormalizedQuat<ScalarFirst, LeftTransform> {
        &self.q
    }

    /// Convenience: copy out the inner [`JeodQuat`] for boundary code
    /// that hasn't yet been migrated to typed quaternions (e.g.,
    /// `RefFrameState::rot::q_parent_this`).
    #[inline]
    pub fn to_jeod_quat(&self) -> JeodQuat {
        self.q.inner()
    }

    /// Step the attitude under a body-frame angular velocity for `dt`
    /// seconds.
    ///
    /// Implements `q̇ = -½(ω ⊗ q)` via the closed-form integrator
    ///
    /// ```text
    /// dq = (cos(|ω|·dt/2),  -ω̂ · sin(|ω|·dt/2))   // left-quat δ
    /// q_new = dq ⊗ q                              // LEFT-multiply
    /// ```
    ///
    /// followed by a Padé-approximant renormalization that preserves the
    /// scalar sign (no canonical-hemisphere flip — sign continuity
    /// across integrator steps matters).
    ///
    /// Numerical behavior:
    ///
    /// * `|ω| == 0` → returns `self` unchanged (no normalization). Bodies
    ///   at rest do not accumulate float roundoff.
    /// * `|ω|·dt < ~π` → optimal accuracy regime (one rotation per step).
    ///
    /// This is the **only** public mutator that uses a quaternion
    /// product, and the product is built and consumed inside this
    /// function — the caller never writes a `multiply` call.
    pub fn advance_under_body_rate(self, ang_vel: AngularVelocity<BodyFrame<V>>, dt: f64) -> Self {
        let omega: DVec3 = ang_vel.raw_si();
        let omega_norm = omega.length();
        if omega_norm == 0.0 {
            return self;
        }
        let half_angle = omega_norm * dt * 0.5;
        let s = half_angle.sin() / omega_norm;
        let c = half_angle.cos();
        // dq = exp(-½ [0, ω] dt) = (cos(θ/2), -ω̂ sin(θ/2)) — same
        // convention as `JeodQuat::left_quat_from_eigen_rotation(|ω|·dt,
        // ω̂)`. Built inline to skip the redundant normalize inside that
        // constructor; we normalize the final product below.
        let dq = JeodQuat::new(c, -omega.x * s, -omega.y * s, -omega.z * s);
        // JEOD_INV: QT.05 — body-rate attitude advance uses LEFT-multiply
        // (`q̇ = -½(ω ⊗ q)`). Operand order is owned by this method, not
        // the caller.
        let mut q_new = dq.multiply(&self.q.inner());
        normalize_integ(&mut q_new);
        // `normalize_integ` brings `q_new` back to within
        // `NormalizedQuat::DEFAULT_TOLERANCE` of unit norm, so the witness
        // constructor's sqrt-based norm check passes by construction —
        // any failure here means `normalize_integ` itself is broken (or
        // somehow received a NaN), and the panic below is the correct
        // loud-failure response per the project's fail-loudly rule.
        let witnessed = NormalizedQuat::new(q_new)
            .expect("normalize_integ leaves quaternion within unit-norm tolerance");
        Self {
            q: witnessed,
            _v: PhantomData,
        }
    }

    /// Compose with a typed body-to-body rotation, producing the
    /// inertial-to-body attitude for the new vehicle `W`.
    ///
    /// Given `q_inertial_V` (this) and `T_V_to_W` (a
    /// `FrameTransform<BodyFrame<V>, BodyFrame<W>>`), the result is
    /// `q_inertial_W = T_V_to_W ⊗ q_inertial_V` under JEOD's
    /// left-transformation convention. The multiply order is owned by
    /// this method.
    ///
    /// This is the **only** way to produce a `BodyAttitude<W>` from a
    /// `BodyAttitude<V>` when `W ≠ V` — there is no raw quaternion
    /// product on `BodyAttitude`.
    pub fn compose_with<W: Vehicle>(
        self,
        rel: FrameTransform<BodyFrame<V>, BodyFrame<W>>,
    ) -> BodyAttitude<W> {
        // JEOD_INV: QT.06 — body-to-body composition uses LEFT-multiply
        // of the relation onto the existing attitude (`q_W = T_VW ⊗ q_V`).
        let composed = rel.quat().inner().multiply(&self.q.inner());
        let witnessed = NormalizedQuat::renormalize(composed)
            .expect("composition of two unit quaternions is finite and non-zero");
        BodyAttitude {
            q: witnessed,
            _v: PhantomData,
        }
    }
}

impl<V: Vehicle> Default for BodyAttitude<V> {
    #[inline]
    fn default() -> Self {
        Self::identity()
    }
}

/// Renormalize a quaternion in place using JEOD's `normalize_integ`
/// algorithm (Padé fast path near unit norm, sqrt fallback otherwise).
///
/// Unlike [`JeodQuat::normalize`], this does **not** force the scalar
/// non-negative — sign continuity across integrator steps is preserved.
/// Faithful port of JEOD `quat_norm.cc:83-101`.
///
/// Kept private to `body_attitude` because the only legitimate caller
/// is [`BodyAttitude::advance_under_body_rate`]. The integrator-side
/// `jeod_dynamics::normalize_integ` (a `pub` function used by RK4 /
/// RKF45 / Gauss-Jackson) is the analog for raw `JeodQuat` values.
#[inline]
fn normalize_integ(q: &mut JeodQuat) {
    let qmagsq = q.norm_sq();
    // JEOD_INV: QT.04 — cannot normalize a zero quaternion
    assert!(
        qmagsq > 0.0,
        "BodyAttitude::advance_under_body_rate produced zero-magnitude quaternion"
    );
    let diff1 = 1.0 - qmagsq;
    let fact = if diff1 > -NORM_LIMIT && diff1 < NORM_LIMIT {
        2.0 / (1.0 + qmagsq)
    } else {
        1.0 / qmagsq.sqrt()
    };
    q.data[0] *= fact;
    q.data[1] *= fact;
    q.data[2] *= fact;
    q.data[3] *= fact;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::SelfRef;
    use crate::qty3::Qty3;

    const TOL: f64 = 1e-14;

    #[test]
    fn identity_is_unit_quat() {
        let a: BodyAttitude<SelfRef> = BodyAttitude::identity();
        let q = a.to_jeod_quat();
        assert!((q.norm() - 1.0).abs() < TOL);
        assert_eq!(q.scalar(), 1.0);
        assert_eq!(q.vector(), DVec3::ZERO);
    }

    #[test]
    fn zero_omega_is_no_op() {
        let q_init = JeodQuat::left_quat_from_eigen_rotation(0.5, DVec3::Z);
        let a: BodyAttitude<SelfRef> =
            BodyAttitude::from_witness(NormalizedQuat::new(q_init).unwrap());
        let omega: AngularVelocity<BodyFrame<SelfRef>> = Qty3::zero();
        let advanced = a.advance_under_body_rate(omega, 1.0);
        assert_eq!(advanced.to_jeod_quat().data, q_init.data);
    }

    /// Regression: this is the property that PR #251 fixed at runtime
    /// and that the type system now also enforces structurally. We keep
    /// a numeric check here so the integrator math stays calibrated
    /// against JEOD even after the helper is reorganized.
    #[test]
    fn advance_uses_left_multiply_not_right() {
        // Non-trivial initial pose so the operand order matters.
        let q_init = JeodQuat::left_quat_from_eigen_rotation(0.5, DVec3::Z);
        let omega_vec = DVec3::new(0.0, 0.001_138_5, 0.0);
        let dt = 0.02_f64;

        let attitude: BodyAttitude<SelfRef> =
            BodyAttitude::from_witness(NormalizedQuat::new(q_init).unwrap());
        let omega: AngularVelocity<BodyFrame<SelfRef>> = Qty3::from_raw_si(omega_vec);
        let q_out = attitude.advance_under_body_rate(omega, dt).to_jeod_quat();

        // Expected: dq ⊗ q_init (LEFT-multiply, JEOD convention).
        let omega_norm = omega_vec.length();
        let half = omega_norm * dt * 0.5;
        let s = half.sin() / omega_norm;
        let c = half.cos();
        let dq = JeodQuat::new(c, -omega_vec.x * s, -omega_vec.y * s, -omega_vec.z * s);
        let mut expected_left = dq.multiply(&q_init);
        normalize_integ(&mut expected_left);
        let mut expected_right = q_init.multiply(&dq);
        normalize_integ(&mut expected_right);

        for i in 0..4 {
            assert!(
                (q_out.data[i] - expected_left.data[i]).abs() < TOL,
                "advance_under_body_rate must match LEFT-multiply convention at component {i}"
            );
        }
        // And differs measurably from the wrong (RIGHT-multiply) order.
        let mut diverged = false;
        for i in 0..4 {
            if (expected_left.data[i] - expected_right.data[i]).abs() > 1e-9 {
                diverged = true;
                break;
            }
        }
        assert!(
            diverged,
            "test setup is degenerate: left and right multiply produce equal results"
        );
    }

    #[test]
    fn small_steps_match_one_shot() {
        // Linear regime: 100 small steps should approximate one big step.
        let q_init = JeodQuat::left_quat_from_eigen_rotation(0.3, DVec3::Y);
        let omega_vec = DVec3::new(0.001, -0.0005, 0.0008);
        let total_dt = 1.0_f64;
        let n = 100_usize;
        let small_dt = total_dt / n as f64;

        let mut q_iter: BodyAttitude<SelfRef> =
            BodyAttitude::from_witness(NormalizedQuat::new(q_init).unwrap());
        let omega: AngularVelocity<BodyFrame<SelfRef>> = Qty3::from_raw_si(omega_vec);
        for _ in 0..n {
            q_iter = q_iter.advance_under_body_rate(omega, small_dt);
        }
        let q_one_shot: BodyAttitude<SelfRef> =
            BodyAttitude::from_witness(NormalizedQuat::new(q_init).unwrap())
                .advance_under_body_rate(omega, total_dt);
        let a = q_iter.to_jeod_quat();
        let b = q_one_shot.to_jeod_quat();
        for i in 0..4 {
            assert!(
                (a.data[i] - b.data[i]).abs() < 1e-6,
                "small-step vs one-shot diverge at component {i}: {} vs {}",
                a.data[i],
                b.data[i]
            );
        }
    }
}
