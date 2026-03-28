use glam::{DMat3, DVec3};
use jeod_math::JeodQuat;

// JEOD_INV: RF.06 — position/velocity in parent coordinates (structural convention)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RefFrameTrans {
    pub position: DVec3, // m, in parent frame
    pub velocity: DVec3, // m/s, in parent frame
}

impl Default for RefFrameTrans {
    fn default() -> Self {
        Self {
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
        }
    }
}

// JEOD_INV: RF.07 — Q_parent_this is left-transformation quaternion (JEOD convention)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RefFrameRot {
    pub q_parent_this: JeodQuat, // left transformation quaternion
    pub t_parent_this: DMat3,    // transformation matrix
    pub ang_vel_this: DVec3,     // rad/s, in this frame
}

impl Default for RefFrameRot {
    fn default() -> Self {
        Self {
            q_parent_this: JeodQuat::identity(),
            t_parent_this: DMat3::IDENTITY,
            ang_vel_this: DVec3::ZERO,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RefFrameState {
    pub trans: RefFrameTrans,
    pub rot: RefFrameRot,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RefFrameInfo {
    pub name: String,
    pub kind: RefFrameKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefFrameKind {
    Inertial,
    PlanetFixed,
    Body,
}

impl RefFrameState {
    /// Negate (invert) a frame state.
    ///
    /// If `source` represents the state of frame B relative to frame A (A->B),
    /// the result represents the state of frame A relative to frame B (B->A).
    ///
    /// Ported from JEOD `ref_frame_state.cc` negate method.
    ///
    /// Convention:
    /// - `trans.position`: position of "this" frame's origin in parent frame coords
    /// - `trans.velocity`: velocity of "this" frame's origin in parent frame coords
    /// - `rot.t_parent_this`: transforms vectors FROM parent TO this frame
    /// - `rot.ang_vel_this`: angular velocity of this frame relative to parent, in this frame coords
    pub fn negate(source: &RefFrameState) -> RefFrameState {
        // Rotation: transpose/conjugate
        let t_new = source.rot.t_parent_this.transpose();
        let q_new = source.rot.q_parent_this.conjugate();

        // Angular velocity: -(T_new * source.ang_vel)
        // source.ang_vel is in source's "this" frame.
        // For negated: we want ang_vel of old_parent relative to old_this, in old_parent coords.
        // = -(T_source^T * source.ang_vel) = -(t_new * source.ang_vel)
        let ang_vel_new = -(t_new * source.rot.ang_vel_this);

        // Position: -(T_source * source.position)
        // source.position is position of source's "this" in source's "parent" coords.
        // T_source transforms from parent to this.
        // Result position = position of old_parent in old_this coords = -(T_source * source.pos)
        let pos_new = -(source.rot.t_parent_this * source.trans.position);

        // Velocity: -(omega_AB x pos_BA) - T_AB * v_AB
        // Derived from d/dt(-T * r_AB) using dT/dt = -[omega]_x * T:
        //   v_BA = omega x (T * r_AB) - T * v_AB = omega x (-pos_BA) - T * v_AB
        //        = -(omega x pos_BA) - T * v_AB
        let t_vel = source.rot.t_parent_this * source.trans.velocity;
        let vel_new = -source.rot.ang_vel_this.cross(pos_new) - t_vel;

        RefFrameState {
            trans: RefFrameTrans {
                position: pos_new,
                velocity: vel_new,
            },
            rot: RefFrameRot {
                q_parent_this: q_new,
                t_parent_this: t_new,
                ang_vel_this: ang_vel_new,
            },
        }
    }

    // JEOD_INV: RF.03 — quaternion normalized after every composition (line 117)
    // JEOD_INV: RF.04 — T_parent_this recomputed from matrix multiplication (line 113)
    /// Compose self (A->B) with s_bc (B->C) to produce A->C.
    ///
    /// "Increment right": given self = S_{A:B} and s_bc = S_{B:C},
    /// compute and return S_{A:C}.
    ///
    /// Ported from JEOD `ref_frame_state.cc` incr_right / compose_state.
    pub fn incr_right(&self, s_bc: &RefFrameState) -> RefFrameState {
        // Rotation: T_{A:C} = T_{B:C} * T_{A:B}
        let t_ac = s_bc.rot.t_parent_this * self.rot.t_parent_this;

        // Quaternion: Q_{A:C} = Q_{B:C} * Q_{A:B}, then normalize
        let mut q_ac = s_bc.rot.q_parent_this.multiply(&self.rot.q_parent_this);
        q_ac.normalize();

        // Angular velocity: omega_{A:C} (in C frame) = T_{B:C} * omega_{A:B} + omega_{B:C}
        // self.ang_vel_this = omega of B relative to A, in B coords
        // s_bc.ang_vel_this = omega of C relative to B, in C coords
        // T_{B:C} transforms from B to C
        let ang_vel_ac = s_bc.rot.t_parent_this * self.rot.ang_vel_this + s_bc.rot.ang_vel_this;

        // Position: x_{A:C} (in A coords) = x_{A:B} + T_{A:B}^T * x_{B:C}
        // T_{A:B} transforms from A to B, so T_{A:B}^T transforms from B to A
        let pos_ac =
            self.trans.position + self.rot.t_parent_this.transpose() * s_bc.trans.position;

        // Velocity: v_{A:C} = v_{A:B} + T_{A:B}^T * (v_{B:C} + omega_{A:B} x x_{B:C})
        // omega_{A:B} is in B coords, x_{B:C} is in B coords, so cross product is in B coords
        // T_{A:B}^T transforms from B to A (parent frame of A:B)
        let omega_cross_pos = self.rot.ang_vel_this.cross(s_bc.trans.position);
        let vel_ac = self.trans.velocity
            + self.rot.t_parent_this.transpose() * (s_bc.trans.velocity + omega_cross_pos);

        RefFrameState {
            trans: RefFrameTrans {
                position: pos_ac,
                velocity: vel_ac,
            },
            rot: RefFrameRot {
                q_parent_this: q_ac,
                t_parent_this: t_ac,
                ang_vel_this: ang_vel_ac,
            },
        }
    }

    /// Compose s_ab (A->B) with self (B->C) to produce A->C, updating self in place.
    ///
    /// "Increment left": given s_ab = S_{A:B} and self = S_{B:C},
    /// update self to become S_{A:C}.
    ///
    /// Same math as incr_right but with different roles.
    pub fn incr_left(&mut self, s_ab: &RefFrameState) {
        let result = s_ab.incr_right(self);
        *self = result;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{DMat3, DVec3};
    use jeod_math::test_utils::{approx_eq_f64, approx_eq_mat3, approx_eq_vec3};
    use jeod_math::JeodQuat;
    use std::f64::consts::FRAC_PI_2;

    const TOL: f64 = 1e-12;

    /// Helper: create a RefFrameState with a rotation about Z axis and a position offset.
    fn make_state(angle_z: f64, pos: DVec3, vel: DVec3, ang_vel: DVec3) -> RefFrameState {
        let q = JeodQuat::left_quat_from_eigen_rotation(angle_z, DVec3::Z);
        let t = q.left_quat_to_transformation();
        RefFrameState {
            trans: RefFrameTrans {
                position: pos,
                velocity: vel,
            },
            rot: RefFrameRot {
                q_parent_this: q,
                t_parent_this: t,
                ang_vel_this: ang_vel,
            },
        }
    }

    /// Helper: create a RefFrameState with arbitrary axis rotation.
    fn make_state_axis(
        angle: f64,
        axis: DVec3,
        pos: DVec3,
        vel: DVec3,
        ang_vel: DVec3,
    ) -> RefFrameState {
        let q = JeodQuat::left_quat_from_eigen_rotation(angle, axis);
        let t = q.left_quat_to_transformation();
        RefFrameState {
            trans: RefFrameTrans {
                position: pos,
                velocity: vel,
            },
            rot: RefFrameRot {
                q_parent_this: q,
                t_parent_this: t,
                ang_vel_this: ang_vel,
            },
        }
    }

    // -----------------------------------------------------------------
    // compose identity with any state -> same state
    // -----------------------------------------------------------------
    #[test]
    fn compose_identity_left() {
        let s = make_state(
            0.5,
            DVec3::new(1e6, 2e6, 3e6),
            DVec3::new(100.0, 200.0, 300.0),
            DVec3::new(0.01, 0.02, 0.03),
        );
        let identity = RefFrameState::default();

        // identity.incr_right(&s) should give s
        let result = identity.incr_right(&s);
        assert!(
            approx_eq_vec3(result.trans.position, s.trans.position, TOL),
            "Position mismatch: {:?} vs {:?}",
            result.trans.position,
            s.trans.position
        );
        assert!(
            approx_eq_vec3(result.trans.velocity, s.trans.velocity, TOL),
            "Velocity mismatch: {:?} vs {:?}",
            result.trans.velocity,
            s.trans.velocity
        );
        assert!(
            approx_eq_mat3(&result.rot.t_parent_this, &s.rot.t_parent_this, TOL),
            "T mismatch"
        );
        assert!(
            approx_eq_vec3(result.rot.ang_vel_this, s.rot.ang_vel_this, TOL),
            "Ang vel mismatch"
        );
    }

    #[test]
    fn compose_identity_right() {
        let s = make_state(
            0.5,
            DVec3::new(1e6, 2e6, 3e6),
            DVec3::new(100.0, 200.0, 300.0),
            DVec3::new(0.01, 0.02, 0.03),
        );
        let identity = RefFrameState::default();

        // s.incr_right(&identity) should give s
        let result = s.incr_right(&identity);
        assert!(
            approx_eq_vec3(result.trans.position, s.trans.position, TOL),
            "Position mismatch: {:?} vs {:?}",
            result.trans.position,
            s.trans.position
        );
        assert!(
            approx_eq_vec3(result.trans.velocity, s.trans.velocity, TOL),
            "Velocity mismatch: {:?} vs {:?}",
            result.trans.velocity,
            s.trans.velocity
        );
        assert!(
            approx_eq_mat3(&result.rot.t_parent_this, &s.rot.t_parent_this, TOL),
            "T mismatch"
        );
        assert!(
            approx_eq_vec3(result.rot.ang_vel_this, s.rot.ang_vel_this, TOL),
            "Ang vel mismatch"
        );
    }

    // -----------------------------------------------------------------
    // compose S with negate(S) -> identity
    // -----------------------------------------------------------------
    #[test]
    fn compose_with_negate_gives_identity() {
        let s = make_state(
            1.2,
            DVec3::new(5e6, -3e6, 1e6),
            DVec3::new(500.0, -300.0, 100.0),
            DVec3::new(0.05, -0.02, 0.01),
        );
        let s_neg = RefFrameState::negate(&s);

        // S composed with negate(S) should give identity
        let result = s.incr_right(&s_neg);

        assert!(
            approx_eq_vec3(result.trans.position, DVec3::ZERO, 1e-6),
            "Position should be ~0, got {:?}",
            result.trans.position
        );
        assert!(
            approx_eq_vec3(result.trans.velocity, DVec3::ZERO, 1e-6),
            "Velocity should be ~0, got {:?}",
            result.trans.velocity
        );
        assert!(
            approx_eq_mat3(&result.rot.t_parent_this, &DMat3::IDENTITY, 1e-10),
            "T should be ~I, got {:?}",
            result.rot.t_parent_this
        );
        assert!(
            approx_eq_vec3(result.rot.ang_vel_this, DVec3::ZERO, 1e-10),
            "Ang vel should be ~0, got {:?}",
            result.rot.ang_vel_this
        );
    }

    #[test]
    fn negate_with_compose_gives_identity_reversed() {
        let s = make_state_axis(
            0.8,
            DVec3::new(1.0, 1.0, 1.0).normalize(),
            DVec3::new(1e7, 0.0, 0.0),
            DVec3::new(1000.0, 2000.0, -500.0),
            DVec3::new(0.0, 0.0, 7.292e-5), // ~Earth rotation rate
        );
        let s_neg = RefFrameState::negate(&s);

        // negate(S) composed with S should also give identity
        let result = s_neg.incr_right(&s);

        assert!(
            approx_eq_vec3(result.trans.position, DVec3::ZERO, 1e-4),
            "Position should be ~0, got {:?}",
            result.trans.position
        );
        assert!(
            approx_eq_vec3(result.trans.velocity, DVec3::ZERO, 1e-4),
            "Velocity should be ~0, got {:?}",
            result.trans.velocity
        );
        assert!(
            approx_eq_mat3(&result.rot.t_parent_this, &DMat3::IDENTITY, 1e-10),
            "T should be ~I, got {:?}",
            result.rot.t_parent_this
        );
        assert!(
            approx_eq_vec3(result.rot.ang_vel_this, DVec3::ZERO, 1e-10),
            "Ang vel should be ~0, got {:?}",
            result.rot.ang_vel_this
        );
    }

    // -----------------------------------------------------------------
    // Double negate returns original
    // -----------------------------------------------------------------
    #[test]
    fn double_negate_is_identity_operation() {
        let s = make_state(
            0.7,
            DVec3::new(2e6, 4e6, -1e6),
            DVec3::new(300.0, -150.0, 75.0),
            DVec3::new(0.01, 0.005, -0.003),
        );

        let s_neg = RefFrameState::negate(&s);
        let s_double_neg = RefFrameState::negate(&s_neg);

        assert!(
            approx_eq_vec3(s_double_neg.trans.position, s.trans.position, 1e-6),
            "Double negate position: {:?} vs {:?}",
            s_double_neg.trans.position,
            s.trans.position
        );
        assert!(
            approx_eq_vec3(s_double_neg.trans.velocity, s.trans.velocity, 1e-6),
            "Double negate velocity: {:?} vs {:?}",
            s_double_neg.trans.velocity,
            s.trans.velocity
        );
        assert!(
            approx_eq_mat3(
                &s_double_neg.rot.t_parent_this,
                &s.rot.t_parent_this,
                1e-10
            ),
            "Double negate T"
        );
        assert!(
            approx_eq_vec3(s_double_neg.rot.ang_vel_this, s.rot.ang_vel_this, 1e-10),
            "Double negate ang_vel"
        );
    }

    // -----------------------------------------------------------------
    // Three-frame chain: known rotation + offset
    // -----------------------------------------------------------------
    #[test]
    fn three_frame_chain() {
        // Frame A -> B: 90-degree rotation about Z, offset [1000, 0, 0] in A coords
        // No velocity/angular velocity for simplicity
        let s_ab = make_state(
            FRAC_PI_2,
            DVec3::new(1000.0, 0.0, 0.0),
            DVec3::ZERO,
            DVec3::ZERO,
        );

        // Frame B -> C: no rotation, offset [500, 0, 0] in B coords
        let s_bc = make_state(
            0.0,
            DVec3::new(500.0, 0.0, 0.0),
            DVec3::ZERO,
            DVec3::ZERO,
        );

        // Compose: A -> C
        let s_ac = s_ab.incr_right(&s_bc);

        // T_{A:C} = T_{B:C} * T_{A:B}
        // T_{B:C} = I, T_{A:B} = 90deg Z rotation
        // So T_{A:C} = T_{A:B} (90 deg Z rotation)
        assert!(
            approx_eq_mat3(&s_ac.rot.t_parent_this, &s_ab.rot.t_parent_this, TOL),
            "T_AC should equal T_AB since T_BC=I"
        );

        // Position of C in A coords:
        // x_AC = x_AB + T_AB^T * x_BC
        // JEOD 90-deg Z: T (row-major) = [[0,1,0],[-1,0,0],[0,0,1]]
        // T^T (row-major) = [[0,-1,0],[1,0,0],[0,0,1]]
        // In glam col-major: T^T cols = [0,1,0], [-1,0,0], [0,0,1]
        // T^T * [500,0,0] = 500 * [0,1,0] = [0, 500, 0]
        // x_AC = [1000,0,0] + [0,500,0] = [1000, 500, 0]

        let expected_pos = DVec3::new(1000.0, 500.0, 0.0);
        assert!(
            approx_eq_vec3(s_ac.trans.position, expected_pos, TOL),
            "Position A->C: expected {:?}, got {:?}",
            expected_pos,
            s_ac.trans.position
        );
    }

    #[test]
    fn three_frame_chain_with_velocity() {
        // Frame A -> B: 90-degree rotation about Z, offset, with angular velocity
        let omega_ab = DVec3::new(0.0, 0.0, 0.1); // rad/s in B frame
        let s_ab = RefFrameState {
            trans: RefFrameTrans {
                position: DVec3::new(1000.0, 0.0, 0.0),
                velocity: DVec3::new(10.0, 0.0, 0.0),
            },
            rot: {
                let q = JeodQuat::left_quat_from_eigen_rotation(FRAC_PI_2, DVec3::Z);
                let t = q.left_quat_to_transformation();
                RefFrameRot {
                    q_parent_this: q,
                    t_parent_this: t,
                    ang_vel_this: omega_ab,
                }
            },
        };

        // Frame B -> C: no rotation, offset [500, 0, 0] in B, velocity [5,0,0] in B
        let s_bc = RefFrameState {
            trans: RefFrameTrans {
                position: DVec3::new(500.0, 0.0, 0.0),
                velocity: DVec3::new(5.0, 0.0, 0.0),
            },
            rot: RefFrameRot::default(),
        };

        let s_ac = s_ab.incr_right(&s_bc);

        // Velocity: v_AC = v_AB + T_AB^T * (v_BC + omega_AB x x_BC)
        // omega_AB x x_BC = [0,0,0.1] x [500,0,0] = [0, 50, 0] (in B coords)
        // v_BC + omega x pos = [5,0,0] + [0,50,0] = [5, 50, 0] (in B coords)
        // T^T cols (glam) = [0,1,0], [-1,0,0], [0,0,1]
        // T^T * [5, 50, 0] = 5*[0,1,0] + 50*[-1,0,0] = [-50, 5, 0]
        // v_AC = [10, 0, 0] + [-50, 5, 0] = [-40, 5, 0]
        let expected_vel = DVec3::new(-40.0, 5.0, 0.0);
        assert!(
            approx_eq_vec3(s_ac.trans.velocity, expected_vel, 1e-10),
            "Velocity A->C: expected {:?}, got {:?}",
            expected_vel,
            s_ac.trans.velocity
        );
    }

    // -----------------------------------------------------------------
    // incr_left matches incr_right
    // -----------------------------------------------------------------
    #[test]
    fn incr_left_matches_incr_right() {
        let s_ab = make_state(
            0.3,
            DVec3::new(1e6, 2e6, 0.0),
            DVec3::new(100.0, 50.0, 0.0),
            DVec3::new(0.0, 0.0, 0.01),
        );
        let s_bc = make_state(
            -0.7,
            DVec3::new(5e5, 0.0, 1e5),
            DVec3::new(20.0, 10.0, 5.0),
            DVec3::new(0.001, 0.0, 0.002),
        );

        let result_right = s_ab.incr_right(&s_bc);

        let mut s_bc_copy = s_bc;
        s_bc_copy.incr_left(&s_ab);

        assert!(
            approx_eq_vec3(s_bc_copy.trans.position, result_right.trans.position, TOL),
            "incr_left position mismatch"
        );
        assert!(
            approx_eq_vec3(s_bc_copy.trans.velocity, result_right.trans.velocity, TOL),
            "incr_left velocity mismatch"
        );
        assert!(
            approx_eq_mat3(
                &s_bc_copy.rot.t_parent_this,
                &result_right.rot.t_parent_this,
                TOL
            ),
            "incr_left T mismatch"
        );
        assert!(
            approx_eq_vec3(
                s_bc_copy.rot.ang_vel_this,
                result_right.rot.ang_vel_this,
                TOL
            ),
            "incr_left ang_vel mismatch"
        );
    }

    // -----------------------------------------------------------------
    // Negate of identity is identity
    // -----------------------------------------------------------------
    #[test]
    fn negate_identity_is_identity() {
        let identity = RefFrameState::default();
        let neg = RefFrameState::negate(&identity);

        assert!(
            approx_eq_vec3(neg.trans.position, DVec3::ZERO, TOL),
            "Negate identity position"
        );
        assert!(
            approx_eq_vec3(neg.trans.velocity, DVec3::ZERO, TOL),
            "Negate identity velocity"
        );
        assert!(
            approx_eq_mat3(&neg.rot.t_parent_this, &DMat3::IDENTITY, TOL),
            "Negate identity T"
        );
        assert!(
            approx_eq_vec3(neg.rot.ang_vel_this, DVec3::ZERO, TOL),
            "Negate identity ang_vel"
        );
    }

    // -----------------------------------------------------------------
    // Negate: pure translation
    // -----------------------------------------------------------------
    #[test]
    fn negate_pure_translation() {
        // No rotation, just a position offset. Negate should give -position.
        let s = RefFrameState {
            trans: RefFrameTrans {
                position: DVec3::new(1000.0, 2000.0, 3000.0),
                velocity: DVec3::new(10.0, 20.0, 30.0),
            },
            rot: RefFrameRot::default(), // identity rotation
        };

        let neg = RefFrameState::negate(&s);

        // With identity rotation: T = I, so negated position = -(I * pos) = -pos
        assert!(
            approx_eq_vec3(
                neg.trans.position,
                DVec3::new(-1000.0, -2000.0, -3000.0),
                TOL
            ),
            "Negate pure translation position"
        );

        // Velocity with identity: ang_vel x pos_new - I * vel = 0 x pos_new - vel = -vel
        assert!(
            approx_eq_vec3(
                neg.trans.velocity,
                DVec3::new(-10.0, -20.0, -30.0),
                TOL
            ),
            "Negate pure translation velocity"
        );
    }

    // -----------------------------------------------------------------
    // Negate: pure rotation
    // -----------------------------------------------------------------
    #[test]
    fn negate_pure_rotation() {
        // No translation, just a rotation
        let q = JeodQuat::left_quat_from_eigen_rotation(1.0, DVec3::new(1.0, 1.0, 1.0).normalize());
        let t = q.left_quat_to_transformation();
        let s = RefFrameState {
            trans: RefFrameTrans::default(),
            rot: RefFrameRot {
                q_parent_this: q,
                t_parent_this: t,
                ang_vel_this: DVec3::new(0.01, 0.02, 0.03),
            },
        };

        let neg = RefFrameState::negate(&s);

        // T^T * T should be I
        let product = neg.rot.t_parent_this * s.rot.t_parent_this;
        assert!(
            approx_eq_mat3(&product, &DMat3::IDENTITY, TOL),
            "T_neg * T should be I"
        );

        // Position should remain zero
        assert!(
            approx_eq_vec3(neg.trans.position, DVec3::ZERO, TOL),
            "Pure rotation negate should have zero position"
        );
    }

    // -----------------------------------------------------------------
    // Composition associativity: (A->B -> C) -> D == A -> (B->C -> D)
    // -----------------------------------------------------------------
    #[test]
    fn composition_associativity() {
        let s_ab = make_state_axis(
            0.3,
            DVec3::X,
            DVec3::new(1e6, 0.0, 0.0),
            DVec3::new(100.0, 0.0, 0.0),
            DVec3::new(0.01, 0.0, 0.0),
        );
        let s_bc = make_state_axis(
            0.5,
            DVec3::Y,
            DVec3::new(0.0, 5e5, 0.0),
            DVec3::new(0.0, 50.0, 0.0),
            DVec3::new(0.0, 0.02, 0.0),
        );
        let s_cd = make_state_axis(
            -0.2,
            DVec3::Z,
            DVec3::new(0.0, 0.0, 3e5),
            DVec3::new(0.0, 0.0, 30.0),
            DVec3::new(0.0, 0.0, 0.005),
        );

        // (A->B -> C) -> D
        let s_ac = s_ab.incr_right(&s_bc);
        let s_ad_left = s_ac.incr_right(&s_cd);

        // A -> (B->C -> D)
        let s_bd = s_bc.incr_right(&s_cd);
        let s_ad_right = s_ab.incr_right(&s_bd);

        assert!(
            approx_eq_vec3(s_ad_left.trans.position, s_ad_right.trans.position, 1e-6),
            "Associativity position: {:?} vs {:?}",
            s_ad_left.trans.position,
            s_ad_right.trans.position
        );
        assert!(
            approx_eq_vec3(s_ad_left.trans.velocity, s_ad_right.trans.velocity, 1e-6),
            "Associativity velocity: {:?} vs {:?}",
            s_ad_left.trans.velocity,
            s_ad_right.trans.velocity
        );
        assert!(
            approx_eq_mat3(
                &s_ad_left.rot.t_parent_this,
                &s_ad_right.rot.t_parent_this,
                1e-10
            ),
            "Associativity T"
        );
        assert!(
            approx_eq_vec3(
                s_ad_left.rot.ang_vel_this,
                s_ad_right.rot.ang_vel_this,
                1e-10
            ),
            "Associativity ang_vel"
        );
    }

    // -----------------------------------------------------------------
    // Default state
    // -----------------------------------------------------------------
    #[test]
    fn default_state_is_identity() {
        let s = RefFrameState::default();
        assert_eq!(s.trans.position, DVec3::ZERO);
        assert_eq!(s.trans.velocity, DVec3::ZERO);
        assert_eq!(s.rot.t_parent_this, DMat3::IDENTITY);
        assert_eq!(s.rot.ang_vel_this, DVec3::ZERO);
        assert!(
            approx_eq_f64(s.rot.q_parent_this.scalar(), 1.0, TOL),
            "Default quaternion scalar"
        );
        assert!(
            approx_eq_vec3(s.rot.q_parent_this.vector(), DVec3::ZERO, TOL),
            "Default quaternion vector"
        );
    }
}
