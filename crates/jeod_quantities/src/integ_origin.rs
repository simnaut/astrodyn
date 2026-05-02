//! Integration-frame origin and the typed shift to root-inertial.
//!
//! `Position<IntegrationFrame>` lives in a body's integration frame — a
//! non-rotating frame whose origin generally differs from the root
//! inertial origin (e.g. integrating a vehicle in `Earth.inertial` when
//! the root frame is `SSB.inertial`).
//!
//! Apply this shift only when the consumer requires root-inertial
//! coordinates — i.e. when it mixes body state with root-inertial source
//! positions (Sun, Moon, gravity sources). Those *shift sites* are:
//! gravity, relativistic corrections, SRP, solar beta, earth lighting.
//!
//! Do **not** shift for atmosphere, drag velocity, LVLH, geodetic, or
//! orbital elements: those operate within a single planet's inertial
//! frame, which is *the body's integration frame* in realistic configs,
//! so `body.trans.position` is already the correct input. They take
//! [`Position<PlanetInertial<P>>`](crate::frame::PlanetInertial) at the
//! typed-sibling API.
//!
//! See issue #255 / `RF.10` in `docs/JEOD_invariants.md` for the
//! shift-vs-no-shift split that motivated this module.

use crate::aliases::{Position, Velocity};
use crate::frame::{IntegrationFrame, RootInertial};

/// The integration-frame origin: position and velocity of the
/// integration-frame origin expressed in root-inertial coordinates.
///
/// Construct from the runner's frame-tree state. `IntegOrigin::zero()` is
/// used when the body integrates in the root frame, so the shift is a
/// no-op.
#[derive(Debug, Clone, Copy, Default)]
pub struct IntegOrigin {
    /// Position of the integration-frame origin in root-inertial coordinates.
    pub position: Position<RootInertial>,
    /// Velocity of the integration-frame origin in root-inertial coordinates.
    pub velocity: Velocity<RootInertial>,
}

impl IntegOrigin {
    /// The zero offset — used when the integration frame coincides with
    /// the root inertial frame.
    #[inline]
    pub fn zero() -> Self {
        Self {
            position: Position::<RootInertial>::zero(),
            velocity: Velocity::<RootInertial>::zero(),
        }
    }

    /// Shift a position from the integration frame to root inertial by
    /// adding the origin's root-inertial position.
    #[inline]
    pub fn shift_position(&self, p: Position<IntegrationFrame>) -> Position<RootInertial> {
        Position::<RootInertial>::from_raw_si(p.raw_si() + self.position.raw_si())
    }

    /// Shift a velocity from the integration frame to root inertial by
    /// adding the origin's root-inertial velocity.
    #[inline]
    pub fn shift_velocity(&self, v: Velocity<IntegrationFrame>) -> Velocity<RootInertial> {
        Velocity::<RootInertial>::from_raw_si(v.raw_si() + self.velocity.raw_si())
    }

    /// Stage-time-interpolated shift to root inertial.
    ///
    /// Used inside RK4 derivative closures where the intermediate
    /// position is sampled at `time_frac * dt` into the step. The
    /// integration-frame origin itself moves under its own velocity, so
    /// the stage-time shift is `origin.position + origin.velocity *
    /// stage_dt` (linear interpolation; matches the arithmetic used by
    /// the runner's gravity stage closure, RF.10). For
    /// integration-frames at rest in root, this collapses to the
    /// step-constant `shift_position`.
    #[inline]
    pub fn shift_position_at_stage(
        &self,
        p: Position<IntegrationFrame>,
        stage_dt: f64,
    ) -> Position<RootInertial> {
        let stage_origin = self.position.raw_si() + self.velocity.raw_si() * stage_dt;
        Position::<RootInertial>::from_raw_si(p.raw_si() + stage_origin)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;

    #[test]
    fn zero_origin_is_identity_for_position() {
        let p = Position::<IntegrationFrame>::from_raw_si(DVec3::new(7e6, 0.0, 0.0));
        let shifted = IntegOrigin::zero().shift_position(p);
        assert_eq!(shifted.raw_si(), p.raw_si());
    }

    #[test]
    fn zero_origin_is_identity_for_velocity() {
        let v = Velocity::<IntegrationFrame>::from_raw_si(DVec3::new(0.0, 7500.0, 0.0));
        let shifted = IntegOrigin::zero().shift_velocity(v);
        assert_eq!(shifted.raw_si(), v.raw_si());
    }

    #[test]
    fn nonzero_origin_adds_to_position() {
        let o = IntegOrigin {
            position: Position::<RootInertial>::from_raw_si(DVec3::new(1.5e11, 0.0, 0.0)),
            velocity: Velocity::<RootInertial>::zero(),
        };
        let p_integ = Position::<IntegrationFrame>::from_raw_si(DVec3::new(7e6, 0.0, 0.0));
        let p_root = o.shift_position(p_integ);
        assert_eq!(p_root.raw_si(), DVec3::new(1.5e11 + 7e6, 0.0, 0.0));
    }

    #[test]
    fn nonzero_origin_adds_to_velocity() {
        let o = IntegOrigin {
            position: Position::<RootInertial>::zero(),
            velocity: Velocity::<RootInertial>::from_raw_si(DVec3::new(0.0, 30_000.0, 0.0)),
        };
        let v_integ = Velocity::<IntegrationFrame>::from_raw_si(DVec3::new(0.0, 7500.0, 0.0));
        let v_root = o.shift_velocity(v_integ);
        assert_eq!(v_root.raw_si(), DVec3::new(0.0, 30_000.0 + 7500.0, 0.0));
    }
}
