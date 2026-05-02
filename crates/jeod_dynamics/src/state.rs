use glam::DVec3;
use jeod_quantities::aliases::{Position, Velocity};
use jeod_quantities::frame::{Frame, IntegrationFrame, RootInertial};
use jeod_quantities::integ_origin::IntegOrigin;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TranslationalState {
    pub position: DVec3, // m, in integration frame
    pub velocity: DVec3, // m/s, in integration frame
}

impl TranslationalState {
    /// Returns true if both position and velocity are exactly zero.
    ///
    /// In JEOD, `check_for_uninitialized_states()` fatally fails if required
    /// state is not set. An exact-zero translational state is almost certainly
    /// unintentional for orbital mechanics (it would place the vehicle at the
    /// center of the integration frame with zero velocity).
    ///
    /// This is a heuristic — a vehicle intentionally at the origin with zero
    /// velocity would be a false positive.
    // JEOD_INV: DM.05 — partial: detects zero translational state (JEOD fatally checks all states)
    // JEOD_INV: DB.11 — partial: zero-state heuristic only (no initialized_states bitfield)
    pub fn is_likely_uninitialized(&self) -> bool {
        self.position == DVec3::ZERO && self.velocity == DVec3::ZERO
    }
}

/// Typed sibling of [`TranslationalState`] carrying a frame phantom on
/// the position and velocity components. Defaults to the inertial frame
/// to match the existing untyped storage convention; override with an
/// explicit frame tag for non-inertial integrations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TranslationalStateTyped<F: Frame = RootInertial> {
    /// Position in frame `F`.
    pub position: Position<F>,
    /// Velocity in frame `F`.
    pub velocity: Velocity<F>,
}

impl<F: Frame> Default for TranslationalStateTyped<F> {
    #[inline]
    fn default() -> Self {
        Self {
            position: Position::<F>::zero(),
            velocity: Velocity::<F>::zero(),
        }
    }
}

impl<F: Frame> TranslationalStateTyped<F> {
    /// Drop the frame phantom and emit the untyped storage form. The
    /// numeric values (in base SI units: m, m/s) are preserved exactly.
    #[inline]
    pub fn to_untyped(&self) -> TranslationalState {
        TranslationalState {
            position: self.position.raw_si(),
            velocity: self.velocity.raw_si(),
        }
    }

    /// Wrap an untyped [`TranslationalState`] as a typed one. **The
    /// caller asserts** that the untyped state is expressed in frame
    /// `F` — there is no runtime check.
    #[inline]
    pub fn from_untyped_unchecked(s: &TranslationalState) -> Self {
        Self {
            position: Position::<F>::from_raw_si(s.position),
            velocity: Velocity::<F>::from_raw_si(s.velocity),
        }
    }

    /// JEOD initialization heuristic — both position and velocity zero.
    ///
    /// Mirrors [`TranslationalState::is_likely_uninitialized`] for
    /// downstream code holding the typed form.
    // JEOD_INV: DM.05 — partial: detects zero translational state
    // JEOD_INV: DB.11 — partial: zero-state heuristic only
    #[inline]
    pub fn is_likely_uninitialized(&self) -> bool {
        self.position.raw_si() == DVec3::ZERO && self.velocity.raw_si() == DVec3::ZERO
    }
}

impl TranslationalStateTyped<IntegrationFrame> {
    /// Shift this integration-frame state to root-inertial by adding the
    /// integration-frame origin's offset (position and velocity, in
    /// root-inertial coordinates).
    ///
    /// This is the only safe path from `IntegrationFrame` to `RootInertial`
    /// for translational state. Forgetting the shift produces a compile
    /// error rather than silently-wrong physics for any vehicle whose
    /// integration frame is not the root frame (issue #255).
    // JEOD_INV: RF.10 — integration-frame state must be shifted to
    // root-inertial via the integration-origin offset before use by
    // root-inertial consumers (gravity, atmosphere, SRP, drag, orbital
    // elements, geodetic, LVLH, solar beta, earth lighting).
    #[inline]
    pub fn to_inertial(&self, o: &IntegOrigin) -> TranslationalStateTyped<RootInertial> {
        TranslationalStateTyped {
            position: o.shift_position(self.position),
            velocity: o.shift_velocity(self.velocity),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jeod_quantities::frame::Ecef;

    #[test]
    fn default_is_likely_uninitialized() {
        assert!(TranslationalState::default().is_likely_uninitialized());
    }

    #[test]
    fn nonzero_position_is_initialized() {
        let s = TranslationalState {
            position: DVec3::new(6.7e6, 0.0, 0.0),
            velocity: DVec3::ZERO,
        };
        assert!(!s.is_likely_uninitialized());
    }

    #[test]
    fn nonzero_velocity_is_initialized() {
        let s = TranslationalState {
            position: DVec3::ZERO,
            velocity: DVec3::new(0.0, 7.5e3, 0.0),
        };
        assert!(!s.is_likely_uninitialized());
    }

    #[test]
    fn typed_round_trips_through_untyped() {
        let untyped = TranslationalState {
            position: DVec3::new(7e6, 0.0, 0.0),
            velocity: DVec3::new(0.0, 7500.0, 0.0),
        };
        let typed = TranslationalStateTyped::<RootInertial>::from_untyped_unchecked(&untyped);
        let back = typed.to_untyped();
        assert_eq!(back, untyped);
    }

    #[test]
    fn typed_default_is_likely_uninitialized() {
        let s = TranslationalStateTyped::<RootInertial>::default();
        assert!(s.is_likely_uninitialized());
    }

    #[test]
    fn typed_inertial_and_ecef_are_distinct_types() {
        // Compile-time check that `TranslationalStateTyped<RootInertial>` and
        // `TranslationalStateTyped<Ecef>` are not assignable to one another;
        // we only verify the same-frame case compiles here.
        let s_inertial = TranslationalStateTyped::<RootInertial>::default();
        let s_ecef = TranslationalStateTyped::<Ecef>::default();
        assert!(s_inertial.is_likely_uninitialized());
        assert!(s_ecef.is_likely_uninitialized());
    }

    #[test]
    fn to_inertial_with_zero_origin_is_bit_identical() {
        let s_integ = TranslationalStateTyped::<IntegrationFrame>::from_untyped_unchecked(
            &TranslationalState {
                position: DVec3::new(7e6, 0.0, 0.0),
                velocity: DVec3::new(0.0, 7500.0, 0.0),
            },
        );
        let o = IntegOrigin::zero();
        let s_inertial = s_integ.to_inertial(&o);
        assert_eq!(s_inertial.position.raw_si(), DVec3::new(7e6, 0.0, 0.0));
        assert_eq!(s_inertial.velocity.raw_si(), DVec3::new(0.0, 7500.0, 0.0));
    }

    #[test]
    fn to_inertial_with_nonzero_origin_adds_offset() {
        let s_integ = TranslationalStateTyped::<IntegrationFrame>::from_untyped_unchecked(
            &TranslationalState {
                position: DVec3::new(7e6, 0.0, 0.0),
                velocity: DVec3::new(0.0, 7500.0, 0.0),
            },
        );
        let o = IntegOrigin {
            position: Position::<RootInertial>::from_raw_si(DVec3::new(1.5e11, 0.0, 0.0)),
            velocity: Velocity::<RootInertial>::from_raw_si(DVec3::new(0.0, 30_000.0, 0.0)),
        };
        let s_inertial = s_integ.to_inertial(&o);
        assert_eq!(
            s_inertial.position.raw_si(),
            DVec3::new(1.5e11 + 7e6, 0.0, 0.0)
        );
        assert_eq!(
            s_inertial.velocity.raw_si(),
            DVec3::new(0.0, 30_000.0 + 7500.0, 0.0)
        );
    }
}
