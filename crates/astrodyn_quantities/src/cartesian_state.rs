//! [`CartesianState`] — a single (de)serializable position + velocity + epoch
//! record, the natural interchange unit for a replay log or telemetry frame.
//!
//! The typed surface keeps [`Position<F>`](crate::aliases::Position),
//! [`Velocity<F>`](crate::aliases::Velocity), and time as separate values so
//! the kernels can stay unit-pure. A downstream consumer that logs or streams
//! state, however, wants one record it can round-trip without inventing a
//! bespoke schema. `CartesianState<F>` is that record: position, velocity, and
//! the epoch they are sampled at, all sharing the static frame tag `F`.
//!
//! Serialization is gated behind the crate's non-default `serde` feature, so
//! the production dependency graph stays serde-free. The epoch is stored as a
//! typed [`SecondsSince<TDB>`] — *not* an external library's `Epoch` type — so
//! the record carries no dependency on the ephemeris/time backend and the
//! typed-quantity facade stays intact.

use crate::aliases::{Position, Velocity};
use crate::frame::Frame;
use crate::time_scale::{SecondsSince, TDB};

/// A Cartesian state sample: position, velocity, and the epoch they hold at,
/// all in frame `F`.
///
/// All three fields are public; construct directly or via [`Self::new`]. The
/// epoch is barycentric-dynamical seconds since the TDB epoch
/// ([`SecondsSince<TDB>`]).
///
/// With the crate's `serde` feature enabled this round-trips as
/// `{ "position": [x, y, z], "velocity": [x, y, z], "epoch": t }` in raw SI
/// units (metres, m/s, seconds). The frame `F` is encoded by the static type,
/// not the payload — deserializing into `CartesianState<Ecef>` vs.
/// `CartesianState<RootInertial>` is a compile-time choice at the call site.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
// Clear serde's auto-generated `F: Serialize`/`F: Deserialize` bound: `F` is a
// phantom carried inside the field types (which implement serde for *all*
// frames), so the struct needs no bound on `F` itself.
#[cfg_attr(feature = "serde", serde(bound = ""))]
pub struct CartesianState<F: Frame> {
    /// Position in frame `F` (metres).
    pub position: Position<F>,
    /// Velocity in frame `F` (m/s).
    pub velocity: Velocity<F>,
    /// Epoch the sample holds at — seconds since the TDB epoch.
    pub epoch: SecondsSince<TDB>,
}

impl<F: Frame> CartesianState<F> {
    /// Bundle a position, velocity, and epoch into a [`CartesianState`].
    #[inline]
    pub const fn new(
        position: Position<F>,
        velocity: Velocity<F>,
        epoch: SecondsSince<TDB>,
    ) -> Self {
        Self {
            position,
            velocity,
            epoch,
        }
    }

    /// Relabel the frame tag from `F` to `F2` without changing any numeric
    /// values — a caller-asserted phantom swap, mirroring
    /// [`Qty3::relabel_to`](crate::qty3::Qty3::relabel_to). The caller is
    /// responsible for the invariant that `F` and `F2` name the same physical
    /// frame.
    #[inline]
    pub fn relabel_to<F2: Frame>(self) -> CartesianState<F2> {
        CartesianState {
            position: self.position.relabel_to::<F2>(),
            velocity: self.velocity.relabel_to::<F2>(),
            epoch: self.epoch,
        }
    }
}

// Phantom-clean trait impls: `F` only appears inside `Position<F>`/`Velocity<F>`
// (which are `Copy`/`Clone`/`Debug`/`PartialEq` for every frame), so these are
// hand-written to avoid `#[derive]` minting a spurious `F: Copy` bound.
impl<F: Frame> Copy for CartesianState<F> {}

impl<F: Frame> Clone for CartesianState<F> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<F: Frame> core::fmt::Debug for CartesianState<F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CartesianState")
            .field("position", &self.position)
            .field("velocity", &self.velocity)
            .field("epoch_s", &self.epoch.as_seconds())
            .finish()
    }
}

impl<F: Frame> PartialEq for CartesianState<F> {
    fn eq(&self, other: &Self) -> bool {
        // Compare the epoch's underlying `uom` `Time` field directly:
        // `SecondsSince<S>` only derives `PartialEq` when `S: PartialEq`, which
        // the time-scale markers don't implement. `Time == Time` is a typed
        // comparison (not a bare-`f64` one), so it also stays clear of the
        // denied `clippy::float_cmp` lint.
        self.position == other.position
            && self.velocity == other.velocity
            && self.epoch.value == other.epoch.value
    }
}

#[cfg(all(test, feature = "serde"))]
mod serde_tests {
    use super::*;
    use crate::ext::Vec3Ext;
    use crate::frame::RootInertial;
    use glam::DVec3;

    #[test]
    fn cartesian_state_json_round_trips_bit_exact() {
        let state = CartesianState::<RootInertial>::new(
            DVec3::new(7_000_123.5, -42.25, 1_234_567.875).m_at::<RootInertial>(),
            DVec3::new(-0.5, 7_546.125, 3.0).m_per_s_at::<RootInertial>(),
            SecondsSince::<TDB>::from_seconds(86_400.5),
        );

        let json = serde_json::to_string(&state).expect("serialize");
        let back: CartesianState<RootInertial> = serde_json::from_str(&json).expect("deserialize");

        // Raw-SI values must survive the round trip bit-for-bit.
        let (p0, p1) = (state.position.raw_si(), back.position.raw_si());
        let (v0, v1) = (state.velocity.raw_si(), back.velocity.raw_si());
        assert_eq!(
            p0.to_array().map(f64::to_bits),
            p1.to_array().map(f64::to_bits)
        );
        assert_eq!(
            v0.to_array().map(f64::to_bits),
            v1.to_array().map(f64::to_bits)
        );
        assert_eq!(
            state.epoch.as_seconds().to_bits(),
            back.epoch.as_seconds().to_bits()
        );
    }

    #[test]
    fn cartesian_state_serializes_as_raw_si_triples() {
        let state = CartesianState::<RootInertial>::new(
            DVec3::new(1.0, 2.0, 3.0).m_at::<RootInertial>(),
            DVec3::new(4.0, 5.0, 6.0).m_per_s_at::<RootInertial>(),
            SecondsSince::<TDB>::from_seconds(10.0),
        );
        let value: serde_json::Value = serde_json::to_value(state).expect("to_value");
        assert_eq!(value["position"], serde_json::json!([1.0, 2.0, 3.0]));
        assert_eq!(value["velocity"], serde_json::json!([4.0, 5.0, 6.0]));
        assert_eq!(value["epoch"], serde_json::json!(10.0));
    }
}
