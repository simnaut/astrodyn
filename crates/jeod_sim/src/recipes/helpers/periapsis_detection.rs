//! Periapsis-passage detection via radial-velocity zero crossing.
//!
//! Used by Mercury / GR perihelion-advance tests where the bespoke
//! per-step loop captures an event each time `dr/dt` flips from
//! negative to positive (the body has passed periapsis since the
//! previous step).

use glam::DVec3;
use jeod_quantities::aliases::{Position, Velocity};
use jeod_quantities::dims::GravParam;
use jeod_quantities::frame::{Planet, PlanetInertial};

/// A periapsis-passage event: the simulation time at the crossing
/// (linear interpolation against the radial-velocity sign change is
/// **not** performed here — callers that need sub-step precision
/// should refit), and the longitude of perihelion
/// `arg_periapsis + long_asc_node` (radians).
#[derive(Debug, Clone, Copy)]
pub struct PeriapsisEvent {
    /// Time of detection (s, simulation epoch-relative).
    pub time: f64,
    /// Longitude of perihelion (rad), invariant to nodal regression.
    pub long_perihelion: f64,
}

/// Streaming periapsis detector.
///
/// Feed each `(time, position, velocity)` sample; emits at most one
/// event per call. Returns `Some(event)` when the radial velocity
/// `r·v / |r|` transitions from negative to non-negative between the
/// previous and current samples.
#[derive(Debug, Clone, Copy)]
pub struct PeriapsisDetector {
    prev_rdot: f64,
    initialized: bool,
}

impl Default for PeriapsisDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl PeriapsisDetector {
    /// Construct a fresh detector. The first sample seeds the previous
    /// radial-rate value; periapsis crossings are reported from the
    /// second sample onwards.
    pub fn new() -> Self {
        Self {
            prev_rdot: 0.0,
            initialized: false,
        }
    }

    /// Reset the detector so the next call seeds the previous radial
    /// velocity without emitting an event.
    pub fn reset(&mut self) {
        self.initialized = false;
        self.prev_rdot = 0.0;
    }

    /// Returns `true` if the previous-to-current sample contains a
    /// periapsis crossing (radial velocity transitions from `< 0` to
    /// `≥ 0`).
    pub fn observe(&mut self, pos: DVec3, vel: DVec3) -> bool {
        let r_dot = pos.dot(vel) / pos.length();
        let crossed = self.initialized && self.prev_rdot < 0.0 && r_dot >= 0.0;
        self.prev_rdot = r_dot;
        self.initialized = true;
        crossed
    }
}

/// Sweep an iterator of `(time, position, velocity)` samples taken in
/// `PlanetInertial<P>` and return all detected periapsis events. The
/// orbital elements are computed at the post-crossing sample via
/// `OrbitalElements::<P>::from_cartesian_typed`, giving longitude of
/// perihelion `arg_periapsis + long_asc_node`.
///
/// `P: Planet` ties the gravitational parameter to the inertial frame
/// of the position/velocity samples — the compiler refuses a `mu_sun`
/// paired with `Position<PlanetInertial<Earth>>` (or vice versa), so
/// the planet-tagging guarantee from RF.11 stays structural through
/// this helper. Callers in `tier3_sim_mercury` pass `<Sun>`; callers
/// for an Earth-orbit sweep would pass `<Earth>`.
///
/// Used by `tier3_sim_mercury` for both the in-memory sim trace and
/// the JEOD CSV trace (they share this loop body).
///
/// # Compile-fail: wrong-planet μ
///
/// Pairing the Sun's μ with samples tagged `<Earth>` is rejected at
/// compile time:
///
/// ```compile_fail
/// use jeod_sim::recipes::helpers::detect_periapsis_passages;
/// use jeod_quantities::aliases::{Position, Velocity};
/// use jeod_quantities::dims::GravParam;
/// use jeod_quantities::frame::{Earth, PlanetInertial, Sun};
///
/// let mu_sun = GravParam::<Sun>::from_si(1.327e20);
/// let samples = std::iter::empty::<(
///     f64,
///     Position<PlanetInertial<Earth>>,
///     Velocity<PlanetInertial<Earth>>,
/// )>();
/// // mu is `<Sun>` but samples are `<Earth>` — refuses to compile.
/// let _ = detect_periapsis_passages(samples, mu_sun);
/// ```
pub fn detect_periapsis_passages<P, I>(samples: I, mu: GravParam<P>) -> Vec<PeriapsisEvent>
where
    P: Planet,
    I: IntoIterator<
        Item = (
            f64,
            Position<PlanetInertial<P>>,
            Velocity<PlanetInertial<P>>,
        ),
    >,
{
    use jeod_math::OrbitalElements;

    let mut det = PeriapsisDetector::new();
    let mut events = Vec::new();
    for (t, r, v) in samples {
        let r_raw = r.raw_si();
        let v_raw = v.raw_si();
        if det.observe(r_raw, v_raw) {
            let oe = OrbitalElements::<P>::from_cartesian_typed(mu, r, v).unwrap_or_else(|e| {
                panic!(
                    "periapsis_detection: from_cartesian_typed failed at t={t}, \
                     position={r_raw:?}, velocity={v_raw:?}: {e:?}"
                )
            });
            events.push(PeriapsisEvent {
                time: t,
                long_perihelion: oe.arg_periapsis + oe.long_asc_node,
            });
        }
    }
    events
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detector_emits_on_negative_to_positive_crossing() {
        let mut det = PeriapsisDetector::new();
        // Sample 1: r·v < 0 (approaching periapsis).
        let crossed_1 = det.observe(DVec3::X, -DVec3::X);
        assert!(!crossed_1, "first sample seeds prev_rdot, no event");
        // Sample 2: r·v > 0 (post-periapsis).
        let crossed_2 = det.observe(DVec3::X, DVec3::X);
        assert!(crossed_2);
    }

    #[test]
    fn detector_quiet_when_monotonic() {
        let mut det = PeriapsisDetector::new();
        det.observe(DVec3::X, DVec3::X); // seed
        let crossed = det.observe(DVec3::X, DVec3::X);
        assert!(!crossed);
    }
}
