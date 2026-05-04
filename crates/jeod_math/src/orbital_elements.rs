//! Classical orbital elements and Cartesian ↔ Keplerian conversion.
//!
//! Ports
//! [`models/utils/orbital_elements/src/orbital_elements.cc`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/utils/orbital_elements/src/orbital_elements.cc)
//! from JEOD v5.4.0. Holds semi-major axis, eccentricity, inclination,
//! RAAN, argument of periapsis, the three anomalies (true, mean,
//! orbital), and energy / angular-momentum diagnostics.
//!
//! Public callers should reach for the typed conversions on
//! [`OrbitalElements`]; the bare-`f64` siblings are kept
//! module-private after the Phase 10 typed-API purge.

use core::marker::PhantomData;
use std::f64::consts::{PI, TAU};

use jeod_quantities::dims::GravParam;
use jeod_quantities::frame::{Planet, PlanetInertial, SelfPlanet};

use crate::error::OrbitalError;
use crate::types::{mat3_from_rows, DVec3};

/// Classical orbital elements computed from Cartesian state vectors.
///
/// Ported from JEOD `models/utils/orbital_elements/src/orbital_elements.cc`.
///
/// The phantom `P: Planet` ties the result to the central body whose μ
/// produced these elements: `OrbitalElements<Earth>` and
/// `OrbitalElements<Sun>` are distinct types and cannot be combined.
/// Defaults to [`SelfPlanet`] when a planet phantom is not specified —
/// the planet-erased variant matches the previous bare `OrbitalElements`
/// for callers that don't need static planet pinning (e.g.
/// [`OrbitalElements::default`], the Bevy adapter components, the
/// `to_cartesian` rebuild path).
#[derive(Debug, Clone)]
pub struct OrbitalElements<P: Planet = SelfPlanet> {
    /// Semi-major axis (negative for hyperbolic orbits).
    pub semi_major_axis: f64,
    /// Semi-latus rectum p = a(1 - e^2).
    pub semiparam: f64,
    /// Eccentricity.
    pub e_mag: f64,
    /// Inclination (rad).
    pub inclination: f64,
    /// Argument of periapsis (rad).
    pub arg_periapsis: f64,
    /// Longitude of ascending node (rad).
    pub long_asc_node: f64,
    /// True anomaly (rad).
    pub true_anom: f64,
    /// Mean anomaly (rad).
    pub mean_anom: f64,
    /// Orbital (eccentric / hyperbolic / parabolic) anomaly (rad).
    pub orbital_anom: f64,
    /// Mean motion (rad/s).
    pub mean_motion: f64,
    /// Position magnitude.
    pub r_mag: f64,
    /// Velocity magnitude.
    pub vel_mag: f64,
    /// Specific orbital energy.
    pub orb_energy: f64,
    /// Specific orbital angular momentum magnitude.
    pub orb_ang_momentum: f64,

    /// Cached sin(true_anomaly).
    sin_v: f64,
    /// Cached cos(true_anomaly).
    cos_v: f64,

    _p: PhantomData<P>,
}

impl<P: Planet> Default for OrbitalElements<P> {
    fn default() -> Self {
        Self {
            semi_major_axis: 0.0,
            semiparam: 0.0,
            e_mag: 0.0,
            inclination: 0.0,
            arg_periapsis: 0.0,
            long_asc_node: 0.0,
            true_anom: 0.0,
            mean_anom: 0.0,
            orbital_anom: 0.0,
            mean_motion: 0.0,
            r_mag: 0.0,
            vel_mag: 0.0,
            orb_energy: 0.0,
            orb_ang_momentum: 0.0,
            sin_v: 0.0,
            cos_v: 0.0,
            _p: PhantomData,
        }
    }
}

// Tolerance thresholds (matching JEOD source: orbital_elements.cc:138-139)
const TOLERANCE: f64 = 1e-13;
const ORBIT_SWITCH_TOL: f64 = 1e-2;

/// Normalize an angle into [0, 2pi).
fn wrap_to_tau(mut angle: f64) -> f64 {
    angle %= TAU;
    if angle < 0.0 {
        angle += TAU;
    }
    angle
}

impl<P: Planet> OrbitalElements<P> {
    /// Relabel `self` as orbital elements computed against planet `Q`.
    ///
    /// **Boundary-only escape hatch.** Use only at relabel sites where
    /// the planet identity is determined at runtime (e.g. wrapping a
    /// raw `from_cartesian_impl` result, the Bevy `OrbitalElementsC`
    /// component which is parameterized by `SelfPlanet`). Mission code
    /// that knows the central body at compile time should reach for
    /// [`OrbitalElements::from_cartesian_typed`] directly.
    #[inline]
    pub fn relabel<Q: Planet>(self) -> OrbitalElements<Q> {
        OrbitalElements::<Q> {
            semi_major_axis: self.semi_major_axis,
            semiparam: self.semiparam,
            e_mag: self.e_mag,
            inclination: self.inclination,
            arg_periapsis: self.arg_periapsis,
            long_asc_node: self.long_asc_node,
            true_anom: self.true_anom,
            mean_anom: self.mean_anom,
            orbital_anom: self.orbital_anom,
            mean_motion: self.mean_motion,
            r_mag: self.r_mag,
            vel_mag: self.vel_mag,
            orb_energy: self.orb_energy,
            orb_ang_momentum: self.orb_ang_momentum,
            sin_v: self.sin_v,
            cos_v: self.cos_v,
            _p: PhantomData,
        }
    }
}

impl OrbitalElements<SelfPlanet> {
    // ----------------------------------------------------------------
    // Cartesian -> orbital elements
    // ----------------------------------------------------------------

    /// Compute classical orbital elements from Cartesian position and velocity
    /// in an inertial frame.
    ///
    /// Internal numeric kernel shared by [`OrbitalElements::from_cartesian_typed`].
    /// Kept module-private after the Phase 10 purge of the bare-`f64`
    /// public surface — new callers should use the typed sibling.
    ///
    /// # Arguments
    /// * `mu`  - gravitational parameter (units consistent with pos/vel, e.g. m^3/s^2)
    /// * `pos` - position vector
    /// * `vel` - velocity vector
    pub(crate) fn from_cartesian_impl(
        mu: f64,
        pos: DVec3,
        vel: DVec3,
    ) -> Result<OrbitalElements<SelfPlanet>, OrbitalError> {
        // JEOD_INV: OE.01 — mu must be finite and positive; NaN/±Inf would
        // propagate through the element computations and produce silent NaNs.
        if !mu.is_finite() || mu <= 0.0 {
            return Err(OrbitalError::InvalidMu(mu));
        }

        let r_mag = pos.length();
        let vel_mag = vel.length();

        // JEOD_INV: OE.07 — both position and velocity must be non-zero for the
        // orbit to be defined; either being zero degenerates `h = r × v`.
        if r_mag < 1e-30 || vel_mag < 1e-30 {
            return Err(OrbitalError::DegenerateOrbit);
        }

        // Angular momentum
        let ang_momntm = pos.cross(vel);
        let h_mag = ang_momntm.length();

        if h_mag < 1e-30 {
            return Err(OrbitalError::DegenerateOrbit);
        }

        // Eccentricity vector: e = ((v^2 - mu/r)*r - (r.v)*v) / mu
        let v2 = vel_mag * vel_mag;
        let pos_dot_vel = pos.dot(vel);
        let e_vec = ((v2 - mu / r_mag) * pos - pos_dot_vel * vel) / mu;
        let ecc = e_vec.length();

        // Specific energy
        let energy = v2 / 2.0 - mu / r_mag;

        // ---- Orbit type branching ----
        let (a, p, n);

        if ecc < TOLERANCE {
            // Circular (or near-circular) — JEOD uses a = r_mag, not energy formula
            a = r_mag;
            p = r_mag;
            n = (mu / a).sqrt() / a;
        } else if ecc < (1.0 - ORBIT_SWITCH_TOL) {
            // Elliptic
            a = -mu / (2.0 * energy);
            p = a * (1.0 - ecc * ecc);
            n = (mu / a).sqrt() / a;
        } else if ecc > (1.0 + ORBIT_SWITCH_TOL) {
            // Hyperbolic (a is negative)
            a = -mu / (2.0 * energy);
            p = a * (1.0 - ecc * ecc);
            n = (mu / -a).sqrt() / -a;
        } else {
            // Parabolic — JEOD nulls out semi_major_axis (it is infinity)
            a = 0.0;
            p = h_mag * h_mag / mu;
            n = 2.0 * (mu / p).sqrt() / p;
        }

        // ---- Inclination ----
        let k_hat = DVec3::Z; // inertial Z
        let k_cross_h = k_hat.cross(ang_momntm);
        let k_cross_h_mag = k_cross_h.length();
        let k_dot_h = k_hat.dot(ang_momntm);
        let incl = k_cross_h_mag.atan2(k_dot_h); // always in [0, pi]

        // ---- Node vector ----
        let line_of_nodes = k_cross_h; // points toward ascending node

        // JEOD_INV: OE.04 — circular regime uses e < TOLERANCE branch
        // JEOD_INV: OE.05 — equatorial regime uses i < TOL || i > π-TOL branch
        // JEOD: (inclination < tolerance) || ((M_PI - tolerance) < inclination)
        #[allow(clippy::manual_range_contains)]
        let is_equatorial = incl < TOLERANCE || (PI - TOLERANCE) < incl;
        let is_circular = ecc < TOLERANCE;

        // JEOD SWP 2005: all angles computed via atan2(|A × B|, A · B) ∈ [0, π],
        // then flipped to [π, 2π] based on a component sign. This avoids acos
        // robustness issues and matches JEOD's exact floating-point behavior.
        let i_hat = DVec3::X; // inertial I

        let (lan, mut aop, mut nu);

        if is_equatorial && is_circular {
            // ---- Case 1: equatorial + circular ----
            // SWP 2005: true longitude from I × pos
            lan = 0.0;
            aop = 0.0;
            let cross_vec = i_hat.cross(pos);
            let sin_in = cross_vec.length();
            let cos_in = i_hat.dot(pos);
            nu = sin_in.atan2(cos_in);
            // Quadrant adjustment based on orbit direction (prograde vs retrograde)
            if incl < TOLERANCE {
                if pos.y < 0.0 {
                    nu = TAU - nu;
                }
            } else if pos.y > 0.0 {
                nu = TAU - nu;
            }
        } else if is_equatorial {
            // ---- Case 2: equatorial + non-circular ----
            // SWP 2005: arg_periapsis from I × e
            lan = 0.0;
            let cross_vec = i_hat.cross(e_vec);
            let sin_in = cross_vec.length();
            let cos_in = i_hat.dot(e_vec);
            aop = sin_in.atan2(cos_in);
            if incl < TOLERANCE {
                if e_vec.y < 0.0 {
                    aop = TAU - aop;
                }
            } else if e_vec.y > 0.0 {
                aop = TAU - aop;
            }

            // SWP 2005: true anomaly from e × pos
            let cross_vec2 = e_vec.cross(pos);
            let sin_in2 = cross_vec2.length();
            let cos_in2 = e_vec.dot(pos);
            nu = sin_in2.atan2(cos_in2);
            if pos_dot_vel < 0.0 {
                nu = TAU - nu;
            }
        } else if is_circular {
            // ---- Case 3: non-equatorial + circular ----
            // SWP 2005: LAN from I × N
            let cross_vec = i_hat.cross(line_of_nodes);
            let sin_in = cross_vec.length();
            let cos_in = i_hat.dot(line_of_nodes);
            lan = {
                let mut v = sin_in.atan2(cos_in);
                if line_of_nodes.y < 0.0 {
                    v = TAU - v;
                }
                v
            };
            aop = 0.0;

            // SWP 2005: true anomaly (argument of latitude) from N × pos
            let cross_vec2 = line_of_nodes.cross(pos);
            let sin_in2 = cross_vec2.length();
            let cos_in2 = line_of_nodes.dot(pos);
            nu = sin_in2.atan2(cos_in2);
            if pos.z < 0.0 {
                nu = TAU - nu;
            }
        } else {
            // ---- Case 4: general (non-equatorial, non-circular) ----
            // SWP 2005: LAN from I × N
            let cross_vec = i_hat.cross(line_of_nodes);
            let sin_in = cross_vec.length();
            let cos_in = i_hat.dot(line_of_nodes);
            lan = {
                let mut v = sin_in.atan2(cos_in);
                if line_of_nodes.y < 0.0 {
                    v = TAU - v;
                }
                v
            };

            // SWP 2005: arg_periapsis from N × e
            let cross_vec2 = line_of_nodes.cross(e_vec);
            let sin_in2 = cross_vec2.length();
            let cos_in2 = line_of_nodes.dot(e_vec);
            aop = {
                let mut v = sin_in2.atan2(cos_in2);
                if e_vec.z < 0.0 {
                    v = TAU - v;
                }
                v
            };

            // SWP 2005: true anomaly from e × pos
            let cross_vec3 = e_vec.cross(pos);
            let sin_in3 = cross_vec3.length();
            let cos_in3 = e_vec.dot(pos);
            nu = sin_in3.atan2(cos_in3);
            if pos_dot_vel < 0.0 {
                nu = TAU - nu;
            }
        }

        let sin_v = nu.sin();
        let cos_v = nu.cos();

        let mut oe = OrbitalElements::<SelfPlanet> {
            semi_major_axis: a,
            semiparam: p,
            e_mag: ecc,
            inclination: incl,
            arg_periapsis: aop,
            long_asc_node: lan,
            true_anom: nu,
            mean_anom: 0.0,
            orbital_anom: 0.0,
            mean_motion: n,
            r_mag,
            vel_mag,
            orb_energy: energy,
            orb_ang_momentum: h_mag,
            sin_v,
            cos_v,
            _p: PhantomData,
        };

        oe.nu_to_anomalies();

        Ok(oe)
    }
}

impl<P: Planet> OrbitalElements<P> {
    /// Typed variant of `from_cartesian` (the file-private kernel below).
    ///
    /// Accepts dimensionally-typed inputs:
    /// * `mu` — gravitational parameter [`GravParam<P>`], pinned to the
    ///   same planet phantom `P` as the position/velocity frames
    /// * `pos` — `Position<PlanetInertial<P>>` in meters
    /// * `vel` — `Velocity<PlanetInertial<P>>` in m/s
    ///
    /// The shared `<P>` makes the mu-vs-frame agreement structural: a
    /// caller cannot pass `mu_sun()` (which is `GravParam<Sun>`) into
    /// `from_cartesian_typed::<Earth>(...)` — the compiler refuses.
    ///
    /// The returned [`OrbitalElements<P>`] carries the planet identity
    /// through to downstream consumers so a Mars-centered set cannot
    /// silently flow into an Earth-orbit code path.
    ///
    /// # Positive control: matching planet phantoms compile
    ///
    /// ```
    /// use glam::DVec3;
    /// use jeod_math::OrbitalElements;
    /// use jeod_quantities::prelude::*;
    ///
    /// let mu = 3.986_004_415e14_f64.m3_per_s2_for::<Earth>();
    /// let pos: Position<PlanetInertial<Earth>> = Qty3::from_raw_si(DVec3::new(7e6, 0.0, 0.0));
    /// let vel: Velocity<PlanetInertial<Earth>> = Qty3::from_raw_si(DVec3::new(0.0, 7546.0, 0.0));
    /// let _oe = OrbitalElements::<Earth>::from_cartesian_typed(mu, pos, vel).unwrap();
    /// ```
    ///
    /// # Compile-fail: μ for wrong planet rejected
    ///
    /// Pairing `mu_sun()` (`GravParam<Sun>`) with Earth-tagged
    /// position/velocity is the load-bearing bug shape that motivated
    /// this typing — the compiler rejects it.
    ///
    /// ```compile_fail
    /// use glam::DVec3;
    /// use jeod_math::OrbitalElements;
    /// use jeod_quantities::prelude::*;
    ///
    /// let mu_sun = 1.327_124_400_18e20_f64.m3_per_s2_for::<Sun>();
    /// let pos: Position<PlanetInertial<Earth>> = Qty3::from_raw_si(DVec3::new(7e6, 0.0, 0.0));
    /// let vel: Velocity<PlanetInertial<Earth>> = Qty3::from_raw_si(DVec3::new(0.0, 7546.0, 0.0));
    /// // mu (Sun) vs pos/vel (Earth) — compile error.
    /// let _bad = OrbitalElements::<Earth>::from_cartesian_typed(mu_sun, pos, vel);
    /// ```
    ///
    /// # Compile-fail: position/velocity from a different planet rejected
    ///
    /// ```compile_fail
    /// use glam::DVec3;
    /// use jeod_math::OrbitalElements;
    /// use jeod_quantities::prelude::*;
    ///
    /// let mu_earth = 3.986_004_415e14_f64.m3_per_s2_for::<Earth>();
    /// let pos: Position<PlanetInertial<Mars>> = Qty3::from_raw_si(DVec3::new(7e6, 0.0, 0.0));
    /// let vel: Velocity<PlanetInertial<Mars>> = Qty3::from_raw_si(DVec3::new(0.0, 3000.0, 0.0));
    /// // mu (Earth) vs pos/vel (Mars) — compile error.
    /// let _bad = OrbitalElements::<Earth>::from_cartesian_typed(mu_earth, pos, vel);
    /// ```
    ///
    /// # Compile-fail: result cannot flow into a different-planet slot
    ///
    /// The returned `OrbitalElements<Earth>` cannot silently be
    /// assigned to a `OrbitalElements<Sun>` slot.
    ///
    /// ```compile_fail
    /// use glam::DVec3;
    /// use jeod_math::OrbitalElements;
    /// use jeod_quantities::prelude::*;
    ///
    /// let mu = 3.986_004_415e14_f64.m3_per_s2_for::<Earth>();
    /// let pos: Position<PlanetInertial<Earth>> = Qty3::from_raw_si(DVec3::new(7e6, 0.0, 0.0));
    /// let vel: Velocity<PlanetInertial<Earth>> = Qty3::from_raw_si(DVec3::new(0.0, 7546.0, 0.0));
    /// let earth_oe = OrbitalElements::<Earth>::from_cartesian_typed(mu, pos, vel).unwrap();
    /// // Can't assign Earth-tagged elements to a Sun-tagged slot:
    /// let _bad: OrbitalElements<Sun> = earth_oe;
    /// ```
    pub fn from_cartesian_typed(
        mu: GravParam<P>,
        pos: jeod_quantities::aliases::Position<PlanetInertial<P>>,
        vel: jeod_quantities::aliases::Velocity<PlanetInertial<P>>,
    ) -> Result<OrbitalElements<P>, OrbitalError> {
        // JEOD_INV: RF.11 — `mu`, `pos`, `vel`, and the returned
        // `OrbitalElements<P>` share the planet phantom `P`, so a μ for
        // the wrong central body is rejected at compile time.
        // Extract SI base values and delegate to the shared planet-erased
        // kernel; relabel the result with the call-site's planet phantom.
        OrbitalElements::<SelfPlanet>::from_cartesian_impl(mu.value, pos.raw_si(), vel.raw_si())
            .map(OrbitalElements::<SelfPlanet>::relabel::<P>)
    }

    // ----------------------------------------------------------------
    // Orbital elements -> Cartesian
    // ----------------------------------------------------------------

    /// Reconstruct Cartesian position and velocity from classical orbital
    /// elements and a gravitational parameter.
    pub fn to_cartesian(&self, mu: f64) -> Result<(DVec3, DVec3), OrbitalError> {
        if mu <= 0.0 {
            return Err(OrbitalError::InvalidMu(mu));
        }

        let p = self.semiparam;
        // JEOD_INV: OE.02 — semi-parameter p must be positive for to_cartesian
        if p <= 0.0 || !p.is_finite() {
            return Err(OrbitalError::DegenerateOrbit);
        }
        let e = self.e_mag;
        let nu = self.true_anom;

        let sin_nu = nu.sin();
        let cos_nu = nu.cos();

        // JEOD_INV: OE.03 — sin²ν + cos²ν must be within 1e-6 of 1
        // JEOD orbital_elements.cc:414-424: verify sin/cos consistency.
        let rss = (sin_nu * sin_nu + cos_nu * cos_nu).sqrt();
        assert!(
            (rss - 1.0).abs() < 1e-6,
            "sin/cos of true anomaly are inconsistent: sin_v={sin_nu}, cos_v={cos_nu}, rss={rss}"
        );

        let denom = 1.0 + e * cos_nu;
        if denom.abs() < 1e-30 {
            return Err(OrbitalError::DegenerateOrbit);
        }
        let r = p / denom;

        // Position and velocity in perifocal (PQW) frame
        let r_pqw = DVec3::new(r * cos_nu, r * sin_nu, 0.0);

        let coeff = (mu / p).sqrt();
        let v_pqw = DVec3::new(-coeff * sin_nu, coeff * (e + cos_nu), 0.0);

        // Build rotation matrix PQW -> inertial from (Omega, omega, i)
        let co = self.long_asc_node.cos();
        let so = self.long_asc_node.sin();
        let cw = self.arg_periapsis.cos();
        let sw = self.arg_periapsis.sin();
        let ci = self.inclination.cos();
        let si = self.inclination.sin();

        // Rotation matrix rows (PQW -> IJK)
        // This is R3(-Omega) * R1(-i) * R3(-omega) built row-wise
        let row0 = DVec3::new(co * cw - so * sw * ci, -co * sw - so * cw * ci, so * si);
        let row1 = DVec3::new(so * cw + co * sw * ci, -so * sw + co * cw * ci, -co * si);
        let row2 = DVec3::new(sw * si, cw * si, ci);

        // mat3_from_rows builds a glam DMat3 such that (M * v)[i] = row_i . v
        // which is exactly the PQW -> IJK rotation applied to a PQW vector.
        let rot = mat3_from_rows(row0, row1, row2);

        let pos = rot * r_pqw;
        let vel = rot * v_pqw;

        Ok((pos, vel))
    }

    // ----------------------------------------------------------------
    // True anomaly -> eccentric/mean anomaly
    // ----------------------------------------------------------------

    /// Convert true anomaly to eccentric (or hyperbolic/parabolic) anomaly
    /// and mean anomaly, storing results in `self`.
    pub fn nu_to_anomalies(&mut self) {
        let e = self.e_mag;
        let nu = self.true_anom;
        let sin_nu = nu.sin();
        let cos_nu = nu.cos();

        if e < (1.0 - ORBIT_SWITCH_TOL) {
            // Elliptic (includes circular)
            // Eccentric anomaly E:  tan(E/2) = sqrt((1-e)/(1+e)) * tan(nu/2)
            let sin_ea = ((1.0 - e * e).sqrt() * sin_nu) / (1.0 + e * cos_nu);
            let cos_ea = (e + cos_nu) / (1.0 + e * cos_nu);
            let ea = wrap_to_tau(sin_ea.atan2(cos_ea));

            // Mean anomaly:  M = E - e*sin(E)
            let ma = wrap_to_tau(ea - e * ea.sin());

            self.orbital_anom = ea;
            self.mean_anom = ma;
        } else if e > (1.0 + ORBIT_SWITCH_TOL) {
            // Hyperbolic
            // Hyperbolic anomaly H:  tanh(H/2) = sqrt((e-1)/(e+1)) * tan(nu/2)
            let sinh_ha = ((e * e - 1.0).sqrt() * sin_nu) / (1.0 + e * cos_nu);
            let cosh_ha = (e + cos_nu) / (1.0 + e * cos_nu);
            // H = ln(cosh(H) + sinh(H)) — more robust than atanh for large H
            let ha = (cosh_ha + sinh_ha).ln();

            // Mean anomaly:  M = e*sinh(H) - H
            let ma = e * ha.sinh() - ha;

            self.orbital_anom = ha;
            self.mean_anom = ma;
        } else {
            // Parabolic / near-parabolic
            // Parabolic anomaly D = tan(nu/2)
            let d = (nu / 2.0).tan();
            // Barker's equation:  M = D + D^3/3
            let ma = d + d * d * d / 3.0;

            self.orbital_anom = d;
            self.mean_anom = ma;
        }
    }

    // ----------------------------------------------------------------
    // Mean anomaly -> true anomaly
    // ----------------------------------------------------------------

    /// Convert mean anomaly to true anomaly, updating `self.true_anom`,
    /// `self.orbital_anom`, `self.sin_v`, and `self.cos_v`.
    pub fn mean_anom_to_nu(&mut self) -> Result<(), OrbitalError> {
        let e = self.e_mag;
        let m = self.mean_anom;

        if e < (1.0 - ORBIT_SWITCH_TOL) {
            // Elliptic
            let ea = kep_eqtn_e(m, e)?;
            self.orbital_anom = ea;

            // E -> nu
            let sin_ea = ea.sin();
            let cos_ea = ea.cos();
            let sin_nu = ((1.0 - e * e).sqrt() * sin_ea) / (1.0 - e * cos_ea);
            let cos_nu = (cos_ea - e) / (1.0 - e * cos_ea);
            let nu = wrap_to_tau(sin_nu.atan2(cos_nu));

            self.true_anom = nu;
            self.sin_v = nu.sin();
            self.cos_v = nu.cos();
        } else if e > (1.0 + ORBIT_SWITCH_TOL) {
            // Hyperbolic
            let ha = kep_eqtn_h(m, e)?;
            self.orbital_anom = ha;

            // H -> nu
            let sinh_ha = ha.sinh();
            let cosh_ha = ha.cosh();
            let sin_nu = ((e * e - 1.0).sqrt() * sinh_ha) / (e * cosh_ha - 1.0);
            let cos_nu = (e - cosh_ha) / (e * cosh_ha - 1.0);
            let nu = wrap_to_tau(sin_nu.atan2(cos_nu));

            self.true_anom = nu;
            self.sin_v = nu.sin();
            self.cos_v = nu.cos();
        } else {
            // Parabolic
            let d = kep_eqtn_b(m);
            self.orbital_anom = d;

            let nu = 2.0 * d.atan();
            let nu = wrap_to_tau(nu);

            self.true_anom = nu;
            self.sin_v = nu.sin();
            self.cos_v = nu.cos();
        }

        Ok(())
    }
}

// ====================================================================
// Kepler solvers
// ====================================================================

/// Solve Kepler's equation for elliptic orbits:  M = E - e sin(E).
///
/// Newton-Raphson iteration with tolerance 1e-14 and maximum 1000 iterations.
/// Port of JEOD `orbital_elements.cc` `kep_eqtn_e()`.
pub fn kep_eqtn_e(m: f64, e: f64) -> Result<f64, OrbitalError> {
    const TOL: f64 = 1e-14;
    const MAX_ITER: usize = 1000;

    // Initial guess (JEOD heuristic)
    let mut ea = if e < 0.8 { m } else { PI };

    for _ in 0..MAX_ITER {
        let f = ea - e * ea.sin() - m;
        let fp = 1.0 - e * ea.cos();
        let delta = f / fp;
        ea -= delta;
        if delta.abs() < TOL {
            return Ok(wrap_to_tau(ea));
        }
    }

    // JEOD_INV: OE.06 — Kepler equation must converge; return error if it does not
    Err(OrbitalError::KeplerConvergence(MAX_ITER))
}

/// Solve Kepler's equation for hyperbolic orbits:  M = e sinh(H) - H.
///
/// Newton-Raphson iteration with tolerance 1e-14 and maximum 1000 iterations.
/// Port of JEOD `orbital_elements.cc` `kep_eqtn_h()`, including the 4-case
/// initial guess heuristic for robust convergence at extreme eccentricities.
pub fn kep_eqtn_h(m: f64, e: f64) -> Result<f64, OrbitalError> {
    const TOL: f64 = 1e-14;
    const MAX_ITER: usize = 1000;

    // JEOD 4-case initial guess heuristic (orbital_elements.cc)
    let mut ha = if e < 1.6 {
        if m > 0.0 {
            m + e
        } else {
            m - e
        }
    } else if e < 3.6 && m.abs() > PI {
        m - m.signum() * e
    } else {
        m / (e - 1.0)
    };

    for _ in 0..MAX_ITER {
        let f = e * ha.sinh() - ha - m;
        let fp = e * ha.cosh() - 1.0;
        let delta = f / fp;
        ha -= delta;
        if delta.abs() < TOL {
            return Ok(ha);
        }
    }

    // JEOD_INV: OE.06 — Kepler equation must converge; return error if it does not
    Err(OrbitalError::KeplerConvergence(MAX_ITER))
}

/// Solve Kepler's equation for parabolic orbits:  M = D + D^3/3.
///
/// Closed-form via the cubic root (Barker's equation).
pub fn kep_eqtn_b(m: f64) -> f64 {
    // Barker's equation: M = D + D^3/3
    // Re-arrange: D^3 + 3*D - 3*M = 0
    // Using the real root of the depressed cubic x^3 + px + q = 0
    // with p=3, q=-3M:
    //   discriminant = (q/2)^2 + (p/3)^3 = (9M^2/4) + 1
    let disc = (9.0 * m * m / 4.0 + 1.0).sqrt();
    (1.5 * m + disc).cbrt() - (-1.5 * m + disc).cbrt()
}

// ====================================================================
// Tests
// ====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DVec3;

    /// Compact test-only wrapper preserving the previous f64 call shape.
    /// Delegates straight to the shared kernel — the bare-`f64` public
    /// `from_cartesian` was removed in Phase 10; tests still need to
    /// exercise the f64 entry points to lock down regressions before the
    /// typed API matured.
    impl OrbitalElements<SelfPlanet> {
        pub(super) fn from_cartesian(
            mu: f64,
            pos: DVec3,
            vel: DVec3,
        ) -> Result<OrbitalElements<SelfPlanet>, OrbitalError> {
            Self::from_cartesian_impl(mu, pos, vel)
        }
    }

    const MU_EARTH: f64 = 398_600.441_50; // km^3/s^2, JEOD earth_GGM05C.cc:40

    /// Helper: verify round-trip from_cartesian -> to_cartesian.
    fn roundtrip_check(mu: f64, pos: DVec3, vel: DVec3, tol: f64) {
        let oe = OrbitalElements::from_cartesian(mu, pos, vel).unwrap();
        let (pos2, vel2) = oe.to_cartesian(mu).unwrap();

        let pos_err = (pos2 - pos).length();
        let vel_err = (vel2 - vel).length();

        assert!(
            pos_err < tol,
            "Position round-trip error {:.2e} exceeds tolerance {:.2e}\n\
             pos={:?}\npos2={:?}\noe={:#?}",
            pos_err,
            tol,
            pos,
            pos2,
            oe
        );
        assert!(
            vel_err < tol,
            "Velocity round-trip error {:.2e} exceeds tolerance {:.2e}\n\
             vel={:?}\nvel2={:?}\noe={:#?}",
            vel_err,
            tol,
            vel,
            vel2,
            oe
        );
    }

    // ---------------------------------------------------------------
    // Circular orbit (e ~ 0)
    // ---------------------------------------------------------------
    #[test]
    fn roundtrip_circular() {
        let r = 6778.0; // km  (ISS-like)
        let v = (MU_EARTH / r).sqrt(); // circular velocity
        let pos = DVec3::new(r, 0.0, 0.0);
        let vel = DVec3::new(0.0, v, 0.0);
        roundtrip_check(MU_EARTH, pos, vel, 1e-10);
    }

    // ---------------------------------------------------------------
    // Eccentric orbits (e = 0.3, 0.7)
    // ---------------------------------------------------------------
    #[test]
    fn roundtrip_eccentric_03() {
        let a = 10000.0;
        let e = 0.3;
        // At periapsis: r = a(1-e), v = sqrt(mu*(1+e)/(a*(1-e)))
        let r = a * (1.0 - e);
        let v = (MU_EARTH * (1.0 + e) / (a * (1.0 - e))).sqrt();
        let pos = DVec3::new(r, 0.0, 0.0);
        let vel = DVec3::new(0.0, v, 0.0);
        roundtrip_check(MU_EARTH, pos, vel, 1e-10);
    }

    #[test]
    fn roundtrip_eccentric_07() {
        let a = 20000.0;
        let e = 0.7;
        let r = a * (1.0 - e);
        let v = (MU_EARTH * (1.0 + e) / (a * (1.0 - e))).sqrt();
        let pos = DVec3::new(r, 0.0, 0.0);
        let vel = DVec3::new(0.0, v, 0.0);
        roundtrip_check(MU_EARTH, pos, vel, 1e-10);
    }

    // ---------------------------------------------------------------
    // Polar orbit (i = 90 degrees)
    // ---------------------------------------------------------------
    #[test]
    fn roundtrip_polar() {
        let r = 7000.0;
        let v = (MU_EARTH / r).sqrt();
        // Velocity in Z direction -> i = 90 degrees
        let pos = DVec3::new(r, 0.0, 0.0);
        let vel = DVec3::new(0.0, 0.0, v);
        roundtrip_check(MU_EARTH, pos, vel, 1e-10);
    }

    // ---------------------------------------------------------------
    // Hyperbolic orbit (e = 1.5)
    // ---------------------------------------------------------------
    #[test]
    fn roundtrip_hyperbolic() {
        // Hyperbolic: energy > 0
        let r = 7000.0;
        let e = 1.5;
        // At periapsis of a hyperbola:  r_p = a(e-1), a = -mu/(2*energy) < 0
        // Choose a so that r_p = r => a = -r/(e-1) (negative)
        let a = -r / (e - 1.0); // a < 0 since e > 1
        let v = (MU_EARTH * (2.0 / r - 1.0 / a)).sqrt(); // vis-viva
        let pos = DVec3::new(r, 0.0, 0.0);
        let vel = DVec3::new(0.0, v, 0.0);
        roundtrip_check(MU_EARTH, pos, vel, 1e-8);
    }

    // ---------------------------------------------------------------
    // Near-parabolic orbit (e = 1 + 1e-3)
    // ---------------------------------------------------------------
    #[test]
    fn roundtrip_near_parabolic() {
        let r = 7000.0;
        let e = 1.0 + 1e-3;
        let a = -r / (e - 1.0);
        let v = (MU_EARTH * (2.0 / r - 1.0 / a)).sqrt();
        let pos = DVec3::new(r, 0.0, 0.0);
        let vel = DVec3::new(0.0, v, 0.0);
        // Near-parabolic has larger numerical sensitivity; use relaxed tolerance
        roundtrip_check(MU_EARTH, pos, vel, 1e-4);
    }

    // ---------------------------------------------------------------
    // Inclined eccentric orbit (general case)
    // ---------------------------------------------------------------
    #[test]
    fn roundtrip_inclined_eccentric() {
        // Build state from known elements:
        //   a = 12000, e = 0.4, i = 45 deg, Omega = 30 deg, omega = 60 deg, nu = 120 deg
        let a = 12000.0;
        let e = 0.4;
        let i = 45.0_f64.to_radians();
        let omega_big = 30.0_f64.to_radians();
        let omega_small = 60.0_f64.to_radians();
        let nu = 120.0_f64.to_radians();
        let p = a * (1.0 - e * e);
        let r = p / (1.0 + e * nu.cos());

        // Perifocal frame
        let r_pqw = DVec3::new(r * nu.cos(), r * nu.sin(), 0.0);
        let coeff = (MU_EARTH / p).sqrt();
        let v_pqw = DVec3::new(-coeff * nu.sin(), coeff * (e + nu.cos()), 0.0);

        // Rotation PQW -> IJK
        let co = omega_big.cos();
        let so = omega_big.sin();
        let cw = omega_small.cos();
        let sw = omega_small.sin();
        let ci = i.cos();
        let si = i.sin();

        let row0 = DVec3::new(co * cw - so * sw * ci, -co * sw - so * cw * ci, so * si);
        let row1 = DVec3::new(so * cw + co * sw * ci, -so * sw + co * cw * ci, -co * si);
        let row2 = DVec3::new(sw * si, cw * si, ci);

        // mat3_from_rows builds PQW->IJK directly: (M * v)[i] = row_i . v.
        // This is the same construction used in to_cartesian(), so no
        // transpose is needed.
        let rot = mat3_from_rows(row0, row1, row2);

        let pos = rot * r_pqw;
        let vel = rot * v_pqw;

        roundtrip_check(MU_EARTH, pos, vel, 1e-8);
    }

    // ---------------------------------------------------------------
    // Kepler solver tests
    // ---------------------------------------------------------------
    #[test]
    fn kepler_elliptic_m_zero() {
        let ea = kep_eqtn_e(0.0, 0.5).unwrap();
        assert!(
            (ea).abs() < 1e-14 || (ea - TAU).abs() < 1e-14,
            "E(M=0) should be 0 (or 2pi), got {}",
            ea
        );
    }

    #[test]
    fn kepler_elliptic_m_pi() {
        let ea = kep_eqtn_e(PI, 0.5).unwrap();
        assert!(
            (ea - PI).abs() < 1e-13,
            "E(M=pi, e=0.5) should be ~pi, got {}",
            ea
        );
    }

    #[test]
    fn kepler_elliptic_high_ecc() {
        let e = 0.98;
        for m in [0.01, 0.5, PI, 5.0] {
            let ea = kep_eqtn_e(m, e).unwrap();
            // Verify M = E - e*sin(E)
            let m_check = ea - e * ea.sin();
            let m_wrapped = wrap_to_tau(m);
            let m_check_wrapped = wrap_to_tau(m_check);
            let diff = (m_wrapped - m_check_wrapped).abs();
            let diff = diff.min(TAU - diff);
            assert!(
                diff < 1e-13,
                "Kepler check failed: M={}, e={}, E={}, M_recomputed={}",
                m,
                e,
                ea,
                m_check
            );
        }
    }

    #[test]
    fn kepler_hyperbolic_convergence() {
        let e = 2.0;
        let m = 5.0;
        let ha = kep_eqtn_h(m, e).unwrap();
        let m_check = e * ha.sinh() - ha;
        assert!(
            (m - m_check).abs() < 1e-13,
            "Hyperbolic Kepler: M={}, H={}, M_check={}",
            m,
            ha,
            m_check
        );
    }

    #[test]
    fn kepler_hyperbolic_extreme_eccentricity() {
        // JEOD 4-case heuristic prevents sinh overflow for large e*M
        for (e, m) in [(10.0, 100.0), (5.0, 50.0), (1.5, 0.1), (3.0, 10.0)] {
            let ha = kep_eqtn_h(m, e).unwrap();
            let m_check = e * ha.sinh() - ha;
            assert!(
                (m - m_check).abs() < 1e-10,
                "Extreme hyperbolic Kepler: e={}, M={}, H={}, M_check={}, err={}",
                e,
                m,
                ha,
                m_check,
                (m - m_check).abs()
            );
        }
    }

    #[test]
    fn kepler_parabolic() {
        // M = D + D^3/3.  For D = 1: M = 1 + 1/3 = 4/3
        let m = 4.0 / 3.0;
        let d = kep_eqtn_b(m);
        let m_check = d + d * d * d / 3.0;
        assert!(
            (m - m_check).abs() < 1e-10,
            "Parabolic Kepler: M={}, D={}, M_check={}",
            m,
            d,
            m_check
        );
    }

    #[test]
    fn kepler_parabolic_negative_m() {
        // M = D + D^3/3.  For D = -1: M = -1 - 1/3 = -4/3
        let m = -4.0 / 3.0;
        let d = kep_eqtn_b(m);
        let m_check = d + d * d * d / 3.0;
        assert!(
            (m - m_check).abs() < 1e-10,
            "Parabolic Kepler negative M: M={}, D={}, M_check={}",
            m,
            d,
            m_check
        );
    }

    // ---------------------------------------------------------------
    // Parabolic / near-parabolic mean motion
    // ---------------------------------------------------------------
    #[test]
    fn near_parabolic_mean_motion_is_positive() {
        // Near-parabolic orbit (e ~ 1): mean_motion must be positive and finite.
        // Regression test: the old code set n=0 for true parabolic orbits.
        let r = 7000.0;
        let v = (2.0 * MU_EARTH / r).sqrt(); // parabolic escape velocity
        let pos = DVec3::new(r, 0.0, 0.0);
        let vel = DVec3::new(0.0, v, 0.0);

        let oe = OrbitalElements::from_cartesian(MU_EARTH, pos, vel).unwrap();

        assert!(
            oe.mean_motion > 0.0 && oe.mean_motion.is_finite(),
            "Near-parabolic mean_motion should be positive and finite, got {}",
            oe.mean_motion,
        );
        // e should be ~1 (within the near-parabolic band)
        assert!(
            (oe.e_mag - 1.0).abs() < 0.02,
            "Expected e ~ 1, got {}",
            oe.e_mag,
        );
    }

    #[test]
    fn true_parabolic_mean_motion_formula() {
        // Exercise the parabolic branch in from_cartesian.
        // A parabolic orbit has v = sqrt(2*mu/r), giving e ~ 1.
        // JEOD nulls out semi_major_axis to 0.0 for parabolic orbits (it is
        // actually infinity) and uses p = h^2/mu, n = 2*sqrt(mu/p)/p.
        let mu: f64 = 1.0; // unit gravitational parameter
        let r: f64 = 1.0e6; // large radius
        let v = (2.0 * mu / r).sqrt(); // parabolic escape speed
        let pos = DVec3::new(r, 0.0, 0.0);
        let vel = DVec3::new(0.0, v, 0.0);

        let oe = OrbitalElements::from_cartesian(mu, pos, vel).unwrap();

        // Verify we hit the parabolic branch: JEOD sets a = 0 for parabolic orbits
        assert_eq!(
            oe.semi_major_axis, 0.0,
            "Parabolic a should be 0.0 (JEOD convention: nulled out)"
        );

        // The JEOD formula: n = 2*sqrt(mu/p)/p
        let p = oe.semiparam;
        let expected_n = 2.0 * (mu / p).sqrt() / p;
        assert!(
            (oe.mean_motion - expected_n).abs() / expected_n < 1e-12,
            "mean_motion mismatch: got {:.6e}, expected {:.6e}",
            oe.mean_motion,
            expected_n,
        );
    }

    // ---------------------------------------------------------------
    // Energy check
    // ---------------------------------------------------------------
    #[test]
    fn energy_check() {
        let r = 7000.0;
        let v = (MU_EARTH / r).sqrt() * 1.1; // slightly above circular
        let pos = DVec3::new(r, 0.0, 0.0);
        let vel = DVec3::new(0.0, v, 0.0);

        let oe = OrbitalElements::from_cartesian(MU_EARTH, pos, vel).unwrap();
        let expected_energy = v * v / 2.0 - MU_EARTH / r;
        assert!(
            (oe.orb_energy - expected_energy).abs() < 1e-10,
            "Energy mismatch: {} vs {}",
            oe.orb_energy,
            expected_energy
        );
    }

    // ---------------------------------------------------------------
    // ISS-like orbit: verify semi-major axis and r_mag
    // ---------------------------------------------------------------
    #[test]
    fn iss_orbit() {
        let alt = 408.0; // km above Earth surface
        let r_earth = 6371.0; // km
        let r = r_earth + alt; // ~6779 km
        let v = (MU_EARTH / r).sqrt();
        let pos = DVec3::new(r, 0.0, 0.0);
        let vel = DVec3::new(0.0, v, 0.0);

        let oe = OrbitalElements::from_cartesian(MU_EARTH, pos, vel).unwrap();

        assert!(
            (oe.r_mag - r).abs() < 1e-8,
            "r_mag should be ~{}, got {}",
            r,
            oe.r_mag
        );
        // For circular orbit, a ~ r
        assert!(
            (oe.semi_major_axis - r).abs() < 1.0, // within 1 km
            "semi_major_axis should be ~{}, got {}",
            r,
            oe.semi_major_axis
        );
        assert!(
            oe.e_mag < 1e-10,
            "eccentricity should be ~0, got {}",
            oe.e_mag
        );
    }

    // ---------------------------------------------------------------
    // Mean anomaly <-> true anomaly round-trip
    // ---------------------------------------------------------------
    #[test]
    fn anomaly_roundtrip_elliptic() {
        let a = 10000.0;
        let e = 0.5;
        let r = a * (1.0 - e);
        let v = (MU_EARTH * (1.0 + e) / (a * (1.0 - e))).sqrt();
        let pos = DVec3::new(r, 0.0, 0.0);
        // Add velocity component in X to get a non-trivial true anomaly position
        let vel = DVec3::new(1.0, v, 0.0);

        let oe = OrbitalElements::from_cartesian(MU_EARTH, pos, vel).unwrap();

        // Now take the mean anomaly and convert back to true anomaly
        let mut oe2 = oe.clone();
        oe2.mean_anom_to_nu().unwrap();

        let nu_diff = (oe.true_anom - oe2.true_anom).abs();
        let nu_diff = nu_diff.min(TAU - nu_diff);
        assert!(
            nu_diff < 1e-13,
            "True anomaly round-trip error: {} (original {} vs reconstructed {})",
            nu_diff,
            oe.true_anom,
            oe2.true_anom,
        );
    }

    // ---------------------------------------------------------------
    // Retrograde equatorial orbit
    // ---------------------------------------------------------------
    #[test]
    fn roundtrip_retrograde_equatorial() {
        let r = 8000.0;
        let v = (MU_EARTH / r).sqrt();
        // Negative Y velocity -> retrograde
        let pos = DVec3::new(r, 0.0, 0.0);
        let vel = DVec3::new(0.0, -v, 0.0);
        roundtrip_check(MU_EARTH, pos, vel, 1e-8);
    }

    // ---------------------------------------------------------------
    // Non-trivial starting position (not at periapsis)
    // ---------------------------------------------------------------
    #[test]
    fn roundtrip_nonperiapsis() {
        // Position at 45 degrees from X axis
        let r = 9000.0;
        let angle = PI / 4.0;
        let pos = DVec3::new(r * angle.cos(), r * angle.sin(), 0.0);
        // Velocity perpendicular to position for near-circular
        let v = (MU_EARTH / r).sqrt();
        let vel = DVec3::new(-v * angle.sin(), v * angle.cos(), 0.0);
        roundtrip_check(MU_EARTH, pos, vel, 1e-10);
    }

    // ---------------------------------------------------------------
    // Invalid inputs
    // ---------------------------------------------------------------
    #[test]
    fn invalid_mu() {
        let pos = DVec3::new(7000.0, 0.0, 0.0);
        let vel = DVec3::new(0.0, 7.0, 0.0);
        assert!(OrbitalElements::from_cartesian(-1.0, pos, vel).is_err());
        assert!(OrbitalElements::from_cartesian(0.0, pos, vel).is_err());
    }

    #[test]
    fn degenerate_orbit() {
        let zero = DVec3::ZERO;
        let vel = DVec3::new(0.0, 7.0, 0.0);
        assert!(OrbitalElements::from_cartesian(MU_EARTH, zero, vel).is_err());
    }

    /// Verify from_cartesian recovers known orbital elements from a state
    /// constructed via the same PQW->IJK rotation used in to_cartesian.
    #[test]
    fn from_cartesian_known_elements() {
        let a = 12000.0;
        let e = 0.4;
        let i = 45.0_f64.to_radians();
        let omega_big = 30.0_f64.to_radians(); // RAAN
        let omega_small = 60.0_f64.to_radians(); // arg periapsis
        let nu = 120.0_f64.to_radians(); // true anomaly

        let p = a * (1.0 - e * e);
        let r = p / (1.0 + e * nu.cos());

        // Position and velocity in perifocal (PQW) frame
        let r_pqw = DVec3::new(r * nu.cos(), r * nu.sin(), 0.0);
        let coeff = (MU_EARTH / p).sqrt();
        let v_pqw = DVec3::new(-coeff * nu.sin(), coeff * (e + nu.cos()), 0.0);

        // Build PQW->IJK rotation matrix using the same approach as to_cartesian:
        // mat3_from_rows builds M such that (M * v)[i] = row_i . v, no transpose needed.
        let co = omega_big.cos();
        let so = omega_big.sin();
        let cw = omega_small.cos();
        let sw = omega_small.sin();
        let ci = i.cos();
        let si = i.sin();

        let row0 = DVec3::new(co * cw - so * sw * ci, -co * sw - so * cw * ci, so * si);
        let row1 = DVec3::new(so * cw + co * sw * ci, -so * sw + co * cw * ci, -co * si);
        let row2 = DVec3::new(sw * si, cw * si, ci);
        let rot = mat3_from_rows(row0, row1, row2);

        let pos = rot * r_pqw;
        let vel = rot * v_pqw;

        // Recover elements via from_cartesian
        let oe = OrbitalElements::from_cartesian(MU_EARTH, pos, vel).unwrap();

        // Verify recovered elements match the input
        assert!(
            (oe.semi_major_axis - a).abs() < 1e-8,
            "semi_major_axis: expected {}, got {}",
            a,
            oe.semi_major_axis
        );
        assert!(
            (oe.e_mag - e).abs() < 1e-10,
            "eccentricity: expected {}, got {}",
            e,
            oe.e_mag
        );
        assert!(
            (oe.inclination - i).abs() < 1e-10,
            "inclination: expected {}, got {}",
            i,
            oe.inclination
        );
        assert!(
            (oe.long_asc_node - omega_big).abs() < 1e-10,
            "RAAN: expected {}, got {}",
            omega_big,
            oe.long_asc_node
        );
        assert!(
            (oe.arg_periapsis - omega_small).abs() < 1e-10,
            "arg_periapsis: expected {}, got {}",
            omega_small,
            oe.arg_periapsis
        );

        // True anomaly wraps to [0, 2pi), so compare modulo 2pi
        let nu_diff = (oe.true_anom - nu).abs();
        let nu_diff = nu_diff.min(TAU - nu_diff);
        assert!(
            nu_diff < 1e-10,
            "true_anom: expected {}, got {} (diff {})",
            nu,
            oe.true_anom,
            nu_diff
        );
    }

    // ---------------------------------------------------------------
    // Typed API: from_cartesian_typed
    // ---------------------------------------------------------------
    //
    // These tests exercise the dimensionally-typed entry point. SI base
    // units are required: m, m/s, m³/s². JEOD's Earth GM constant converted
    // to SI from earth_GGM05C.cc:40 (398_600.441_50 km³/s² -> 3.986004415e14
    // m³/s²).
    #[test]
    fn from_cartesian_typed_iss_like() {
        use jeod_quantities::aliases::{Position, Velocity};
        use jeod_quantities::ext::F64Ext;
        use jeod_quantities::frame::{Earth, PlanetInertial};
        use jeod_quantities::qty3::Qty3;

        // ISS-ish: 408 km altitude circular orbit, SI units.
        // The planet phantom on `mu_si` is pinned to `Earth` so it
        // matches the position/velocity frames at the call site —
        // mismatching `<Earth>` and `<Sun>` here is a compile error.
        let mu_si: GravParam<Earth> = 3.986_004_415e14_f64.m3_per_s2_for::<Earth>();
        let r = 6_779_000.0_f64; // m (~6371 + 408 km)
        let v = (3.986_004_415e14_f64 / r).sqrt(); // m/s

        // Qty3::from_raw_si wraps a DVec3 of SI-base-unit values in the
        // typed frame-tagged envelope without any unit conversion.
        let pos: Position<PlanetInertial<Earth>> = Qty3::from_raw_si(DVec3::new(r, 0.0, 0.0));
        let vel: Velocity<PlanetInertial<Earth>> = Qty3::from_raw_si(DVec3::new(0.0, v, 0.0));

        let oe = OrbitalElements::<Earth>::from_cartesian_typed(mu_si, pos, vel).unwrap();

        // ISS-like semi-major axis falls in the 6.5e6 - 7.2e6 m band.
        assert!(
            oe.semi_major_axis > 6.5e6 && oe.semi_major_axis < 7.2e6,
            "semi_major_axis {} out of ISS range [6.5e6, 7.2e6]",
            oe.semi_major_axis
        );
        // Circular orbit has near-zero eccentricity.
        assert!(
            oe.e_mag < 1e-10,
            "eccentricity should be ~0 for circular orbit, got {}",
            oe.e_mag
        );
        // r_mag matches the input radius in meters.
        assert!(
            (oe.r_mag - r).abs() < 1e-6,
            "r_mag should be ~{r}, got {}",
            oe.r_mag
        );
    }

    #[test]
    fn from_cartesian_typed_matches_raw_bit_for_bit() {
        use jeod_quantities::aliases::{Position, Velocity};
        use jeod_quantities::ext::F64Ext;
        use jeod_quantities::frame::{Earth, PlanetInertial};
        use jeod_quantities::qty3::Qty3;

        let mu_si: GravParam<Earth> = 3.986_004_415e14_f64.m3_per_s2_for::<Earth>();
        // Mildly eccentric, slightly inclined ISS-ish state in SI units.
        let pos_raw = DVec3::new(6_779_000.0, 0.0, 0.0);
        let vel_raw = DVec3::new(0.0, 7_000.0, 1_500.0);

        let pos: Position<PlanetInertial<Earth>> = Qty3::from_raw_si(pos_raw);
        let vel: Velocity<PlanetInertial<Earth>> = Qty3::from_raw_si(vel_raw);

        let oe_typed = OrbitalElements::<Earth>::from_cartesian_typed(mu_si, pos, vel).unwrap();
        let oe_raw = OrbitalElements::from_cartesian(mu_si.value, pos_raw, vel_raw).unwrap();

        // Bit-identical delegation: typed wrapper extracts SI values and calls
        // the raw implementation — no intermediate arithmetic, so every field
        // must match exactly.
        assert_eq!(oe_typed.semi_major_axis, oe_raw.semi_major_axis);
        assert_eq!(oe_typed.semiparam, oe_raw.semiparam);
        assert_eq!(oe_typed.e_mag, oe_raw.e_mag);
        assert_eq!(oe_typed.inclination, oe_raw.inclination);
        assert_eq!(oe_typed.arg_periapsis, oe_raw.arg_periapsis);
        assert_eq!(oe_typed.long_asc_node, oe_raw.long_asc_node);
        assert_eq!(oe_typed.true_anom, oe_raw.true_anom);
        assert_eq!(oe_typed.mean_anom, oe_raw.mean_anom);
        assert_eq!(oe_typed.orbital_anom, oe_raw.orbital_anom);
        assert_eq!(oe_typed.mean_motion, oe_raw.mean_motion);
        assert_eq!(oe_typed.r_mag, oe_raw.r_mag);
        assert_eq!(oe_typed.vel_mag, oe_raw.vel_mag);
        assert_eq!(oe_typed.orb_energy, oe_raw.orb_energy);
        assert_eq!(oe_typed.orb_ang_momentum, oe_raw.orb_ang_momentum);
    }
}
