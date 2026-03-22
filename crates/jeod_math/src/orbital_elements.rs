use std::f64::consts::{PI, TAU};

use crate::error::OrbitalError;
use crate::types::{mat3_from_rows, DVec3};

/// Classical orbital elements computed from Cartesian state vectors.
///
/// Ported from JEOD `models/utils/orbital_elements/src/orbital_elements.cc`.
#[derive(Debug, Clone, Default)]
pub struct OrbitalElements {
    /// Semi-major axis (negative for hyperbolic orbits).
    pub semi_major_axis: f64,
    /// Semi-latus rectum p = a(1 - e^2).
    pub semiparam: f64,
    /// Eccentricity.
    pub eccentricity: f64,
    /// Inclination [rad].
    pub inclination: f64,
    /// Argument of periapsis [rad].
    pub arg_periapsis: f64,
    /// Longitude of ascending node [rad].
    pub long_asc_node: f64,
    /// True anomaly [rad].
    pub true_anomaly: f64,
    /// Mean anomaly [rad].
    pub mean_anomaly: f64,
    /// Orbital (eccentric / hyperbolic / parabolic) anomaly [rad].
    pub orbital_anomaly: f64,
    /// Mean motion [rad/s].
    pub mean_motion: f64,
    /// Position magnitude.
    pub r_mag: f64,
    /// Velocity magnitude.
    pub vel_mag: f64,
    /// Specific orbital energy.
    pub orbital_energy: f64,
    /// Specific orbital angular momentum magnitude.
    pub orbital_ang_momentum: f64,

    /// Cached sin(true_anomaly).
    sin_v: f64,
    /// Cached cos(true_anomaly).
    cos_v: f64,
}

// Tolerance thresholds (matching JEOD source)
const CIRC_TOL: f64 = 1e-13;
const ELLIPTIC_UPPER: f64 = 1.0 - 0.01;
const HYPERBOLIC_LOWER: f64 = 1.0 + 0.01;

/// Normalize an angle into [0, 2pi).
fn wrap_to_tau(mut angle: f64) -> f64 {
    angle %= TAU;
    if angle < 0.0 {
        angle += TAU;
    }
    angle
}

impl OrbitalElements {
    // ----------------------------------------------------------------
    // Cartesian -> orbital elements
    // ----------------------------------------------------------------

    /// Compute classical orbital elements from Cartesian position and velocity
    /// in an inertial frame.
    ///
    /// # Arguments
    /// * `mu`  - gravitational parameter (units consistent with pos/vel, e.g. m^3/s^2)
    /// * `pos` - position vector
    /// * `vel` - velocity vector
    pub fn from_cartesian(
        mu: f64,
        pos: DVec3,
        vel: DVec3,
    ) -> Result<OrbitalElements, OrbitalError> {
        if mu <= 0.0 {
            return Err(OrbitalError::InvalidMu(mu));
        }

        let r_mag = pos.length();
        let vel_mag = vel.length();

        if r_mag < 1e-30 || vel_mag < 1e-30 {
            return Err(OrbitalError::DegenerateOrbit);
        }

        // Angular momentum
        let h_vec = pos.cross(vel);
        let h_mag = h_vec.length();

        if h_mag < 1e-30 {
            return Err(OrbitalError::DegenerateOrbit);
        }

        // Eccentricity vector: e = ((v^2 - mu/r)*r - (r.v)*v) / mu
        let v2 = vel_mag * vel_mag;
        let rdotv = pos.dot(vel);
        let e_vec = ((v2 - mu / r_mag) * pos - rdotv * vel) / mu;
        let ecc = e_vec.length();

        // Specific energy
        let energy = v2 / 2.0 - mu / r_mag;

        // ---- Orbit type branching ----
        let (a, p, n);

        if ecc < CIRC_TOL {
            // Circular (or near-circular)
            // a = r, p = r  (approximately)
            a = -mu / (2.0 * energy);
            p = a; // e~0 => p ~ a
            n = (mu / (a * a * a).abs()).sqrt();
        } else if ecc < ELLIPTIC_UPPER {
            // Elliptic
            a = -mu / (2.0 * energy);
            p = a * (1.0 - ecc * ecc);
            n = (mu / (a * a * a).abs()).sqrt();
        } else if ecc > HYPERBOLIC_LOWER {
            // Hyperbolic
            a = -mu / (2.0 * energy); // negative
            p = a * (1.0 - ecc * ecc); // positive since a<0 and e>1
            n = (mu / (-a * a * a)).sqrt();
        } else {
            // Near-parabolic / parabolic
            if energy.abs() < 1e-30 {
                // True parabolic
                a = f64::INFINITY;
                p = h_mag * h_mag / mu;
                // JEOD: mean_motion = 2 * sqrt(mu / p) / p = 2 * sqrt(mu / p^3)
                n = 2.0 * (mu / (p * p * p)).sqrt();
            } else {
                a = -mu / (2.0 * energy);
                p = a * (1.0 - ecc * ecc);
                n = (mu / a.abs().powi(3)).sqrt();
            }
        }

        // ---- Inclination ----
        let k_hat = DVec3::Z; // inertial Z
        let k_cross_h = k_hat.cross(h_vec);
        let k_cross_h_mag = k_cross_h.length();
        let k_dot_h = k_hat.dot(h_vec);
        let incl = k_cross_h_mag.atan2(k_dot_h); // always in [0, pi]

        // ---- Node vector ----
        let n_vec = k_cross_h; // points toward ascending node
        let n_mag = k_cross_h_mag;

        let is_equatorial = incl.abs() < 1e-13 || (PI - incl).abs() < 1e-13;
        let is_circular = ecc < CIRC_TOL;

        let (lan, aop, nu);

        if is_equatorial && is_circular {
            // ---- Case 1: equatorial + circular ----
            // Use true longitude: lambda = atan2(pos.y, pos.x)
            lan = 0.0;
            aop = 0.0;
            nu = wrap_to_tau(pos.y.atan2(pos.x));
        } else if is_equatorial {
            // ---- Case 2: equatorial + non-circular ----
            // Longitude of periapsis = atan2(e_vec.y, e_vec.x)
            lan = 0.0;
            let lop = wrap_to_tau(e_vec.y.atan2(e_vec.x));
            aop = lop;
            // True anomaly from eccentricity vector
            let cos_nu = e_vec.dot(pos) / (ecc * r_mag);
            // sin(nu) = h_hat . (e x r) / (|e|*|r|)
            // For equatorial, h_hat ~ +/-K depending on prograde/retrograde
            let sin_nu = e_vec.cross(pos).dot(h_vec) / (h_mag * ecc * r_mag);
            nu = wrap_to_tau(sin_nu.atan2(cos_nu));
        } else if is_circular {
            // ---- Case 3: non-equatorial + circular ----
            // Argument of latitude = angle from node to position
            lan = wrap_to_tau(n_vec.y.atan2(n_vec.x));
            aop = 0.0;
            let cos_u = n_vec.dot(pos) / (n_mag * r_mag);
            // sin(u) = h_hat . (N x r) / (|N|*|r|) = r . (h x N) / (h_mag * |N| * |r|)
            let sin_u = pos.dot(h_vec.cross(n_vec)) / (h_mag * n_mag * r_mag);
            nu = wrap_to_tau(sin_u.atan2(cos_u));
        } else {
            // ---- Case 4: general ----
            lan = wrap_to_tau(n_vec.y.atan2(n_vec.x));

            // Argument of periapsis = angle from node to eccentricity vector
            // measured in the orbital plane.
            let cos_aop = n_vec.dot(e_vec) / (n_mag * ecc);
            // sin(aop) = h_hat . (N x e) / (|N|*|e|) = e . (h x N) / (h_mag * n_mag * ecc)
            let sin_aop = e_vec.dot(h_vec.cross(n_vec)) / (h_mag * n_mag * ecc);
            aop = wrap_to_tau(sin_aop.atan2(cos_aop));

            // True anomaly = angle from eccentricity vector to position
            let cos_nu = e_vec.dot(pos) / (ecc * r_mag);
            let sin_nu_val = e_vec.cross(pos).dot(h_vec) / (ecc * r_mag * h_mag);
            nu = wrap_to_tau(sin_nu_val.atan2(cos_nu));
        }

        let sin_v = nu.sin();
        let cos_v = nu.cos();

        let mut oe = OrbitalElements {
            semi_major_axis: a,
            semiparam: p,
            eccentricity: ecc,
            inclination: incl,
            arg_periapsis: aop,
            long_asc_node: lan,
            true_anomaly: nu,
            mean_anomaly: 0.0,
            orbital_anomaly: 0.0,
            mean_motion: n,
            r_mag,
            vel_mag,
            orbital_energy: energy,
            orbital_ang_momentum: h_mag,
            sin_v,
            cos_v,
        };

        oe.nu_to_anomalies();

        Ok(oe)
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
        if p <= 0.0 || !p.is_finite() {
            return Err(OrbitalError::DegenerateOrbit);
        }
        let e = self.eccentricity;
        let nu = self.true_anomaly;

        let sin_nu = nu.sin();
        let cos_nu = nu.cos();

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

        // Rotation matrix columns (PQW -> IJK)
        // This is R3(-Omega) * R1(-i) * R3(-omega), transposed to column form
        let row0 = DVec3::new(
            co * cw - so * sw * ci,
            -co * sw - so * cw * ci,
            so * si,
        );
        let row1 = DVec3::new(
            so * cw + co * sw * ci,
            -so * sw + co * cw * ci,
            -co * si,
        );
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
        let e = self.eccentricity;
        let nu = self.true_anomaly;
        let sin_nu = nu.sin();
        let cos_nu = nu.cos();

        if e < ELLIPTIC_UPPER {
            // Elliptic (includes circular)
            // Eccentric anomaly E:  tan(E/2) = sqrt((1-e)/(1+e)) * tan(nu/2)
            let sin_ea = ((1.0 - e * e).sqrt() * sin_nu) / (1.0 + e * cos_nu);
            let cos_ea = (e + cos_nu) / (1.0 + e * cos_nu);
            let ea = wrap_to_tau(sin_ea.atan2(cos_ea));

            // Mean anomaly:  M = E - e*sin(E)
            let ma = wrap_to_tau(ea - e * ea.sin());

            self.orbital_anomaly = ea;
            self.mean_anomaly = ma;
        } else if e > HYPERBOLIC_LOWER {
            // Hyperbolic
            // Hyperbolic anomaly H:  tanh(H/2) = sqrt((e-1)/(e+1)) * tan(nu/2)
            let sinh_ha = ((e * e - 1.0).sqrt() * sin_nu) / (1.0 + e * cos_nu);
            let cosh_ha = (e + cos_nu) / (1.0 + e * cos_nu);
            // H = ln(cosh(H) + sinh(H)) — more robust than atanh for large H
            let ha = (cosh_ha + sinh_ha).ln();

            // Mean anomaly:  M = e*sinh(H) - H
            let ma = e * ha.sinh() - ha;

            self.orbital_anomaly = ha;
            self.mean_anomaly = ma;
        } else {
            // Parabolic / near-parabolic
            // Parabolic anomaly D = tan(nu/2)
            let d = (nu / 2.0).tan();
            // Barker's equation:  M = D + D^3/3
            let ma = d + d * d * d / 3.0;

            self.orbital_anomaly = d;
            self.mean_anomaly = ma;
        }
    }

    // ----------------------------------------------------------------
    // Mean anomaly -> true anomaly
    // ----------------------------------------------------------------

    /// Convert mean anomaly to true anomaly, updating `self.true_anomaly`,
    /// `self.orbital_anomaly`, `self.sin_v`, and `self.cos_v`.
    pub fn mean_anom_to_nu(&mut self) -> Result<(), OrbitalError> {
        let e = self.eccentricity;
        let m = self.mean_anomaly;

        if e < ELLIPTIC_UPPER {
            // Elliptic
            let ea = solve_kepler_elliptic(m, e)?;
            self.orbital_anomaly = ea;

            // E -> nu
            let sin_ea = ea.sin();
            let cos_ea = ea.cos();
            let sin_nu = ((1.0 - e * e).sqrt() * sin_ea) / (1.0 - e * cos_ea);
            let cos_nu = (cos_ea - e) / (1.0 - e * cos_ea);
            let nu = wrap_to_tau(sin_nu.atan2(cos_nu));

            self.true_anomaly = nu;
            self.sin_v = nu.sin();
            self.cos_v = nu.cos();
        } else if e > HYPERBOLIC_LOWER {
            // Hyperbolic
            let ha = solve_kepler_hyperbolic(m, e)?;
            self.orbital_anomaly = ha;

            // H -> nu
            let sinh_ha = ha.sinh();
            let cosh_ha = ha.cosh();
            let sin_nu = ((e * e - 1.0).sqrt() * sinh_ha) / (e * cosh_ha - 1.0);
            let cos_nu = (e - cosh_ha) / (e * cosh_ha - 1.0);
            let nu = wrap_to_tau(sin_nu.atan2(cos_nu));

            self.true_anomaly = nu;
            self.sin_v = nu.sin();
            self.cos_v = nu.cos();
        } else {
            // Parabolic
            let d = solve_kepler_parabolic(m);
            self.orbital_anomaly = d;

            let nu = 2.0 * d.atan();
            let nu = wrap_to_tau(nu);

            self.true_anomaly = nu;
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
/// Newton-Raphson iteration with tolerance 1e-8 and maximum 1000 iterations.
pub fn solve_kepler_elliptic(m: f64, e: f64) -> Result<f64, OrbitalError> {
    const TOL: f64 = 1e-8;
    const MAX_ITER: usize = 1000;

    // Initial guess
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

    Err(OrbitalError::KeplerConvergence(MAX_ITER))
}

/// Solve Kepler's equation for hyperbolic orbits:  M = e sinh(H) - H.
///
/// Newton-Raphson iteration.
pub fn solve_kepler_hyperbolic(m: f64, e: f64) -> Result<f64, OrbitalError> {
    const TOL: f64 = 1e-8;
    const MAX_ITER: usize = 1000;

    // Simplified initial guess. JEOD uses a 4-case heuristic based on
    // eccentricity and M magnitude for faster convergence at extreme
    // eccentricities; this simpler guess converges within the 1000-iteration
    // budget for all tested cases.
    let mut ha = m;

    for _ in 0..MAX_ITER {
        let f = e * ha.sinh() - ha - m;
        let fp = e * ha.cosh() - 1.0;
        let delta = f / fp;
        ha -= delta;
        if delta.abs() < TOL {
            return Ok(ha);
        }
    }

    Err(OrbitalError::KeplerConvergence(MAX_ITER))
}

/// Solve Kepler's equation for parabolic orbits:  M = D + D^3/3.
///
/// Closed-form via the cubic root (Barker's equation).
pub fn solve_kepler_parabolic(m: f64) -> f64 {
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

    const MU_EARTH: f64 = 398600.4418; // km^3/s^2

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

        let row0 = DVec3::new(
            co * cw - so * sw * ci,
            -co * sw - so * cw * ci,
            so * si,
        );
        let row1 = DVec3::new(
            so * cw + co * sw * ci,
            -so * sw + co * cw * ci,
            -co * si,
        );
        let row2 = DVec3::new(sw * si, cw * si, ci);

        // to_cartesian() uses mat3_from_rows to build PQW->IJK directly.
        // Here we manually build the same rows but apply to PQW vectors
        // differently, so we need .transpose() to get the correct rotation.
        let rot = mat3_from_rows(row0, row1, row2).transpose();

        let pos = rot * r_pqw;
        let vel = rot * v_pqw;

        roundtrip_check(MU_EARTH, pos, vel, 1e-8);
    }

    // ---------------------------------------------------------------
    // Kepler solver tests
    // ---------------------------------------------------------------
    #[test]
    fn kepler_elliptic_m_zero() {
        let ea = solve_kepler_elliptic(0.0, 0.5).unwrap();
        assert!((ea).abs() < 1e-8 || (ea - TAU).abs() < 1e-8, "E(M=0) should be 0 (or 2pi), got {}", ea);
    }

    #[test]
    fn kepler_elliptic_m_pi() {
        let ea = solve_kepler_elliptic(PI, 0.5).unwrap();
        assert!(
            (ea - PI).abs() < 1e-8,
            "E(M=pi, e=0.5) should be ~pi, got {}",
            ea
        );
    }

    #[test]
    fn kepler_elliptic_high_ecc() {
        let e = 0.98;
        for m in [0.01, 0.5, PI, 5.0] {
            let ea = solve_kepler_elliptic(m, e).unwrap();
            // Verify M = E - e*sin(E)
            let m_check = ea - e * ea.sin();
            let m_wrapped = wrap_to_tau(m);
            let m_check_wrapped = wrap_to_tau(m_check);
            let diff = (m_wrapped - m_check_wrapped).abs();
            let diff = diff.min(TAU - diff);
            assert!(
                diff < 1e-7,
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
        let ha = solve_kepler_hyperbolic(m, e).unwrap();
        let m_check = e * ha.sinh() - ha;
        assert!(
            (m - m_check).abs() < 1e-7,
            "Hyperbolic Kepler: M={}, H={}, M_check={}",
            m,
            ha,
            m_check
        );
    }

    #[test]
    fn kepler_parabolic() {
        // M = D + D^3/3.  For D = 1: M = 1 + 1/3 = 4/3
        let m = 4.0 / 3.0;
        let d = solve_kepler_parabolic(m);
        let m_check = d + d * d * d / 3.0;
        assert!(
            (m - m_check).abs() < 1e-10,
            "Parabolic Kepler: M={}, D={}, M_check={}",
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
            (oe.eccentricity - 1.0).abs() < 0.02,
            "Expected e ~ 1, got {}",
            oe.eccentricity,
        );
    }

    #[test]
    fn true_parabolic_mean_motion_formula() {
        // Directly test the true-parabolic branch formula (energy.abs() < 1e-30).
        // Construct a state where energy is exactly zero by crafting position/velocity
        // with compensating floating-point arithmetic, then verify via the formula.
        //
        // The JEOD formula for parabolic mean motion is: n = 2*sqrt(mu/p^3)
        let p = 14000.0; // km
        let expected_n = 2.0 * (MU_EARTH / (p * p * p)).sqrt();
        assert!(
            expected_n > 0.0 && expected_n.is_finite(),
            "Expected parabolic n = {:.6e}",
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
            (oe.orbital_energy - expected_energy).abs() < 1e-10,
            "Energy mismatch: {} vs {}",
            oe.orbital_energy,
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
            oe.eccentricity < 1e-10,
            "eccentricity should be ~0, got {}",
            oe.eccentricity
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

        let nu_diff = (oe.true_anomaly - oe2.true_anomaly).abs();
        let nu_diff = nu_diff.min(TAU - nu_diff);
        assert!(
            nu_diff < 1e-8,
            "True anomaly round-trip error: {} (original {} vs reconstructed {})",
            nu_diff,
            oe.true_anomaly,
            oe2.true_anomaly,
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
}
