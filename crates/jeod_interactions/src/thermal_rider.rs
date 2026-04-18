//! Thermal-rider model: inter-facet radiation exchange and environmental heating.
//!
//! Port of JEOD `models/interactions/thermal_rider/` with the following scope:
//!
//! * [`ThermalFacet`] — per-facet thermal and optical material properties
//!   (emissivity, solar absorptivity, albedo, conductivity, specific heat,
//!   mass, current temperature). Mirrors JEOD `ThermalFacetRider` +
//!   `ThermalParams` combined into a single Rust struct.
//! * [`ViewFactorMatrix`] — view factors for multi-body IR exchange between
//!   facets (`F_{ij}` = fraction of facet *i*'s diffuse emission intercepted
//!   by facet *j*). Implements reciprocity checks and the standard energy
//!   conservation constraint `sum_j F_{ij} <= 1`.
//! * [`ThermalEnvironment`] — aggregated environmental fluxes (solar,
//!   Earth albedo, planet thermal IR) and their directions in the vehicle
//!   structural frame.
//! * [`compute_thermal_power_balance`] — pure function that returns the
//!   time derivative of temperature for each facet given its absorbed and
//!   emitted powers.
//!
//! JEOD's thermal rider itself is intentionally minimal: the header notes
//! that conduction is "for future implementation" (see
//! `thermal_facet_rider.cc:61-90`) and that the module is a rider on the
//! radiation-pressure surface model, not a stand-alone interaction. All
//! verification tests in JEOD for this module are structural
//! (`verif/unit_tests/*_ut.cc` just construct/destruct instances). This
//! module therefore provides analytical physics tests for the pieces that
//! JEOD leaves implicit, and integrates cleanly with the existing
//! [`crate::radiation_pressure`] SRP-thermal facets.
//!
//! The Stefan-Boltzmann constant is shared with
//! [`crate::radiation_pressure::STEFAN_BOLTZMANN`].

use glam::DVec3;

use crate::radiation_pressure::STEFAN_BOLTZMANN;

/// Thermal and optical material properties for a single facet.
///
/// Combines the material-dependent parameters from JEOD `ThermalParams`
/// (emissivity, heat capacity, thermal power dump) with the surface-
/// interaction optical properties (solar absorptivity, albedo/reflectance)
/// used by radiative environmental heating.
///
/// Units follow JEOD: SI throughout. `mass * specific_heat` is the facet
/// heat capacity `C = m·c_p` (J/K); the [`compute_thermal_power_balance`]
/// function uses this product directly so callers can equivalently store
/// `heat_capacity` and leave either term at 1.0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThermalFacet {
    /// Facet area (m²). Must be > 0 for radiative contributions to be non-trivial.
    pub area: f64,
    /// Infrared emissivity (0..=1). Fraction of σ·T⁴ actually emitted in the IR band.
    pub emissivity_ir: f64,
    /// Solar absorptivity (0..=1). Fraction of incident solar flux absorbed
    /// when the facet is illuminated by the Sun (or Sun-reflected Earth albedo).
    pub absorptivity_solar: f64,
    /// Bolometric albedo (0..=1). Fraction of incident solar flux *reflected*
    /// by this facet — complement of `absorptivity_solar` for an opaque surface
    /// (i.e. `absorptivity_solar + albedo ≈ 1`), exposed as a separate field so
    /// partially transparent materials can be modeled.
    pub albedo: f64,
    /// Bulk thermal conductivity (W/(m·K)). Stored for use by conduction
    /// solvers outside this function; this module itself does not discretize
    /// conduction (JEOD likewise leaves this for future work, see
    /// `thermal_facet_rider.cc:61-90`).
    pub conductivity: f64,
    /// Specific heat capacity (J/(kg·K)).
    pub specific_heat: f64,
    /// Facet mass (kg). Heat capacity used in dT/dt is `mass * specific_heat`.
    pub mass: f64,
    /// Current facet temperature (K).
    pub temperature: f64,
}

impl ThermalFacet {
    /// Lumped heat capacity `m·c_p` (J/K) used by the temperature ODE.
    #[inline]
    pub fn heat_capacity(&self) -> f64 {
        self.mass * self.specific_heat
    }

    /// Stefan-Boltzmann radiative constant `A·ε·σ` (W/K⁴) used by the
    /// temperature ODE and by view-factor exchange.
    #[inline]
    pub fn radiative_constant(&self) -> f64 {
        self.area * self.emissivity_ir * STEFAN_BOLTZMANN
    }
}

/// Square view-factor matrix between *N* facets.
///
/// `factors[i][j]` is the fraction of the diffuse thermal radiation leaving
/// facet *i* that is intercepted directly by facet *j*. View factors are
/// subject to two physical constraints:
///
/// 1. **Reciprocity**: `A_i · F_{ij} = A_j · F_{ji}` for every pair.
/// 2. **Conservation**: `sum_j F_{ij} ≤ 1` for every *i* (the remainder is
///    emitted to space / the environment). Equality holds only for a fully
///    enclosed cavity; most spacecraft facets have a significant view to
///    deep space.
///
/// [`ViewFactorMatrix::validate`] checks both constraints within a tolerance.
#[derive(Debug, Clone, Default)]
pub struct ViewFactorMatrix {
    /// Square matrix `F_{ij}`; must be `factors.len() == factors[i].len()` for all `i`.
    factors: Vec<Vec<f64>>,
}

/// Failure returned by [`ViewFactorMatrix::validate`].
#[derive(Debug, Clone, PartialEq)]
pub enum ViewFactorError {
    /// Matrix is not square.
    NotSquare {
        /// Outer dimension of the matrix.
        rows: usize,
        /// First offending row length.
        bad_cols: usize,
    },
    /// Areas slice does not match matrix size.
    AreaLengthMismatch {
        /// Matrix dimension.
        n: usize,
        /// Number of areas supplied.
        areas: usize,
    },
    /// An entry is outside `[0, 1]` beyond `tol`.
    OutOfRange {
        /// Row index.
        i: usize,
        /// Column index.
        j: usize,
        /// Offending value.
        value: f64,
    },
    /// Reciprocity `A_i·F_{ij} = A_j·F_{ji}` violated beyond `tol`.
    ReciprocityViolation {
        /// First facet index.
        i: usize,
        /// Second facet index.
        j: usize,
        /// `A_i · F_{ij}`.
        lhs: f64,
        /// `A_j · F_{ji}`.
        rhs: f64,
    },
    /// Row sum exceeds `1 + tol`.
    RowSumExceedsUnity {
        /// Row index.
        i: usize,
        /// Observed `sum_j F_{ij}`.
        sum: f64,
    },
}

impl ViewFactorMatrix {
    /// Construct a view-factor matrix from its raw rows. No validation is
    /// performed — call [`Self::validate`] to check physical constraints.
    #[inline]
    pub fn new(factors: Vec<Vec<f64>>) -> Self {
        Self { factors }
    }

    /// Matrix dimension (number of facets).
    #[inline]
    pub fn dim(&self) -> usize {
        self.factors.len()
    }

    /// Read-only access to the underlying rows.
    #[inline]
    pub fn rows(&self) -> &[Vec<f64>] {
        &self.factors
    }

    /// `F_{ij}` — fraction of facet *i*'s emission intercepted by facet *j*.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.factors[i][j]
    }

    /// Construct a diagonal-zero matrix where every off-diagonal entry is
    /// `uniform` and every diagonal entry is 0 (a facet cannot see itself
    /// for a convex geometry).
    pub fn uniform_offdiagonal(n: usize, uniform: f64) -> Self {
        let mut factors = vec![vec![0.0_f64; n]; n];
        for (i, row) in factors.iter_mut().enumerate().take(n) {
            for (j, entry) in row.iter_mut().enumerate().take(n) {
                if i != j {
                    *entry = uniform;
                }
            }
        }
        Self { factors }
    }

    /// Verify that the matrix satisfies range (`0 ≤ F_{ij} ≤ 1`),
    /// reciprocity (`A_i·F_{ij} = A_j·F_{ji}`) and row-sum
    /// conservation (`sum_j F_{ij} ≤ 1`). `tol` is applied additively to
    /// every constraint.
    pub fn validate(&self, areas: &[f64], tol: f64) -> Result<(), ViewFactorError> {
        let n = self.factors.len();
        for (i, row) in self.factors.iter().enumerate() {
            if row.len() != n {
                return Err(ViewFactorError::NotSquare {
                    rows: n,
                    bad_cols: row.len(),
                });
            }
            for (j, &f) in row.iter().enumerate() {
                if f < -tol || f > 1.0 + tol {
                    return Err(ViewFactorError::OutOfRange { i, j, value: f });
                }
            }
        }
        if areas.len() != n {
            return Err(ViewFactorError::AreaLengthMismatch {
                n,
                areas: areas.len(),
            });
        }
        // Reciprocity: A_i * F_ij = A_j * F_ji.
        for i in 0..n {
            for j in (i + 1)..n {
                let lhs = areas[i] * self.factors[i][j];
                let rhs = areas[j] * self.factors[j][i];
                if (lhs - rhs).abs() > tol {
                    return Err(ViewFactorError::ReciprocityViolation { i, j, lhs, rhs });
                }
            }
        }
        // Row-sum conservation.
        for (i, row) in self.factors.iter().enumerate() {
            let sum: f64 = row.iter().sum();
            if sum > 1.0 + tol {
                return Err(ViewFactorError::RowSumExceedsUnity { i, sum });
            }
        }
        Ok(())
    }
}

/// Environmental heat sources seen by a vehicle, expressed in the vehicle's
/// *structural* frame.
///
/// Fluxes follow the JEOD convention (`RadiationSource::calculate_flux`):
/// each is a scalar W/m² evaluated at the vehicle position, and each has a
/// unit direction in the structural frame pointing *from the source toward
/// the vehicle* — i.e. the direction the flux is travelling.
///
/// * `solar_flux` uses `sun_direction`.
/// * `earth_albedo_flux` uses `earth_direction` (reflected sunlight reaches
///   the vehicle from Earth below).
/// * `earth_ir_flux` uses `earth_direction` (thermal IR also propagates
///   from the planet outward).
///
/// For orbital-mechanics use, the caller rotates the Sun- and Earth-relative
/// directions from inertial (or ECEF) into the structural frame before
/// constructing this struct.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThermalEnvironment {
    /// Solar flux at the vehicle (W/m²). ≈1361 at 1 AU.
    pub solar_flux: f64,
    /// Earth-albedo flux (reflected solar) at the vehicle (W/m²).
    /// Zero on the night side of Earth.
    pub earth_albedo_flux: f64,
    /// Earth thermal-IR flux at the vehicle (W/m²). Roughly 240 W/m² at LEO,
    /// nearly isotropic over the Earth disk.
    pub earth_ir_flux: f64,
    /// Unit vector from Sun toward vehicle, in the structural frame.
    pub sun_direction: DVec3,
    /// Unit vector from Earth toward vehicle, in the structural frame.
    pub earth_direction: DVec3,
}

impl Default for ThermalEnvironment {
    fn default() -> Self {
        Self {
            solar_flux: 0.0,
            earth_albedo_flux: 0.0,
            earth_ir_flux: 0.0,
            sun_direction: DVec3::X,
            earth_direction: -DVec3::Z,
        }
    }
}

/// Time derivative of temperature (K/s) and the decomposed absorbed and
/// emitted powers (W) for every facet.
///
/// The fields `q_solar`, `q_albedo`, `q_ir`, `q_viewfactor`, and `q_emitted`
/// are returned so callers can trace which environmental term dominates the
/// facet's thermal budget, or log the energy balance at a given timestep.
#[derive(Debug, Clone)]
pub struct ThermalPowerBalance {
    /// Per-facet dT/dt (K/s). Length = number of facets.
    pub temp_dots: Vec<f64>,
    /// Absorbed solar power per facet (W).
    pub q_solar: Vec<f64>,
    /// Absorbed Earth-albedo power per facet (W).
    pub q_albedo: Vec<f64>,
    /// Absorbed Earth-IR power per facet (W).
    pub q_ir: Vec<f64>,
    /// Net radiative exchange power per facet (W). Positive = net gain.
    pub q_viewfactor: Vec<f64>,
    /// Stefan-Boltzmann emission power per facet (W). Always ≥ 0.
    pub q_emitted: Vec<f64>,
}

/// Compute the per-facet temperature time derivative dT/dt from the
/// environmental, inter-facet, and Stefan-Boltzmann emission terms.
///
/// The temperature ODE per facet is
///
/// ```text
/// m·c_p · dT_i/dt = Q_solar_i + Q_albedo_i + Q_ir_i + Q_viewfactor_i - Q_emitted_i
/// ```
///
/// where, for facet `i` with outward structural-frame normal `n_i`:
///
/// | Term          | Formula |
/// |---------------|---------|
/// | `Q_solar`     | `absorptivity_solar · solar_flux · max(0, -n·s) · area` |
/// | `Q_albedo`    | `absorptivity_solar · earth_albedo_flux · max(0, -n·e) · area` |
/// | `Q_ir`        | `emissivity_ir · earth_ir_flux · max(0, -n·e) · area` |
/// | `Q_viewfactor`| `sum_j A_i·ε_i·ε_j·σ·F_{ij}·(T_j⁴ − T_i⁴)` |
/// | `Q_emitted`   | `ε_i · σ · T_i⁴ · area` |
///
/// `s = env.sun_direction`, `e = env.earth_direction` (both unit vectors in
/// the structural frame pointing *from the source toward the vehicle*).
/// The `max(0, -n·s)` term is the cosine of the angle between the facet's
/// outward normal and the incoming flux, clamped at grazing.
///
/// Earth albedo is Sun-reflected broadband solar radiation, so it is
/// absorbed at the solar-band absorptivity (same coefficient as direct
/// solar). The separate [`ThermalFacet::albedo`] field is retained for
/// radiation-pressure reflectance modelling.
///
/// The view-factor coefficient `ε_j` is the absorbing facet's IR
/// absorptivity, which equals its emissivity by Kirchhoff's law. See
/// JEOD `thermal_facet_rider.cc:97` and the standard radiative network
/// formulation (Holman, *Heat Transfer*, 10th ed., §8-7).
///
/// # Panics
/// Panics if any slice length differs from `facets.len()`, or if
/// [`ViewFactorMatrix`] is not square (`view_factors.rows()[i].len() !=
/// view_factors.dim()` for any `i`). Callers are additionally expected to
/// have passed the matrix through [`ViewFactorMatrix::validate`] for
/// physical constraints (reciprocity, row-sum ≤ 1); this function does
/// not re-check those to avoid O(N²) work on every step.
pub fn compute_thermal_power_balance(
    facets: &[ThermalFacet],
    view_factors: &ViewFactorMatrix,
    env: &ThermalEnvironment,
    structural_normals: &[DVec3],
) -> ThermalPowerBalance {
    let n = facets.len();
    assert_eq!(
        structural_normals.len(),
        n,
        "structural_normals length must match facets length"
    );
    assert_eq!(
        view_factors.dim(),
        n,
        "view-factor matrix must be n-by-n where n = facets.len()"
    );
    for (i, row) in view_factors.rows().iter().enumerate() {
        assert_eq!(
            row.len(),
            n,
            "view-factor matrix row {i} has length {} but matrix dim is {n}",
            row.len()
        );
    }

    let mut temp_dots = vec![0.0_f64; n];
    let mut q_solar = vec![0.0_f64; n];
    let mut q_albedo = vec![0.0_f64; n];
    let mut q_ir = vec![0.0_f64; n];
    let mut q_viewfactor = vec![0.0_f64; n];
    let mut q_emitted = vec![0.0_f64; n];

    // Pre-compute T^4 so each facet's emission is evaluated once.
    let t_pow4: Vec<f64> = facets.iter().map(|f| f.temperature.powi(4)).collect();

    for (i, facet) in facets.iter().enumerate() {
        let n_i = structural_normals[i];

        // ── Direct solar absorption ──
        // cos θ_sun = -n · s   (s points Sun → vehicle, so a normal facing
        // the Sun has -n·s > 0).
        let cos_sun = (-n_i.dot(env.sun_direction)).max(0.0);
        q_solar[i] = facet.absorptivity_solar * env.solar_flux * cos_sun * facet.area;

        // ── Earth albedo (reflected solar) ──
        // Reflected solar is a solar-band flux, so it is absorbed at the
        // same solar-band absorptivity as direct sunlight.
        let cos_earth = (-n_i.dot(env.earth_direction)).max(0.0);
        q_albedo[i] = facet.absorptivity_solar * env.earth_albedo_flux * cos_earth * facet.area;

        // ── Earth thermal IR ──
        // Earth IR is broadband thermal radiation, absorbed at the IR
        // emissivity by Kirchhoff's law. Uses the same geometric factor as
        // albedo (facets facing Earth receive IR).
        q_ir[i] = facet.emissivity_ir * env.earth_ir_flux * cos_earth * facet.area;

        // ── Stefan-Boltzmann emission ──
        // JEOD_INV: IN.18 — power_emit must be non-negative
        q_emitted[i] = facet.radiative_constant() * t_pow4[i];
    }

    // ── Inter-facet radiative exchange via view factors ──
    // For each ordered pair (i, j) with F_ij > 0, facet i emits
    // A_i·ε_i·σ·F_ij·T_i⁴ toward facet j; a fraction α_j = ε_j (Kirchhoff)
    // is absorbed. By reciprocity, facet j likewise radiates
    // A_j·ε_j·σ·F_ji·T_j⁴ = A_i·ε_j·σ·F_ij·T_j⁴ toward facet i, of which
    // α_i = ε_i is absorbed at i. The net exchange flux on i from j is:
    //   A_i·σ·F_ij·ε_i·ε_j·(T_j⁴ − T_i⁴)
    // which is an even function of (i, j): opposite signs on the two
    // facets, conserving energy.
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let f_ij = view_factors.get(i, j);
            if f_ij <= 0.0 {
                continue;
            }
            // A_i * σ * F_ij * ε_i * ε_j
            let coeff = facets[i].area
                * STEFAN_BOLTZMANN
                * f_ij
                * facets[i].emissivity_ir
                * facets[j].emissivity_ir;
            q_viewfactor[i] += coeff * (t_pow4[j] - t_pow4[i]);
        }
    }

    // ── Temperature derivatives ──
    for i in 0..n {
        let c = facets[i].heat_capacity();
        if c > 0.0 {
            let q_in = q_solar[i] + q_albedo[i] + q_ir[i] + q_viewfactor[i];
            temp_dots[i] = (q_in - q_emitted[i]) / c;
        }
    }

    ThermalPowerBalance {
        temp_dots,
        q_solar,
        q_albedo,
        q_ir,
        q_viewfactor,
        q_emitted,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;

    fn simple_facet(area: f64, eps_ir: f64, abs_sol: f64, temp: f64) -> ThermalFacet {
        ThermalFacet {
            area,
            emissivity_ir: eps_ir,
            absorptivity_solar: abs_sol,
            albedo: 1.0 - abs_sol,
            conductivity: 0.0,
            specific_heat: 1000.0,
            mass: 1.0,
            temperature: temp,
        }
    }

    /// Isolated facet in direct sunlight: at equilibrium, the absorbed solar
    /// power equals the Stefan-Boltzmann emission, so
    /// `T_eq = (α · flux / (ε · σ))^(1/4)` and dT/dt = 0.
    #[test]
    fn thermal_equilibrium_no_coupling() {
        let eps = 0.9_f64;
        let alpha = 0.8_f64;
        let flux = 1361.0_f64;
        let t_eq = (alpha * flux / (eps * STEFAN_BOLTZMANN)).powf(0.25);

        let facet = ThermalFacet {
            area: 1.0,
            emissivity_ir: eps,
            absorptivity_solar: alpha,
            albedo: 1.0 - alpha,
            conductivity: 0.0,
            specific_heat: 1000.0,
            mass: 1.0,
            temperature: t_eq,
        };
        let view_factors = ViewFactorMatrix::new(vec![vec![0.0]]);
        let env = ThermalEnvironment {
            solar_flux: flux,
            earth_albedo_flux: 0.0,
            earth_ir_flux: 0.0,
            sun_direction: DVec3::X,
            earth_direction: -DVec3::Z,
        };
        // Facet normal pointing toward the Sun: -n·s = 1 (full illumination).
        let normals = [-DVec3::X];

        let balance = compute_thermal_power_balance(&[facet], &view_factors, &env, &normals);
        assert!(
            balance.temp_dots[0].abs() < 1e-9,
            "At equilibrium dT/dt ≈ 0, got {:e}",
            balance.temp_dots[0]
        );
        // And Q_in == Q_out directly.
        let q_in = balance.q_solar[0];
        assert!(
            (q_in - balance.q_emitted[0]).abs() < 1e-6,
            "Q_solar={q_in:e}, Q_emitted={:e}",
            balance.q_emitted[0]
        );
    }

    /// View-factor reciprocity constraint: A_i·F_ij = A_j·F_ji holds for a
    /// well-constructed matrix and is detected otherwise.
    #[test]
    fn view_factor_reciprocity() {
        // Two facets with A_1 = 2, A_2 = 1; set F_12 = 0.3 and enforce
        // reciprocity F_21 = A_1/A_2 · F_12 = 0.6.
        let areas = [2.0_f64, 1.0];
        let vf = ViewFactorMatrix::new(vec![vec![0.0, 0.3], vec![0.6, 0.0]]);
        vf.validate(&areas, TOL).expect("reciprocity holds");

        // Break reciprocity: F_21 should be 0.6, set to 0.5.
        let bad = ViewFactorMatrix::new(vec![vec![0.0, 0.3], vec![0.5, 0.0]]);
        match bad.validate(&areas, TOL) {
            Err(ViewFactorError::ReciprocityViolation { i, j, .. }) => {
                assert_eq!((i, j), (0, 1));
            }
            other => panic!("expected reciprocity violation, got {other:?}"),
        }
    }

    /// Row-sum ≤ 1: detects a matrix that sends > 100% of a facet's emission
    /// to other facets (impossible for a convex geometry).
    #[test]
    fn view_factor_completeness() {
        let areas = [1.0_f64, 1.0, 1.0];
        // Three-facet cavity: each row sums to ≤ 1. Symmetric F_ij = 0.3.
        let good = ViewFactorMatrix::new(vec![
            vec![0.0, 0.3, 0.3],
            vec![0.3, 0.0, 0.3],
            vec![0.3, 0.3, 0.0],
        ]);
        good.validate(&areas, TOL).expect("row sums <= 1");

        // Over-unity row: 0.6 + 0.6 = 1.2 > 1 for row 0.
        let bad = ViewFactorMatrix::new(vec![
            vec![0.0, 0.6, 0.6],
            vec![0.6, 0.0, 0.3],
            vec![0.6, 0.3, 0.0],
        ]);
        match bad.validate(&areas, TOL) {
            Err(ViewFactorError::RowSumExceedsUnity { sum, i }) => {
                assert_eq!(i, 0);
                assert!((sum - 1.2).abs() < TOL);
            }
            other => panic!("expected row-sum violation, got {other:?}"),
        }
    }

    /// Range check: negative or > 1 entries are rejected.
    #[test]
    fn view_factor_range_check() {
        let areas = [1.0_f64; 2];
        let bad = ViewFactorMatrix::new(vec![vec![0.0, -0.1], vec![-0.1, 0.0]]);
        assert!(matches!(
            bad.validate(&areas, TOL),
            Err(ViewFactorError::OutOfRange { .. })
        ));
    }

    /// Facets on the night side of Earth receive zero albedo contribution:
    /// a facet whose normal has the same sign as the Earth direction
    /// (i.e. pointing *away* from Earth, `-n·e < 0`) is shadowed.
    #[test]
    fn earth_albedo_zero_at_night() {
        let facet = simple_facet(2.0, 0.9, 0.8, 300.0);
        let vf = ViewFactorMatrix::new(vec![vec![0.0]]);
        let env = ThermalEnvironment {
            solar_flux: 0.0,
            earth_albedo_flux: 300.0,
            earth_ir_flux: 0.0,
            sun_direction: DVec3::X,
            // earth_direction points *from Earth toward vehicle*, so e = -Z
            // places the vehicle at -Z relative to Earth (Earth at +Z above
            // the vehicle); the flux travels along -Z, from the source down
            // to the vehicle.
            earth_direction: -DVec3::Z,
        };
        // A facet whose outward normal is along -Z points *away* from Earth
        // (into deep space below). Its back is turned to the reflected
        // solar flux, so it receives no albedo: -n·e = -(-Z)·(-Z) = -1,
        // clamped to 0 by max(0, ·).
        let night_normal = [-DVec3::Z];
        let balance = compute_thermal_power_balance(&[facet], &vf, &env, &night_normal);
        assert_eq!(
            balance.q_albedo[0], 0.0,
            "Night-side facet should receive zero albedo"
        );

        // And the Earth-facing normal *does* receive albedo:
        let day_normal = [DVec3::Z];
        let balance_day = compute_thermal_power_balance(&[facet], &vf, &env, &day_normal);
        assert!(balance_day.q_albedo[0] > 0.0);
    }

    /// Earth IR is roughly isotropic over the Earth disk: two facets with
    /// different orientations but both pointing generally at Earth receive
    /// IR proportional to cos(θ) (Lambert's law, implicit in the projected-
    /// area formula).
    #[test]
    fn earth_ir_scales_with_cosine() {
        let facet = simple_facet(1.0, 0.9, 0.0, 300.0);
        let vf = ViewFactorMatrix::new(vec![vec![0.0]]);
        let env = ThermalEnvironment {
            solar_flux: 0.0,
            earth_albedo_flux: 0.0,
            earth_ir_flux: 240.0,
            sun_direction: DVec3::X,
            // e = -Z: Earth at +Z (above), flux travels along -Z toward vehicle.
            earth_direction: -DVec3::Z,
        };
        // Normal along +Z points straight at Earth: -n·e = -(+Z)·(-Z) = 1.
        let straight_down = [DVec3::Z];
        let b1 = compute_thermal_power_balance(&[facet], &vf, &env, &straight_down);
        // Normal at 60° off-nadir: dot product cos(60) = 0.5
        let tilted = [DVec3::new(3.0_f64.sqrt() / 2.0, 0.0, 0.5)];
        let b2 = compute_thermal_power_balance(&[facet], &vf, &env, &tilted);
        assert!((b2.q_ir[0] / b1.q_ir[0] - 0.5).abs() < 1e-12);
    }

    /// Stefan-Boltzmann emission: Q_emit = ε · σ · A · T⁴.
    #[test]
    fn stefan_boltzmann_emission() {
        let eps = 0.8_f64;
        let area = 3.0_f64;
        let t = 350.0_f64;
        let facet = ThermalFacet {
            area,
            emissivity_ir: eps,
            absorptivity_solar: 0.0,
            albedo: 1.0,
            conductivity: 0.0,
            specific_heat: 1000.0,
            mass: 1.0,
            temperature: t,
        };
        let vf = ViewFactorMatrix::new(vec![vec![0.0]]);
        let env = ThermalEnvironment::default();
        let normals = [DVec3::Z];

        let balance = compute_thermal_power_balance(&[facet], &vf, &env, &normals);
        let expected = eps * STEFAN_BOLTZMANN * area * t.powi(4);
        assert!(
            (balance.q_emitted[0] - expected).abs() / expected < 1e-14,
            "Q_emit: expected {expected:e}, got {:e}",
            balance.q_emitted[0]
        );
    }

    /// Two facets connected by a view factor and at different temperatures:
    /// the hotter one loses net energy, the colder one gains net energy,
    /// and the magnitudes match by energy conservation.
    #[test]
    fn multi_facet_heat_transfer() {
        // Two identical facets with a view factor of 0.5 between them.
        let facets = vec![
            ThermalFacet {
                area: 1.0,
                emissivity_ir: 0.9,
                absorptivity_solar: 0.0,
                albedo: 1.0,
                conductivity: 0.0,
                specific_heat: 1000.0,
                mass: 1.0,
                temperature: 400.0, // hot
            },
            ThermalFacet {
                area: 1.0,
                emissivity_ir: 0.9,
                absorptivity_solar: 0.0,
                albedo: 1.0,
                conductivity: 0.0,
                specific_heat: 1000.0,
                mass: 1.0,
                temperature: 200.0, // cold
            },
        ];
        let vf = ViewFactorMatrix::new(vec![vec![0.0, 0.5], vec![0.5, 0.0]]);
        // No environmental input. Both normals along +Z (anti-parallel to
        // Earth direction — but both fluxes are zero anyway).
        let env = ThermalEnvironment {
            solar_flux: 0.0,
            earth_albedo_flux: 0.0,
            earth_ir_flux: 0.0,
            sun_direction: DVec3::X,
            earth_direction: -DVec3::Z,
        };
        let normals = [DVec3::Z, DVec3::Z];

        let balance = compute_thermal_power_balance(&facets, &vf, &env, &normals);

        // Q_vf_0 = A·σ·F·ε_0·ε_1·(T_1⁴ − T_0⁴) ; T_1 < T_0 → negative
        // Q_vf_1 = A·σ·F·ε_1·ε_0·(T_0⁴ − T_1⁴) ; equal magnitude, opposite sign
        assert!(
            balance.q_viewfactor[0] < 0.0,
            "hot facet loses energy via VF"
        );
        assert!(
            balance.q_viewfactor[1] > 0.0,
            "cold facet gains energy via VF"
        );
        assert!(
            (balance.q_viewfactor[0] + balance.q_viewfactor[1]).abs() < 1e-9,
            "view-factor exchange must conserve energy: {:e} + {:e}",
            balance.q_viewfactor[0],
            balance.q_viewfactor[1]
        );

        // Hot cools faster than it radiates alone; cold warms despite its
        // own Stefan-Boltzmann loss? Check: since both emit to space, we
        // compare the *relative* dT/dt.
        assert!(balance.temp_dots[0] < 0.0, "hot cools");
        // The cold facet's view-factor gain may or may not overcome its SB
        // emission depending on its temperature; just assert that adding
        // the view factor makes it warmer than without.
        let vf_none = ViewFactorMatrix::new(vec![vec![0.0, 0.0], vec![0.0, 0.0]]);
        let baseline = compute_thermal_power_balance(&facets, &vf_none, &env, &normals);
        assert!(
            balance.temp_dots[1] > baseline.temp_dots[1],
            "view-factor coupling should warm the cold facet relative to no-coupling"
        );
        assert!(
            balance.temp_dots[0] < baseline.temp_dots[0],
            "view-factor coupling should cool the hot facet relative to no-coupling"
        );
    }

    /// Uniform-temperature facets exchange zero net energy via view factors
    /// (the T⁴ − T⁴ term vanishes for every pair).
    #[test]
    fn view_factor_zero_at_equal_temperatures() {
        let t = 300.0_f64;
        let facets = vec![simple_facet(1.0, 0.9, 0.0, t); 3];
        let vf = ViewFactorMatrix::new(vec![
            vec![0.0, 0.3, 0.3],
            vec![0.3, 0.0, 0.3],
            vec![0.3, 0.3, 0.0],
        ]);
        let env = ThermalEnvironment {
            solar_flux: 0.0,
            earth_albedo_flux: 0.0,
            earth_ir_flux: 0.0,
            sun_direction: DVec3::X,
            earth_direction: -DVec3::Z,
        };
        let normals = [DVec3::Z; 3];

        let balance = compute_thermal_power_balance(&facets, &vf, &env, &normals);
        for (i, q) in balance.q_viewfactor.iter().enumerate() {
            assert!(q.abs() < 1e-12, "facet {i} view-factor flux = {q:e}");
        }
    }

    /// Back-facing solar facet: the normal makes an obtuse angle with -s,
    /// so the clamped cosine is 0 and Q_solar = 0.
    #[test]
    fn solar_backface_zero_absorption() {
        let facet = simple_facet(1.0, 0.9, 0.8, 300.0);
        let vf = ViewFactorMatrix::new(vec![vec![0.0]]);
        let env = ThermalEnvironment {
            solar_flux: 1361.0,
            earth_albedo_flux: 0.0,
            earth_ir_flux: 0.0,
            sun_direction: DVec3::X, // flux travels +X
            earth_direction: -DVec3::Z,
        };
        // Normal along +X: points with the flux (-n·s = -1 → clamp to 0).
        let normals = [DVec3::X];
        let balance = compute_thermal_power_balance(&[facet], &vf, &env, &normals);
        assert_eq!(balance.q_solar[0], 0.0);
    }

    /// Uniform off-diagonal matrix satisfies reciprocity for equal areas.
    #[test]
    fn uniform_offdiagonal_reciprocity() {
        let areas = vec![1.0_f64; 4];
        // With four equal-area facets sharing a cavity, F_ij = 1/3 for every
        // (i, j) with i ≠ j saturates the row-sum constraint and obeys
        // reciprocity.
        let vf = ViewFactorMatrix::uniform_offdiagonal(4, 1.0 / 3.0);
        vf.validate(&areas, TOL)
            .expect("uniform off-diagonal is valid");
        assert_eq!(vf.dim(), 4);
        assert_eq!(vf.get(0, 0), 0.0);
        assert!((vf.get(0, 1) - 1.0 / 3.0).abs() < TOL);
    }

    /// Non-illuminated facet with no internal sources cools purely by
    /// Stefan-Boltzmann emission: dT/dt < 0.
    #[test]
    fn facet_cools_in_deep_space() {
        let facet = simple_facet(1.0, 0.9, 0.0, 300.0);
        let vf = ViewFactorMatrix::new(vec![vec![0.0]]);
        let env = ThermalEnvironment {
            solar_flux: 0.0,
            earth_albedo_flux: 0.0,
            earth_ir_flux: 0.0,
            sun_direction: DVec3::X,
            earth_direction: -DVec3::Z,
        };
        let normals = [DVec3::Z];
        let balance = compute_thermal_power_balance(&[facet], &vf, &env, &normals);
        assert!(
            balance.temp_dots[0] < 0.0,
            "Facet in deep space should cool, got {}",
            balance.temp_dots[0]
        );
    }

    /// Heat-capacity convenience accessor: m·c_p and A·ε·σ.
    #[test]
    fn thermal_facet_accessors() {
        let f = ThermalFacet {
            area: 2.0,
            emissivity_ir: 0.5,
            absorptivity_solar: 0.7,
            albedo: 0.3,
            conductivity: 200.0,
            specific_heat: 900.0,
            mass: 5.0,
            temperature: 300.0,
        };
        assert!((f.heat_capacity() - 5.0 * 900.0).abs() < TOL);
        let expected_rad = 2.0 * 0.5 * STEFAN_BOLTZMANN;
        assert!((f.radiative_constant() - expected_rad).abs() < TOL * expected_rad);
    }

    /// Energy conservation across the whole network: for an isolated
    /// ensemble (no environment), the total view-factor flux over all
    /// facets sums to zero.
    #[test]
    fn network_energy_conservation_viewfactor() {
        let facets = vec![
            simple_facet(1.0, 0.9, 0.0, 400.0),
            simple_facet(2.0, 0.9, 0.0, 300.0),
            simple_facet(1.5, 0.9, 0.0, 350.0),
        ];
        // Build a reciprocal view-factor matrix. Pick F_ij for i<j and
        // derive F_ji = A_i/A_j · F_ij.
        let a = [1.0_f64, 2.0, 1.5];
        let f01 = 0.2;
        let f02 = 0.15;
        let f12 = 0.1;
        let vf = ViewFactorMatrix::new(vec![
            vec![0.0, f01, f02],
            vec![a[0] / a[1] * f01, 0.0, f12],
            vec![a[0] / a[2] * f02, a[1] / a[2] * f12, 0.0],
        ]);
        vf.validate(&a, TOL).expect("valid view-factor matrix");
        let env = ThermalEnvironment {
            solar_flux: 0.0,
            earth_albedo_flux: 0.0,
            earth_ir_flux: 0.0,
            sun_direction: DVec3::X,
            earth_direction: -DVec3::Z,
        };
        let normals = [DVec3::Z; 3];
        let balance = compute_thermal_power_balance(&facets, &vf, &env, &normals);
        let total_vf: f64 = balance.q_viewfactor.iter().sum();
        assert!(
            total_vf.abs() < 1e-9,
            "Network view-factor exchange must conserve energy, got {total_vf:e}"
        );
    }
}
