//! Thermal-rider model: per-facet environmental heating and Stefan-Boltzmann
//! emission.
//!
//! Port of JEOD `models/interactions/thermal_rider/` with the following scope:
//!
//! * [`ThermalFacet`] — per-facet thermal and optical material properties
//!   (emissivity, solar absorptivity, albedo, conductivity, specific heat,
//!   mass, current temperature). Mirrors JEOD `ThermalFacetRider` +
//!   `ThermalParams` combined into a single Rust struct.
//! * [`ThermalEnvironment`] — aggregated environmental fluxes (solar,
//!   Earth albedo, planet thermal IR) and their directions in the vehicle
//!   structural frame.
//! * [`compute_thermal_power_balance`] — pure function that returns the
//!   time derivative of temperature for each facet given its absorbed and
//!   emitted powers.
//!
//! JEOD's thermal rider is deliberately minimal: each facet uses its full
//! `A·ε·σ·T⁴` Stefan-Boltzmann emission with no inter-facet radiative
//! coupling (see `thermal_integrable_object.cc:144` —
//! `power_emit = rad_constant * t_pow4`). Conduction is "for future
//! implementation" (`thermal_facet_rider.cc:61-90`). All JEOD verification
//! tests for this module are structural (`verif/unit_tests/*_ut.cc` just
//! construct/destruct instances). This port matches JEOD exactly: each
//! facet radiates freely into space at its full Stefan-Boltzmann rate, and
//! environmental absorption is driven by the solar-band, IR-band, and
//! albedo fluxes supplied in [`ThermalEnvironment`].
//!
//! The Stefan-Boltzmann constant is shared with
//! [`crate::radiation_pressure::STEFAN_BOLTZMANN`].

use glam::DVec3;

use crate::radiation_pressure::STEFAN_BOLTZMANN;

/// Thermal and optical material properties for a single facet.
///
/// Combines the material-dependent parameters represented here from JEOD
/// `ThermalParams` (emissivity, heat-capacity-related quantities) with the
/// surface-interaction optical properties (solar absorptivity,
/// albedo/reflectance) used by radiative environmental heating. JEOD's
/// `ThermalParams::thermal_power_dump` (internal heat source, W) is not
/// modeled here — add it in a follow-up if a use case demands it.
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
    /// Solar albedo / solar reflectance (0..=1). Fraction of incident
    /// solar-band flux *reflected* by this facet — complement of
    /// `absorptivity_solar` for an opaque surface (i.e.
    /// `absorptivity_solar + albedo ≈ 1`), exposed as a separate field so
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

    /// Stefan-Boltzmann radiative constant `A·ε·σ` (W/K⁴) used by this
    /// module's temperature ODE for free-space thermal emission.
    #[inline]
    pub fn radiative_constant(&self) -> f64 {
        self.area * self.emissivity_ir * STEFAN_BOLTZMANN
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
/// * `solar_flux` uses `sun_direction` (direct sunlight arrives from the
///   Sun along `sun_direction`, Sun → vehicle).
/// * `earth_albedo_flux` uses `earth_direction` (reflected sunlight arrives
///   from Earth along `earth_direction`, Earth → vehicle).
/// * `earth_ir_flux` uses `earth_direction` (thermal IR also arrives from
///   Earth along `earth_direction`, Earth → vehicle).
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
/// The fields `q_solar`, `q_albedo`, `q_ir`, and `q_emitted` are returned
/// so callers can trace which environmental term dominates the facet's
/// thermal budget, or log the energy balance at a given timestep.
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
    /// Stefan-Boltzmann emission power per facet (W). Non-negative for
    /// physically valid inputs (non-negative `area` and `emissivity_ir`,
    /// finite `temperature`).
    pub q_emitted: Vec<f64>,
}

/// Compute the per-facet temperature time derivative dT/dt from the
/// environmental absorption and Stefan-Boltzmann emission terms.
///
/// The temperature ODE per facet is
///
/// ```text
/// m·c_p · dT_i/dt = Q_solar_i + Q_albedo_i + Q_ir_i - Q_emitted_i
/// ```
///
/// where, for facet `i` with outward structural-frame normal `n_i`:
///
/// | Term          | Formula |
/// |---------------|---------|
/// | `Q_solar`     | `absorptivity_solar · solar_flux · max(0, -n·s) · area` |
/// | `Q_albedo`    | `absorptivity_solar · earth_albedo_flux · max(0, -n·e) · area` |
/// | `Q_ir`        | `emissivity_ir · earth_ir_flux · max(0, -n·e) · area` |
/// | `Q_emitted`   | `ε_i · σ · T_i⁴ · area` |
///
/// `s = env.sun_direction`, `e = env.earth_direction` (both unit vectors in
/// the structural frame pointing *from the source toward the vehicle*),
/// and each `n_i` in `structural_normals` is the facet's outward-pointing
/// **unit** normal in the same frame. The `max(0, -n·s)` term is the
/// cosine of the angle between the facet's outward normal and the incoming
/// flux, clamped at grazing; it is only a cosine if the normals are unit
/// vectors, so callers must normalize before passing them in. Both debug
/// and release builds assert this — see the `# Panics` section below.
///
/// Earth albedo is Sun-reflected broadband solar radiation, so it is
/// absorbed at the solar-band absorptivity (same coefficient as direct
/// solar). The separate [`ThermalFacet::albedo`] field is retained for
/// radiation-pressure reflectance modelling.
///
/// This matches JEOD `thermal_integrable_object.cc:144`, which emits at
/// the full `rad_constant · T⁴` rate per facet with no inter-facet
/// radiative coupling. A spacecraft that needs a cavity / view-factor
/// radiosity network should implement it as a separate model on top of
/// this per-facet balance.
///
/// # Panics
/// - `structural_normals.len() != facets.len()`.
/// - `env.sun_direction` or `env.earth_direction` is not a unit vector
///   (within `1e-6` of `|v|² = 1`).
/// - any `structural_normals[i]` is not a unit vector.
///
/// The unit-vector preconditions panic in both debug **and release** builds
/// (#485 H5): the `max(0, -n·s)` cosine term only carries physical meaning
/// when the inputs are normalized, and silently propagating a non-unit
/// normal would corrupt the solar / albedo / IR power balance.
pub fn compute_thermal_power_balance(
    facets: &[ThermalFacet],
    env: &ThermalEnvironment,
    structural_normals: &[DVec3],
) -> ThermalPowerBalance {
    let n = facets.len();
    assert_eq!(
        structural_normals.len(),
        n,
        "structural_normals length must match facets length"
    );
    assert!(
        (env.sun_direction.length_squared() - 1.0).abs() < 1e-6,
        "env.sun_direction must be a unit vector (|s|² = {})",
        env.sun_direction.length_squared()
    );
    assert!(
        (env.earth_direction.length_squared() - 1.0).abs() < 1e-6,
        "env.earth_direction must be a unit vector (|e|² = {})",
        env.earth_direction.length_squared()
    );

    let mut temp_dots = vec![0.0_f64; n];
    let mut q_solar = vec![0.0_f64; n];
    let mut q_albedo = vec![0.0_f64; n];
    let mut q_ir = vec![0.0_f64; n];
    let mut q_emitted = vec![0.0_f64; n];

    for (i, facet) in facets.iter().enumerate() {
        let n_i = structural_normals[i];
        assert!(
            (n_i.length_squared() - 1.0).abs() < 1e-6,
            "structural_normals[{i}] must be a unit vector (|n|² = {})",
            n_i.length_squared()
        );

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
        q_emitted[i] = facet.radiative_constant() * facet.temperature.powi(4);

        let c = facet.heat_capacity();
        if c > 0.0 {
            let q_in = q_solar[i] + q_albedo[i] + q_ir[i];
            temp_dots[i] = (q_in - q_emitted[i]) / c;
        }
    }

    ThermalPowerBalance {
        temp_dots,
        q_solar,
        q_albedo,
        q_ir,
        q_emitted,
    }
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "thermal-state tests assert bit-exact recovery of literal-built facet temperatures"
)]
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
        let env = ThermalEnvironment {
            solar_flux: flux,
            earth_albedo_flux: 0.0,
            earth_ir_flux: 0.0,
            sun_direction: DVec3::X,
            earth_direction: -DVec3::Z,
        };
        // Facet normal pointing toward the Sun: -n·s = 1 (full illumination).
        let normals = [-DVec3::X];

        let balance = compute_thermal_power_balance(&[facet], &env, &normals);
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

    /// Facets on the night side of Earth receive zero albedo contribution:
    /// a facet whose normal has the same sign as the Earth direction
    /// (i.e. pointing *away* from Earth, `-n·e < 0`) is shadowed.
    #[test]
    fn earth_albedo_zero_at_night() {
        let facet = simple_facet(2.0, 0.9, 0.8, 300.0);
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
        let balance = compute_thermal_power_balance(&[facet], &env, &night_normal);
        assert_eq!(
            balance.q_albedo[0], 0.0,
            "Night-side facet should receive zero albedo"
        );

        // And the Earth-facing normal *does* receive albedo:
        let day_normal = [DVec3::Z];
        let balance_day = compute_thermal_power_balance(&[facet], &env, &day_normal);
        assert!(balance_day.q_albedo[0] > 0.0);
    }

    /// Earth IR is roughly isotropic over the Earth disk: two facets with
    /// different orientations but both pointing generally at Earth receive
    /// IR proportional to cos(θ) (Lambert's law, implicit in the projected-
    /// area formula).
    #[test]
    fn earth_ir_scales_with_cosine() {
        let facet = simple_facet(1.0, 0.9, 0.0, 300.0);
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
        let b1 = compute_thermal_power_balance(&[facet], &env, &straight_down);
        // Normal at 60° off-nadir: dot product cos(60) = 0.5
        let tilted = [DVec3::new(3.0_f64.sqrt() / 2.0, 0.0, 0.5)];
        let b2 = compute_thermal_power_balance(&[facet], &env, &tilted);
        assert!((b2.q_ir[0] / b1.q_ir[0] - 0.5).abs() < 1e-12);
    }

    /// Stefan-Boltzmann emission: Q_emit = ε · σ · A · T⁴ (matches JEOD
    /// `thermal_integrable_object.cc:144`).
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
        let env = ThermalEnvironment::default();
        let normals = [DVec3::Z];

        let balance = compute_thermal_power_balance(&[facet], &env, &normals);
        let expected = eps * STEFAN_BOLTZMANN * area * t.powi(4);
        assert!(
            (balance.q_emitted[0] - expected).abs() / expected < 1e-14,
            "Q_emit: expected {expected:e}, got {:e}",
            balance.q_emitted[0]
        );
    }

    /// Back-facing solar facet: the normal makes an obtuse angle with -s,
    /// so the clamped cosine is 0 and Q_solar = 0.
    #[test]
    fn solar_backface_zero_absorption() {
        let facet = simple_facet(1.0, 0.9, 0.8, 300.0);
        let env = ThermalEnvironment {
            solar_flux: 1361.0,
            earth_albedo_flux: 0.0,
            earth_ir_flux: 0.0,
            sun_direction: DVec3::X, // flux travels +X
            earth_direction: -DVec3::Z,
        };
        // Normal along +X: points with the flux (-n·s = -1 → clamp to 0).
        let normals = [DVec3::X];
        let balance = compute_thermal_power_balance(&[facet], &env, &normals);
        assert_eq!(balance.q_solar[0], 0.0);
    }

    /// Non-illuminated facet with no internal sources cools purely by
    /// Stefan-Boltzmann emission: dT/dt < 0.
    #[test]
    fn facet_cools_in_deep_space() {
        let facet = simple_facet(1.0, 0.9, 0.0, 300.0);
        let env = ThermalEnvironment {
            solar_flux: 0.0,
            earth_albedo_flux: 0.0,
            earth_ir_flux: 0.0,
            sun_direction: DVec3::X,
            earth_direction: -DVec3::Z,
        };
        let normals = [DVec3::Z];
        let balance = compute_thermal_power_balance(&[facet], &env, &normals);
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
}
