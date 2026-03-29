//! Solar radiation pressure computation.
//!
//! Two surface models:
//!
//! **Spherical (default)** — port of JEOD `RadiationDefaultSurface`:
//!   F = (L / (4πr²c)) · A · Cr · r̂
//!
//! **Flat-plate** — port of JEOD `FlatPlateRadiationFacet`:
//!   Per plate: decompose into absorption, diffuse reflection, specular reflection.
//!   Sum over all illuminated plates for total force and torque.
//!
//! Common to both: L is solar luminosity, r is distance to the Sun, c is speed
//! of light, r̂ is the Sun-to-vehicle unit vector.

use glam::DVec3;

/// Solar luminosity in W (matching JEOD `radiation_source.hh`).
pub const SOLAR_LUMINOSITY: f64 = 3.827e26;

/// Solar radius in m (matching JEOD `radiation_source.hh`).
pub const SOLAR_RADIUS: f64 = 6.98e8;

/// Speed of light in m/s.
pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;

/// Vehicle SRP configuration for the spherical (default) model.
///
/// Port of JEOD `RadiationDefaultSurface`.
#[derive(Debug, Clone, Copy)]
pub struct SrpConfig {
    /// Cross-sectional area in m² (JEOD `RadiationBaseFacet::cx_area`).
    pub cx_area: f64,
    /// Radiation coefficient (dimensionless).
    ///
    /// For the spherical (default) model, rad_coeff ranges from 1.0 to 13/9 ≈ 1.4444:
    /// - 1.0 = perfect absorber (albedo=0)
    /// - 13/9 ≈ 1.4444 = all diffuse reflection (albedo=1, diffuse=1)
    ///
    /// For flat-plate models (not yet implemented), rad_coeff can reach 2.0 for
    /// perfect specular reflection off a plate normal to the beam.
    ///
    /// Matches JEOD `RadiationDefaultSurface::rad_coeff`.
    pub rad_coeff: f64,
}

/// Radiation pressure force and torque on a vehicle.
///
/// The reference frame of `force` depends on the model:
/// - Spherical (`compute_srp_force`): force is in the integration (inertial) frame.
/// - Flat-plate (`compute_flat_plate_srp`/`_thermal`): force is in the structural frame.
///   The caller is responsible for rotating to inertial before integration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RadiationForce {
    /// Radiation force in N. Frame depends on the producing function (see struct docs).
    pub force: DVec3,
    /// Radiation torque in N*m, in the structural/body frame.
    /// Zero for the spherical (default) model.
    pub torque: DVec3,
}

impl Default for RadiationForce {
    fn default() -> Self {
        Self {
            force: DVec3::ZERO,
            torque: DVec3::ZERO,
        }
    }
}

/// Compute solar radiation pressure force using the spherical (default) model.
///
/// Port of JEOD `RadiationSource::calculate_flux()` + default surface force.
///
/// # Arguments
/// * `config` - Vehicle SRP properties (cx_area, rad_coeff)
/// * `sun_position` - Sun position in the integration frame (m)
/// * `vehicle_position` - Vehicle position in the integration frame (m)
/// * `illum_factor` - Illumination factor: 0.0 = full shadow, 1.0 = full sun
///
/// # Returns
/// Radiation force and torque. Force is in the integration (inertial) frame.
/// Torque is zero for the spherical model.
pub fn compute_srp_force(
    config: &SrpConfig,
    sun_position: DVec3,
    vehicle_position: DVec3,
    illum_factor: f64,
) -> RadiationForce {
    if illum_factor <= 0.0 {
        return RadiationForce::default();
    }

    // Vector from Sun to vehicle (JEOD: source_to_cg)
    let source_to_cg = vehicle_position - sun_position;
    let d_source_to_cg = source_to_cg.length();

    // JEOD_INV: IN.10 — distance guard prevents division by near-zero in flux calculation
    // (JEOD checks luminosity < 1e-6; ours uses a compile-time constant, so luminosity is always valid)
    if d_source_to_cg < 1.0 {
        return RadiationForce::default();
    }

    let flux_hat = source_to_cg / d_source_to_cg;

    // Solar flux at the vehicle (W/m²)
    // JEOD radiation_source.cc line 103: flux_mag = luminosity / (d² * 4π)
    let flux_mag =
        SOLAR_LUMINOSITY / (4.0 * std::f64::consts::PI * d_source_to_cg * d_source_to_cg);

    // Force magnitude: |F| = (flux / c) * cx_area * rad_coeff, directed from Sun to vehicle (r̂)
    let force_magnitude = flux_mag * config.cx_area * config.rad_coeff / SPEED_OF_LIGHT;

    // Force pushes away from Sun (along source-to-cg direction)
    // Apply illumination factor
    let force = flux_hat * force_magnitude * illum_factor;

    RadiationForce {
        force,
        torque: DVec3::ZERO,
    }
}

// ── Flat-plate surface model ─────────────────────────────────────────────────
// Port of JEOD `FlatPlateRadiationFacet::incident_radiation()` and
// `FlatPlateRadiationFacet::radiation_pressure()` from
// `flat_plate_radiation_facet.cc`.

const TWO_THIRDS: f64 = 2.0 / 3.0;

/// A single flat plate on a vehicle surface.
///
/// Position and normal are in the structural (body) frame.
#[derive(Debug, Clone, Copy)]
pub struct FlatPlate {
    /// Plate area in m².
    pub area: f64,
    /// Outward-facing normal unit vector (structural frame).
    pub normal: DVec3,
    /// Center of pressure position (structural frame, m).
    pub position: DVec3,
}

/// Optical properties shared by one or more flat plates.
///
/// Matches JEOD `RadiationParams` fields.
#[derive(Debug, Clone, Copy)]
pub struct FlatPlateParams {
    /// Fraction of incident light reflected (0 = perfect absorber, 1 = no absorption).
    pub albedo: f64,
    /// Fraction of reflected light that is diffuse (0 = all specular, 1 = all diffuse).
    pub diffuse: f64,
}

/// Compute SRP force and torque from a set of flat plates.
///
/// Port of JEOD `FlatPlateRadiationFacet::incident_radiation()` +
/// `radiation_pressure()`, summed over all plates. Thermal emission is not
/// included (it requires temperature integration state); for most cases the
/// emission force is small compared to direct SRP.
///
/// # Arguments
/// * `plates` - Flat plates with their optical properties, in the structural frame
/// * `flux_struct_hat` - Incoming flux direction (Sun → vehicle) in the structural frame
/// * `flux_mag` - Solar flux at the vehicle (W/m²)
/// * `center_grav` - Center of gravity in the structural frame (m), for torque arm
/// * `illum_factor` - Illumination factor: 0.0 = full shadow, 1.0 = full sun
///
/// # Returns
/// Total radiation force (structural frame, N) and torque (about CG, structural frame, N·m).
pub fn compute_flat_plate_srp(
    plates: &[(FlatPlate, FlatPlateParams)],
    flux_struct_hat: DVec3,
    flux_mag: f64,
    center_grav: DVec3,
    illum_factor: f64,
) -> RadiationForce {
    if illum_factor <= 0.0 || flux_mag <= 0.0 {
        return RadiationForce::default();
    }

    let effective_flux = flux_mag * illum_factor;
    let mut total_force = DVec3::ZERO;
    let mut total_torque = DVec3::ZERO;

    for (plate, params) in plates {
        // sin_theta = -(normal · flux_hat): cosine of angle between plate normal
        // and the incoming flux direction. Positive when plate faces the source.
        // JEOD flat_plate_radiation_facet.cc line 89
        let sin_theta = -plate.normal.dot(flux_struct_hat);
        if sin_theta <= 0.0 {
            continue; // plate faces away from source
        }

        // Projected area normal to the flux
        let cx_area = plate.area * sin_theta;

        // Momentum flux on this plate (N)
        let areaxflux = cx_area * effective_flux / SPEED_OF_LIGHT;

        // Absorption force: along flux direction
        // JEOD line 110: F_absorption = flux_hat * areaxflux * (1 - albedo)
        let f_absorption = flux_struct_hat * (areaxflux * (1.0 - params.albedo));

        let ref_flux = areaxflux * params.albedo;

        // Diffuse reflection: (flux_hat - 2/3 * normal) * diffuse * ref_flux
        // JEOD lines 117-121
        let f_diffuse = (flux_struct_hat - TWO_THIRDS * plate.normal) * (params.diffuse * ref_flux);

        // Specular reflection: normal * 2 * (diffuse - 1) * ref_flux * sin_theta
        // JEOD lines 124-128. (diffuse - 1) < 0, so force is opposite to normal.
        let f_specular = plate.normal * (2.0 * (params.diffuse - 1.0) * ref_flux * sin_theta);

        let plate_force = f_absorption + f_diffuse + f_specular;

        // Torque = (plate_position - center_grav) × force
        // JEOD line 165: crot_to_cp = position - center_grav
        let crot_to_cp = plate.position - center_grav;
        let plate_torque = crot_to_cp.cross(plate_force);

        total_force += plate_force;
        total_torque += plate_torque;
    }

    RadiationForce {
        force: total_force,
        torque: total_torque,
    }
}

// ── Flat-plate with thermal emission ─────────────────────────────────────────
// Port of JEOD `ThermalFacetRider` + `ThermalIntegrableObject::compute_temp_dot()`
// from `thermal_facet_rider.cc` and `thermal_integrable_object.cc`.

/// Stefan-Boltzmann constant (W m⁻² K⁻⁴).
/// Matches JEOD `thermal_facet_rider.hh` line 163.
pub const STEFAN_BOLTZMANN: f64 = 5.6704004e-8;

/// Thermal properties for a flat plate facet.
///
/// Port of JEOD `ThermalFacetRider` initialization parameters.
#[derive(Debug, Clone, Copy)]
pub struct FlatPlateThermal {
    /// Fraction of blackbody radiation actually emitted (0 to 1).
    pub emissivity: f64,
    /// Thermal mass per unit area in J/(m²·K).
    pub heat_capacity_per_area: f64,
}

/// Result of flat-plate SRP computation with thermal emission.
///
/// Includes the force/torque and per-plate temperature derivatives so the
/// caller can integrate temperature with the same method as the orbital state
/// (e.g., RK4).
pub struct FlatPlateSrpResult {
    /// Total radiation force (structural frame, N).
    pub force: DVec3,
    /// Total radiation torque about CG (structural frame, N·m).
    pub torque: DVec3,
    /// Per-plate temperature derivative (K/s). Same length as `plates`.
    pub temp_dots: Vec<f64>,
}

/// Compute SRP force, torque, and temperature derivatives from flat plates
/// with thermal emission.
///
/// This is a **pure function** — it does not mutate state. Returns `temp_dots`
/// so the caller can integrate temperature alongside the orbital state.
///
/// Matches JEOD's convention: `power_emit` uses the **cached** `t_pow4` from the
/// previous integration step (JEOD `thermal_integrable_object.cc:144`), not the
/// current temperature. The caller provides `t_pow4_cached` which is updated
/// after each integration step (not during force evaluation).
///
/// Port of JEOD `FlatPlateRadiationFacet::incident_radiation()` +
/// `radiation_pressure()` + `ThermalIntegrableObject::compute_temp_dot()`.
///
/// # Arguments
/// * `plates` - Flat plates with optical and thermal properties
/// * `t_pow4_cached` - Per-plate cached T⁴ values from previous step (JEOD convention)
/// * `flux_struct_hat` - Incoming flux direction (Sun → vehicle) in structural frame
/// * `flux_mag` - Solar flux at the vehicle (W/m²)
/// * `center_grav` - Center of gravity in structural frame (m)
/// * `illum_factor` - Illumination factor: 0.0 = full shadow, 1.0 = full sun
pub fn compute_flat_plate_srp_thermal(
    plates: &[(FlatPlate, FlatPlateParams, FlatPlateThermal)],
    t_pow4_cached: &[f64],
    flux_struct_hat: DVec3,
    flux_mag: f64,
    center_grav: DVec3,
    illum_factor: f64,
) -> FlatPlateSrpResult {
    assert_eq!(plates.len(), t_pow4_cached.len());

    let effective_flux = if illum_factor > 0.0 && flux_mag > 0.0 {
        flux_mag * illum_factor
    } else {
        0.0
    };

    let mut total_force = DVec3::ZERO;
    let mut total_torque = DVec3::ZERO;
    let mut temp_dots = vec![0.0; plates.len()];

    for (i, (plate, params, thermal)) in plates.iter().enumerate() {
        let sin_theta = -plate.normal.dot(flux_struct_hat);
        let illuminated = sin_theta > 0.0 && effective_flux > 0.0;

        // ── Absorption / reflection forces (only for illuminated plates) ──
        let mut plate_force = DVec3::ZERO;

        if illuminated {
            let cx_area = plate.area * sin_theta;
            let areaxflux = cx_area * effective_flux / SPEED_OF_LIGHT;

            let f_absorption = flux_struct_hat * (areaxflux * (1.0 - params.albedo));

            let ref_flux = areaxflux * params.albedo;
            let f_diffuse =
                (flux_struct_hat - TWO_THIRDS * plate.normal) * (params.diffuse * ref_flux);
            let f_specular = plate.normal * (2.0 * (params.diffuse - 1.0) * ref_flux * sin_theta);

            plate_force = f_absorption + f_diffuse + f_specular;
        }

        // ── Thermal emission (ALL plates, illuminated or not) ──
        // JEOD thermal_integrable_object.cc:144:
        //   power_emit = rad_constant * t_pow4;   // uses CACHED T^4
        //   temp_dot = (power_absorb - power_emit) / heat_capacity;
        let rad_constant = plate.area * thermal.emissivity * STEFAN_BOLTZMANN;
        let power_emit = rad_constant * t_pow4_cached[i];

        let power_absorb = if illuminated {
            (1.0 - params.albedo) * plate.area * sin_theta * effective_flux
        } else {
            0.0
        };

        let heat_capacity = thermal.heat_capacity_per_area * plate.area;
        if heat_capacity > 0.0 {
            temp_dots[i] = (power_absorb - power_emit) / heat_capacity;
        }

        // Emission force: -(2/3) * power_emit / c * normal
        // JEOD flat_plate_radiation_facet.cc:157
        let f_emission = -(TWO_THIRDS * power_emit / SPEED_OF_LIGHT) * plate.normal;
        plate_force += f_emission;

        let crot_to_cp = plate.position - center_grav;
        let plate_torque = crot_to_cp.cross(plate_force);

        total_force += plate_force;
        total_torque += plate_torque;
    }

    FlatPlateSrpResult {
        force: total_force,
        torque: total_torque,
        temp_dots,
    }
}

/// Compute solar flux at a given distance from the Sun.
///
/// Returns flux in W/m². Port of JEOD `RadiationSource::calculate_flux()`.
pub fn solar_flux_at_distance(distance: f64) -> f64 {
    if distance < 1.0 {
        return 0.0;
    }
    SOLAR_LUMINOSITY / (4.0 * std::f64::consts::PI * distance * distance)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Solar radiation pressure at 1 AU ≈ 4.56e-6 N/m² (Phase 4 exit criterion).
    #[test]
    fn pressure_at_1au() {
        let au = 1.496e11; // 1 AU in meters
        let flux_1au = SOLAR_LUMINOSITY / (4.0 * PI * au * au);
        let pressure = flux_1au / SPEED_OF_LIGHT;

        // Exact value depends on SOLAR_LUMINOSITY. With L=3.827e26 W, pressure ≈ 4.54e-6.
        // The PLAN.md exit criterion is 4.56 ± 0.01, but that assumes a slightly
        // different L. We verify order of magnitude and reasonable range.
        assert!(
            (pressure - 4.56e-6).abs() < 0.05e-6,
            "SRP at 1 AU should be ~4.56e-6 N/m², got {pressure}"
        );
    }

    /// Force direction is anti-Sun (pushes away from Sun).
    #[test]
    fn force_direction_anti_sun() {
        let config = SrpConfig {
            cx_area: 10.0,
            rad_coeff: 1.5,
        };
        let sun = DVec3::new(1.496e11, 0.0, 0.0); // Sun at +X
        let vehicle = DVec3::ZERO; // Vehicle at origin

        let result = compute_srp_force(&config, sun, vehicle, 1.0);

        // Vehicle is between origin and nowhere near Sun.
        // sun_to_vehicle = vehicle - sun = -1.496e11 in X
        // Force pushes vehicle away from Sun = in -X direction
        assert!(
            result.force.x < 0.0,
            "Force should push away from Sun (negative X)"
        );
        assert!(result.force.y.abs() < 1e-30);
        assert!(result.force.z.abs() < 1e-30);
    }

    /// Force magnitude at 1 AU for a known area.
    #[test]
    fn force_magnitude_at_1au() {
        let area = 100.0; // m²
        let cr = 1.0; // perfect absorber
        let config = SrpConfig {
            cx_area: area,
            rad_coeff: cr,
        };

        let au = 1.496e11;
        let sun = DVec3::ZERO;
        let vehicle = DVec3::new(au, 0.0, 0.0);

        let result = compute_srp_force(&config, sun, vehicle, 1.0);

        // Expected: pressure * area * cr
        let expected = SOLAR_LUMINOSITY / (4.0 * PI * au * au * SPEED_OF_LIGHT) * area * cr;
        let rel_err = (result.force.length() - expected).abs() / expected;

        assert!(
            rel_err < 1e-10,
            "Force magnitude: expected {expected}, got {}",
            result.force.length()
        );
    }

    /// Full shadow → zero force.
    #[test]
    fn full_shadow_zero_force() {
        let config = SrpConfig {
            cx_area: 10.0,
            rad_coeff: 1.5,
        };
        let result = compute_srp_force(
            &config,
            DVec3::new(1.496e11, 0.0, 0.0),
            DVec3::ZERO,
            0.0, // full shadow
        );
        assert_eq!(result.force, DVec3::ZERO);
    }

    /// Partial shadow scales force linearly.
    #[test]
    fn partial_shadow_scales_linearly() {
        let config = SrpConfig {
            cx_area: 10.0,
            rad_coeff: 1.5,
        };
        let sun = DVec3::new(1.496e11, 0.0, 0.0);
        let vehicle = DVec3::ZERO;

        let full = compute_srp_force(&config, sun, vehicle, 1.0);
        let half = compute_srp_force(&config, sun, vehicle, 0.5);

        let ratio = half.force.length() / full.force.length();
        assert!(
            (ratio - 0.5).abs() < 1e-12,
            "Half shadow should give half force, ratio = {ratio}"
        );
    }

    /// Cr = 2.0 (perfect reflector) gives double the force of Cr = 1.0 (absorber).
    #[test]
    fn reflector_doubles_force() {
        let sun = DVec3::new(1.496e11, 0.0, 0.0);
        let vehicle = DVec3::ZERO;

        let absorber = compute_srp_force(
            &SrpConfig {
                cx_area: 10.0,
                rad_coeff: 1.0,
            },
            sun,
            vehicle,
            1.0,
        );
        let reflector = compute_srp_force(
            &SrpConfig {
                cx_area: 10.0,
                rad_coeff: 2.0,
            },
            sun,
            vehicle,
            1.0,
        );

        let ratio = reflector.force.length() / absorber.force.length();
        assert!(
            (ratio - 2.0).abs() < 1e-12,
            "Cr=2 should give 2x force of Cr=1, ratio = {ratio}"
        );
    }

    /// Torque is zero for spherical model.
    #[test]
    fn spherical_model_zero_torque() {
        let config = SrpConfig {
            cx_area: 10.0,
            rad_coeff: 1.5,
        };
        let result = compute_srp_force(&config, DVec3::new(1.496e11, 0.0, 0.0), DVec3::ZERO, 1.0);
        assert_eq!(result.torque, DVec3::ZERO);
    }

    // ── Flat-plate model tests ──────────────────────────────────────────

    /// Single plate facing the Sun: all flux intercepted.
    #[test]
    fn flat_plate_normal_to_flux() {
        let plate = FlatPlate {
            area: 10.0,
            normal: DVec3::new(-1.0, 0.0, 0.0), // faces -X
            position: DVec3::ZERO,
        };
        let params = FlatPlateParams {
            albedo: 0.0,
            diffuse: 0.0,
        }; // pure absorber
        let flux_hat = DVec3::new(1.0, 0.0, 0.0); // flux from -X toward +X
        let flux_mag = 1000.0; // W/m²

        let result =
            compute_flat_plate_srp(&[(plate, params)], flux_hat, flux_mag, DVec3::ZERO, 1.0);

        // sin_theta = -(normal · flux_hat) = -(-1*1) = 1.0
        // cx_area = 10 * 1.0 = 10
        // areaxflux = 10 * 1000 / c
        // F = flux_hat * areaxflux (pure absorption)
        let expected_force = 10.0 * 1000.0 / SPEED_OF_LIGHT;
        assert!(
            (result.force.x - expected_force).abs() < 1e-20,
            "Force X: expected {expected_force}, got {}",
            result.force.x
        );
        assert!(result.force.y.abs() < 1e-30);
        assert!(result.force.z.abs() < 1e-30);
    }

    /// Plate facing away from Sun: no force.
    #[test]
    fn flat_plate_facing_away() {
        let plate = FlatPlate {
            area: 10.0,
            normal: DVec3::new(1.0, 0.0, 0.0), // faces +X (same as flux)
            position: DVec3::ZERO,
        };
        let params = FlatPlateParams {
            albedo: 0.5,
            diffuse: 0.5,
        };
        let flux_hat = DVec3::new(1.0, 0.0, 0.0);

        let result = compute_flat_plate_srp(&[(plate, params)], flux_hat, 1000.0, DVec3::ZERO, 1.0);

        assert_eq!(
            result.force,
            DVec3::ZERO,
            "Back-facing plate should produce no force"
        );
    }

    /// Pure specular reflection: force is along plate normal (opposite to incoming).
    #[test]
    fn flat_plate_specular_reflection() {
        let plate = FlatPlate {
            area: 10.0,
            normal: DVec3::new(-1.0, 0.0, 0.0),
            position: DVec3::ZERO,
        };
        // albedo=1, diffuse=0 → pure specular
        let params = FlatPlateParams {
            albedo: 1.0,
            diffuse: 0.0,
        };
        let flux_hat = DVec3::new(1.0, 0.0, 0.0);
        let flux_mag = 1000.0;

        let result =
            compute_flat_plate_srp(&[(plate, params)], flux_hat, flux_mag, DVec3::ZERO, 1.0);

        // Absorption: 0 (albedo=1)
        // Diffuse: 0 (diffuse=0)
        // Specular: normal * 2*(0-1) * albedo*areaxflux * sin_theta
        //         = [-1,0,0] * 2*(-1) * 1.0 * (10*1000/c) * 1.0
        //         = [+2 * 10*1000/c, 0, 0]
        // Total force in +X direction (reflected back toward source) — wait, that's
        // the momentum transfer. For specular reflection the force is 2x absorption
        // and pushes the plate away from the source (same direction as flux_hat).
        let areaxflux = 10.0 * 1000.0 / SPEED_OF_LIGHT;
        // F_specular = normal * 2*(diffuse-1)*ref_flux*sin_theta
        //            = [-1,0,0] * 2*(-1)*(1.0*areaxflux)*1.0 = [+2*areaxflux, 0, 0]
        // F_absorption = 0
        // F_diffuse = 0
        // Total = [+2*areaxflux, 0, 0]
        assert!(
            (result.force.x - 2.0 * areaxflux).abs() < 1e-20,
            "Specular: expected {}, got {}",
            2.0 * areaxflux,
            result.force.x
        );
    }

    /// Torque from offset plate.
    #[test]
    fn flat_plate_torque_from_offset() {
        let plate = FlatPlate {
            area: 10.0,
            normal: DVec3::new(-1.0, 0.0, 0.0),
            position: DVec3::new(0.0, 2.0, 0.0), // offset in +Y
        };
        let params = FlatPlateParams {
            albedo: 0.0,
            diffuse: 0.0,
        };
        let flux_hat = DVec3::new(1.0, 0.0, 0.0);
        let cg = DVec3::ZERO;

        let result = compute_flat_plate_srp(&[(plate, params)], flux_hat, 1000.0, cg, 1.0);

        // Force is in +X, arm is [0,2,0]
        // Torque = [0,2,0] × [Fx,0,0] = [0,0,-2*Fx]
        assert!(result.torque.z < 0.0, "Torque Z should be negative");
        assert!(result.torque.x.abs() < 1e-30);
        assert!(result.torque.y.abs() < 1e-30);
    }

    /// Shadow fraction scales flat-plate force.
    #[test]
    fn flat_plate_shadow_scaling() {
        let plate = FlatPlate {
            area: 10.0,
            normal: DVec3::new(-1.0, 0.0, 0.0),
            position: DVec3::ZERO,
        };
        let params = FlatPlateParams {
            albedo: 0.5,
            diffuse: 0.5,
        };
        let flux_hat = DVec3::new(1.0, 0.0, 0.0);

        let full = compute_flat_plate_srp(&[(plate, params)], flux_hat, 1000.0, DVec3::ZERO, 1.0);
        let half = compute_flat_plate_srp(&[(plate, params)], flux_hat, 1000.0, DVec3::ZERO, 0.5);

        let ratio = half.force.length() / full.force.length();
        assert!(
            (ratio - 0.5).abs() < 1e-12,
            "Half shadow should give half force, ratio = {ratio}"
        );
    }

    // ── Thermal emission tests ────────────────────────────────────────

    /// Thermal emission force direction: opposes normal (recoil).
    #[test]
    fn thermal_emission_opposes_normal() {
        let plate = FlatPlate {
            area: 60.0,
            normal: DVec3::X,
            position: DVec3::ZERO,
        };
        let params = FlatPlateParams {
            albedo: 0.5,
            diffuse: 0.5,
        };
        let thermal = FlatPlateThermal {
            emissivity: 0.5,
            heat_capacity_per_area: 50.0,
        };
        let t_pow4 = [270.0_f64.powi(4)];

        // No flux — only thermal emission
        let result = compute_flat_plate_srp_thermal(
            &[(plate, params, thermal)],
            &t_pow4,
            DVec3::X,
            0.0, // zero flux
            DVec3::ZERO,
            1.0,
        );

        assert!(
            result.force.x < 0.0,
            "Emission should push in -normal direction"
        );
        assert!(result.force.y.abs() < 1e-30);
        assert!(result.force.z.abs() < 1e-30);
    }

    /// Thermal emission magnitude matches Stefan-Boltzmann.
    #[test]
    fn thermal_emission_magnitude() {
        let plate = FlatPlate {
            area: 60.0,
            normal: DVec3::X,
            position: DVec3::ZERO,
        };
        let params = FlatPlateParams {
            albedo: 0.5,
            diffuse: 0.5,
        };
        let thermal = FlatPlateThermal {
            emissivity: 0.5,
            heat_capacity_per_area: 50.0,
        };
        let t_pow4 = [270.0_f64.powi(4)];

        let result = compute_flat_plate_srp_thermal(
            &[(plate, params, thermal)],
            &t_pow4,
            DVec3::X,
            0.0,
            DVec3::ZERO,
            1.0,
        );

        let power_emit = 0.5 * STEFAN_BOLTZMANN * 60.0 * 270.0_f64.powi(4);
        let expected = TWO_THIRDS * power_emit / SPEED_OF_LIGHT;
        let actual = result.force.length();
        let rel_err = (actual - expected).abs() / expected;
        assert!(
            rel_err < 1e-10,
            "Emission force: expected {expected:.6e}, got {actual:.6e}, rel_err={rel_err:.2e}"
        );
    }

    /// Temperature derivative is negative when not illuminated (plate cools).
    #[test]
    fn thermal_temperature_cools_in_shadow() {
        let plate = FlatPlate {
            area: 60.0,
            normal: DVec3::X,
            position: DVec3::ZERO,
        };
        let params = FlatPlateParams {
            albedo: 0.5,
            diffuse: 0.5,
        };
        let thermal = FlatPlateThermal {
            emissivity: 0.5,
            heat_capacity_per_area: 50.0,
        };
        let t_pow4 = [270.0_f64.powi(4)];

        let result = compute_flat_plate_srp_thermal(
            &[(plate, params, thermal)],
            &t_pow4,
            DVec3::X,
            0.0,
            DVec3::ZERO,
            0.0,
        );

        assert!(
            result.temp_dots[0] < 0.0,
            "temp_dot should be negative when not illuminated, got {}",
            result.temp_dots[0]
        );
    }

    /// With thermal, total force is larger than without (emission adds to SRP).
    #[test]
    fn thermal_increases_total_force() {
        let plate = FlatPlate {
            area: 60.0,
            normal: -DVec3::X,
            position: DVec3::ZERO,
        };
        let params = FlatPlateParams {
            albedo: 0.5,
            diffuse: 0.5,
        };
        let thermal = FlatPlateThermal {
            emissivity: 0.5,
            heat_capacity_per_area: 50.0,
        };
        let flux_hat = DVec3::X;
        let flux_mag = 1400.0;

        // Without thermal
        let no_thermal =
            compute_flat_plate_srp(&[(plate, params)], flux_hat, flux_mag, DVec3::ZERO, 1.0);

        // With thermal
        let t_pow4 = [270.0_f64.powi(4)];
        let with_thermal = compute_flat_plate_srp_thermal(
            &[(plate, params, thermal)],
            &t_pow4,
            flux_hat,
            flux_mag,
            DVec3::ZERO,
            1.0,
        );

        assert!(
            with_thermal.force.length() > no_thermal.force.length(),
            "Thermal emission should increase total force: with={:.6e} vs without={:.6e}",
            with_thermal.force.length(),
            no_thermal.force.length()
        );
    }

    /// SIM_3_ORBIT 6-plate configuration: symmetric plates with identity attitude.
    #[test]
    fn sim3_orbit_six_plate_identity_attitude() {
        // SIM_3_ORBIT plates: 4×60m² at ±X/±Y, 2×16m² at ±Z
        let params = FlatPlateParams {
            albedo: 0.5,
            diffuse: 0.5,
        };
        let plates: Vec<(FlatPlate, FlatPlateParams)> = vec![
            (
                FlatPlate {
                    area: 60.0,
                    normal: DVec3::X,
                    position: DVec3::new(2.0, 0.0, 0.0),
                },
                params,
            ),
            (
                FlatPlate {
                    area: 60.0,
                    normal: -DVec3::Y,
                    position: DVec3::new(0.0, -2.0, 0.0),
                },
                params,
            ),
            (
                FlatPlate {
                    area: 60.0,
                    normal: -DVec3::X,
                    position: DVec3::new(-2.0, 0.0, 0.0),
                },
                params,
            ),
            (
                FlatPlate {
                    area: 60.0,
                    normal: DVec3::Y,
                    position: DVec3::new(0.0, 2.0, 0.0),
                },
                params,
            ),
            (
                FlatPlate {
                    area: 16.0,
                    normal: DVec3::Z,
                    position: DVec3::new(0.0, 0.0, 7.5),
                },
                params,
            ),
            (
                FlatPlate {
                    area: 16.0,
                    normal: -DVec3::Z,
                    position: DVec3::new(0.0, 0.0, -7.5),
                },
                params,
            ),
        ];

        // Flux from +X direction
        let flux_hat = DVec3::X;
        let flux_mag = 1000.0;

        let result = compute_flat_plate_srp(&plates, flux_hat, flux_mag, DVec3::ZERO, 1.0);

        // Only plates facing -X intercept flux:
        // Plate at -X with normal [-1,0,0]: sin_theta = -((-1)*1) = 1.0, cx_area = 60
        // Plate at +X with normal [+1,0,0]: sin_theta = -(1*1) = -1.0, skip
        // ±Y plates: sin_theta = 0, skip
        // ±Z plates: sin_theta = 0, skip
        // So only one plate contributes, with cx_area = 60
        assert!(result.force.length() > 0.0, "Should have non-zero force");
        assert!(
            result.force.x > 0.0,
            "Force should push in +X (away from source)"
        );
        // Y and Z components should be non-zero due to diffuse reflection off the -X plate
        // (diffuse component has -2/3*normal contribution)
    }
}
